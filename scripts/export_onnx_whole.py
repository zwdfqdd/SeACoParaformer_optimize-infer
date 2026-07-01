"""
SeACo-Paraformer 模型 ONNX 导出脚本

流程：
1. 使用 FunASR AutoModel.export() 导出 fp32 ONNX
2. 使用 onnxconverter-common mixed precision 转为 fp16
   - keep_io_types=True 保持输入输出为 fp32
   - op_block_list 保留精度敏感算子为 fp32
3. opset_version=17
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 整体模型 Wrapper（含热词 bias_embed 输入）
# ============================================================
class WholeModelWithBiasWrapper(nn.Module):
    """整体模型导出 wrapper（encoder + predictor + decoder + SeACo）。

    输入：
        speech (B, T, feat_dim)
        speech_lengths (B,) int64
        bias_embed (B, H, D) — 热词编码（来自 model_eb.onnx）；无热词时传 (B, 1, D) 全零

    输出：
        logits (B, N, vocab) — log_softmax 合并后
        token_num (B,)

    SeACo 解码逻辑与分段导出 DecoderWithSeACoWrapper 完全一致（直接吃 bias_embed 张量，
    不在图内做 ASF 过滤；ASF/Top256 截断由外部完成）。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder = model.decoder
        self.seaco_decoder = model.seaco_decoder
        self.hotword_output_layer = model.hotword_output_layer
        self.NO_BIAS = model.NO_BIAS
        self.seaco_weight = model.seaco_weight

    def _calc_predictor_vectorized(self, encoder_out, encoder_out_lens):
        """向量化 CIF predictor（无 for 循环，支持 batch>1）。

        与分段导出 CIFWrapper 逻辑完全一致：避免原始 cif() 的
        `for b in range(batch_size)` 在 batch=1 导出时被固化，
        导致 batch>1 推理时 `index out of bounds`。
        """
        from seaco_paraformer.utils import make_pad_mask
        from seaco_paraformer.predictor import cif_v1_export

        pred = self.model.predictor
        h = encoder_out
        b, t, d = h.shape

        # mask: (B, 1, T)
        mask = (~make_pad_mask(encoder_out_lens, maxlen=t)[:, None, :]).to(h.device)

        # 1. CIF conv → relu → cif_output → sigmoid → relu(smooth)
        context = h.transpose(1, 2)
        queries = pred.pad(context)
        output = torch.relu(pred.cif_conv1d(queries))
        output_t = output.transpose(1, 2)
        cif_logit = pred.cif_output(output_t)
        alphas = torch.sigmoid(cif_logit)
        alphas = torch.nn.functional.relu(alphas * pred.smooth_factor - pred.noise_threshold)

        # 2. mask
        mask_t = mask.transpose(-1, -2).float()  # (B, T, 1)
        alphas = alphas * mask_t
        alphas = alphas.squeeze(-1)  # (B, T)
        mask_squeezed = mask_t.squeeze(-1)  # (B, T)

        # 3. tail_process
        zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
        ones_t = torch.ones_like(zeros_t)
        mask_1 = torch.cat([mask_squeezed, zeros_t], dim=1)
        mask_2 = torch.cat([ones_t, mask_squeezed], dim=1)
        tail_mask = mask_2 - mask_1
        tail_threshold = tail_mask * pred.tail_threshold
        alphas = torch.cat([alphas, zeros_t], dim=1)
        alphas = alphas + tail_threshold

        zeros_hidden = torch.zeros((b, 1, d), dtype=h.dtype, device=h.device)
        hidden = torch.cat([h, zeros_hidden], dim=1)
        token_num = alphas.sum(dim=-1)

        # 4. CIF 核心（向量化）
        acoustic_embeds, _ = cif_v1_export(hidden, alphas, pred.threshold)
        return acoustic_embeds, token_num

    def forward(self, speech, speech_lengths, bias_embed):
        # encoder + 向量化 predictor（无 Loop，支持 batch>1）
        encoder_out, encoder_out_lens = self.model.encode(speech, speech_lengths)
        acoustic_embeds, token_num = self._calc_predictor_vectorized(
            encoder_out, encoder_out_lens
        )
        pre_token_length = token_num.round().long()

        # 主 decoder（return logits + hidden）
        decoder_out, decoder_hidden, _ = self.decoder(
            encoder_out, encoder_out_lens,
            acoustic_embeds, pre_token_length,
            return_hidden=True, return_both=True,
        )
        decoder_pred = torch.log_softmax(decoder_out, dim=-1)

        # SeACo decoder：bias_embed 作为 memory
        B, H, D = bias_embed.shape
        contextual_length = torch.full(
            (B,), H, dtype=torch.long, device=bias_embed.device
        )

        cif_attended, _ = self.seaco_decoder(
            bias_embed, contextual_length,
            acoustic_embeds, pre_token_length,
        )
        dec_attended, _ = self.seaco_decoder(
            bias_embed, contextual_length,
            decoder_hidden, pre_token_length,
        )

        merged = cif_attended + dec_attended
        dha_output = self.hotword_output_layer(merged)
        dha_pred = torch.log_softmax(dha_output, dim=-1)

        # NO_BIAS mask 合并
        lmbd = self.seaco_weight
        a = (1.0 - lmbd) / lmbd
        b = 1.0 / lmbd
        dha_ids = dha_pred.max(-1)[1]
        dha_mask = (dha_ids == self.NO_BIAS).int().unsqueeze(-1).float()
        dha_mask_scaled = (dha_mask + a) / b

        final_logits = decoder_pred * dha_mask_scaled + dha_pred * (1.0 - dha_mask_scaled)
        return final_logits, token_num


# ============================================================
# Bias Encoder Wrapper（输出已按热词长度取最后有效时间步 → (H, D)）
# ============================================================
class BiasEncoderWrapper(nn.Module):
    """Bias Encoder 导出 wrapper（整体模型配套）。

    输入：
        hotword (H, L) int64 — H 个热词的 token IDs（已 pad）
        hotword_lengths (H,) int64 — 每个热词实际长度

    输出：
        selected (H, D) — 每个热词取 LSTM 最后有效时间步

    与分段版不同：这里内部完成 gather（输出 (H, D)），asr_engine ORT 路径直接加 batch 维使用。
    """

    def __init__(self, model):
        super().__init__()
        self.embed = model.decoder.embed
        self.bias_encoder = model.bias_encoder
        self.lstm_proj = model.lstm_proj

    def forward(self, hotword, hotword_lengths):
        hw_embed = self.embed(hotword)            # (H, L, D)
        rnn_output, _ = self.bias_encoder(hw_embed)  # (H, L, D[/2D])
        if self.lstm_proj is not None:
            rnn_output = self.lstm_proj(rnn_output)

        H = rnn_output.shape[0]
        D = rnn_output.shape[2]
        idx = (hotword_lengths.long() - 1).clamp(min=0).view(H, 1, 1).expand(H, 1, D)
        selected = torch.gather(rnn_output, 1, idx).squeeze(1)  # (H, D)
        return selected


def export_fp32_onnx(model_id: str, output_dir: Path, opset_version: int = 17):
    """使用 seaco_paraformer 加载模型并导出 fp32 ONNX（v1 整体导出，含热词）。

    导出两个产物：
        model.onnx    — 主模型（speech + speech_lengths + bias_embed → logits + token_num）
        model_eb.onnx — bias encoder（hotword + hotword_lengths → selected）
    """
    from seaco_paraformer.load_model import load_model

    export_dir = output_dir / "fp32"
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 加载模型: {model_id}")
    pt_model = load_model(model_id)

    print(f"[2/3] 导出主模型 model.onnx (opset_version={opset_version}, 含 bias_embed)...")

    inner_dim = pt_model.inner_dim
    main_wrapper = WholeModelWithBiasWrapper(pt_model)
    main_wrapper.eval()

    batch, seq_len, feat_dim = 1, 289, 560
    num_hotwords = 4
    speech = torch.randn(batch, seq_len, feat_dim)
    speech_lengths = torch.tensor([seq_len], dtype=torch.long)
    bias_embed = torch.randn(batch, num_hotwords, inner_dim)

    model_path = export_dir / "model.onnx"
    torch.onnx.export(
        main_wrapper,
        (speech, speech_lengths, bias_embed),
        str(model_path),
        opset_version=opset_version,
        input_names=["speech", "speech_lengths", "bias_embed"],
        output_names=["logits", "token_num"],
        dynamic_axes={
            "speech": {0: "batch", 1: "seq_len"},
            "speech_lengths": {0: "batch"},
            "bias_embed": {0: "batch", 1: "num_hotwords"},
            "logits": {0: "batch", 1: "token_len"},
            "token_num": {0: "batch"},
        },
    )
    print(f"   导出完成: {model_path.name}")

    print(f"[3/3] 导出 bias encoder model_eb.onnx (opset_version={opset_version})...")
    bias_wrapper = BiasEncoderWrapper(pt_model)
    bias_wrapper.eval()

    hw_len = 4
    hotword = torch.randint(1, 8404, (num_hotwords, hw_len), dtype=torch.long)
    hotword_lengths = torch.tensor([hw_len] * num_hotwords, dtype=torch.long)

    eb_path = export_dir / "model_eb.onnx"
    torch.onnx.export(
        bias_wrapper,
        (hotword, hotword_lengths),
        str(eb_path),
        opset_version=opset_version,
        input_names=["hotword", "hotword_lengths"],
        output_names=["bias_embed"],
        dynamic_axes={
            "hotword": {0: "num_hotwords", 1: "hw_len"},
            "hotword_lengths": {0: "num_hotwords"},
            "bias_embed": {0: "num_hotwords"},
        },
    )
    print(f"   导出完成: {eb_path.name}")

    onnx_files = list(export_dir.rglob("*.onnx"))
    print(f"   全部产物: {[f.name for f in onnx_files]}")
    return onnx_files


def convert_to_fp16(
    fp32_onnx_path: Path,
    output_path: Path,
    op_block_list: list[str],
):
    """将 fp32 ONNX 转为 fp16（使用 onnxruntime float16 工具，支持子图处理）。"""
    import onnx

    print(f"   转换 fp16: {fp32_onnx_path.name}")

    try:
        # 方案一：使用 onnxruntime.transformers.float16（支持子图）
        from onnxruntime.transformers import float16 as ort_float16

        model = onnx.load(str(fp32_onnx_path))
        model_fp16 = ort_float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            op_block_list=op_block_list,
            node_block_list=None,
        )
    except (ImportError, TypeError):
        # 方案二：使用 float16_converter 的 auto_mixed_precision
        try:
            from onnxruntime.transformers.onnx_model import OnnxModel
            from onnxruntime.transformers.float16 import convert_float_to_float16

            model = onnx.load(str(fp32_onnx_path))
            model_fp16 = convert_float_to_float16(
                model,
                keep_io_types=True,
                op_block_list=op_block_list,
            )
        except (ImportError, TypeError):
            # 方案三：onnxconverter-common + 禁用子图内所有节点
            from onnxconverter_common import float16

            model = onnx.load(str(fp32_onnx_path))

            # 收集所有子图内节点名称作为 node_block_list
            node_block_list = []
            for node in model.graph.node:
                if node.op_type in ("Loop", "If", "Scan"):
                    for attr in node.attribute:
                        if attr.g:
                            for sub_node in attr.g.node:
                                if sub_node.name:
                                    node_block_list.append(sub_node.name)

            # 收集子图内所有 op_type
            all_blocked_ops = set(op_block_list)
            for node in model.graph.node:
                if node.op_type in ("Loop", "If", "Scan"):
                    for attr in node.attribute:
                        if attr.g:
                            for sub_node in attr.g.node:
                                all_blocked_ops.add(sub_node.op_type)

            model_fp16 = float16.convert_float_to_float16(
                model,
                keep_io_types=True,
                op_block_list=list(all_blocked_ops),
                node_block_list=node_block_list if node_block_list else None,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 修复 Range 算子输入类型：Range 不支持 fp16 输入，需将其输入常量从 fp16 转回 fp32
    model_fp16 = _fix_range_inputs(model_fp16)

    onnx.save(model_fp16, str(output_path))

    fp32_mb = fp32_onnx_path.stat().st_size / (1024 * 1024)
    fp16_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   {fp32_mb:.1f}MB → {fp16_mb:.1f}MB ({fp16_mb/fp32_mb*100:.0f}%)")


def _fix_range_inputs(model):
    """
    修复 fp16 模型中 Range 算子的输入类型。
    Range 算子只接受 int32/int64/float32，但 fp16 转换可能将其输入常量转为 fp16。
    解决方案：找到所有 Range 节点，将其 fp16 输入的 initializer 转回 float32。
    """
    import onnx
    from onnx import numpy_helper, TensorProto

    graph = model.graph

    # 收集所有 Range 节点的输入名称
    range_input_names = set()
    for node in graph.node:
        if node.op_type == "Range":
            for inp in node.input:
                range_input_names.add(inp)

    if not range_input_names:
        return model

    # 修复 initializer 中的 fp16 → fp32
    for i, init in enumerate(graph.initializer):
        if init.name in range_input_names and init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype("float32")
            new_init = numpy_helper.from_array(arr, name=init.name)
            graph.initializer[i].CopyFrom(new_init)

    # 修复 graph input 中的类型声明
    for inp in graph.input:
        if inp.name in range_input_names:
            if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
                inp.type.tensor_type.elem_type = TensorProto.FLOAT

    # 修复 Constant 节点输出的类型（如果 Range 输入来自 Constant 节点）
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) > 0 and node.output[0] in range_input_names:
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype("float32")
                    new_tensor = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new_tensor)

    print(f"   修复 Range 输入: {len(range_input_names)} 个输入已转回 float32")
    return model


def main():
    parser = argparse.ArgumentParser(description="SeACo-Paraformer ONNX 导出")
    parser.add_argument("--model-id", default="./models/asr/pt",
                        help="PT 模型本地目录路径（默认 ./models/asr/pt，不联网下载）")
    parser.add_argument("--output-dir", default="./models/asr")
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--op-block-list", nargs="+",
                        default=["Range"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("=" * 60)
    print("SeACo-Paraformer ONNX 导出")
    print("=" * 60)
    print(f"模型: {args.model_id}")
    print(f"输出目录: {output_dir}")
    print(f"opset_version: {args.opset_version}")
    print()

    fp32_files = export_fp32_onnx(args.model_id, output_dir, args.opset_version)

    if not args.skip_fp16:
        print(f"\n转换 fp16（保留算子: {args.op_block_list}）")
        fp16_dir = output_dir / "fp16"
        for fp32_path in fp32_files:
            convert_to_fp16(fp32_path, fp16_dir / fp32_path.name, args.op_block_list)

    print("\n导出完成！")


if __name__ == "__main__":
    main()
