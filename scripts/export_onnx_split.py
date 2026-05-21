"""
SeACo-Paraformer 分段 ONNX 导出脚本

从 PyTorch 模型分别导出三个子模型：
1. encoder.onnx — Encoder（speech → encoder_output）
2. cif.onnx — CIF Predictor（encoder_output → acoustic_embeds, token_num, cif_peak）
3. decoder.onnx — Decoder + SeACo Decoder（acoustic_embeds + encoder_output + bias → logits）

各子模型有干净的输入输出接口，可独立优化：
- encoder → TRT fp16（计算密集，无问题算子）
- cif → ORT fp32（含 NonZero/CumSum，轻量）
- decoder → TRT fp16（MatMul/Attention，无问题算子）

环境要求：转换容器（含 FunASR + PyTorch + onnx）

用法：
    python scripts/export_onnx_split.py --output-dir ./models/asr/split
    python scripts/export_onnx_split.py --output-dir ./models/asr/split --opset 16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_model(model_id: str, revision: str = "v2.0.4"):
    """加载 FunASR SeACo-Paraformer 模型。"""
    from funasr import AutoModel

    # Patch CIF（与 export_onnx.py 一致）
    try:
        from funasr.models.paraformer.cif_predictor import cif_v1_export, cif_wo_hidden_v1
        import funasr.models.bicif_paraformer.cif_predictor as bicif_module
        bicif_module.cif_export = cif_v1_export
        bicif_module.cif_wo_hidden_export = cif_wo_hidden_v1
    except Exception as e:
        print(f"  警告：CIF patch 失败: {e}")

    # 注册 Export 类
    try:
        from funasr.register import tables
        from funasr.models.bicif_paraformer.cif_predictor import CifPredictorV3Export
        tables.predictor_classes["CifPredictorV3Export"] = CifPredictorV3Export
    except Exception:
        pass

    model = AutoModel(
        model=model_id,
        model_revision=revision,
        device="cpu",
        disable_update=True,
    )

    # 获取内部 PyTorch 模型
    pt_model = model.model
    pt_model.eval()
    return pt_model


class EncoderWrapper(nn.Module):
    """Encoder 包装器：使用 FunASR SANMEncoderExport 处理动态 mask。"""

    def __init__(self, encoder):
        super().__init__()
        from funasr.models.sanm.encoder import SANMEncoderExport
        self.encoder_export = SANMEncoderExport(encoder)

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        # SANMEncoderExport.forward 正确处理动态 attention mask
        results = self.encoder_export(speech, speech_lengths)
        if isinstance(results, (tuple, list)):
            encoder_out = results[0]
            encoder_out_lens = results[1] if len(results) > 1 else speech_lengths
        else:
            encoder_out = results
            encoder_out_lens = speech_lengths
        return encoder_out, encoder_out_lens


class CIFWrapper(nn.Module):
    """CIF Predictor 包装器：encoder_output + encoder_out_lens → acoustic_embeds, token_num, alphas, cif_peak"""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor):
        # 内联 CIF predictor 逻辑
        hidden = encoder_out
        b = hidden.shape[0]
        t = hidden.shape[1]
        d = hidden.shape[2]

        # Conv1D + Relu
        context = hidden.transpose(1, 2)  # (b, 512, t)
        queries = self.predictor.pad(context)
        output = torch.relu(self.predictor.cif_conv1d(queries))  # (b, 512, t)
        output = output.transpose(1, 2)  # (b, t, 512)

        # alphas（用 cif_output，输入 512）
        alphas = torch.sigmoid(self.predictor.cif_output(output))  # (b, t, 1)
        alphas = alphas.squeeze(-1)  # (b, t)

        # upsample: ConvTranspose1d → BLSTM → cif_output2
        _output = context  # (b, 512, t)
        if self.predictor.use_cif1_cnn:
            _output = output.transpose(1, 2)  # (b, 512, t)
        output2 = self.predictor.upsample_cnn(_output)  # (b, 512, t*3)
        output2 = output2.transpose(1, 2)  # (b, t*3, 512)
        output2, (_, _) = self.predictor.blstm(output2)  # (b, t*3, 1024)
        us_alphas = torch.sigmoid(self.predictor.cif_output2(output2))  # (b, t*3, 1)
        us_alphas = us_alphas.squeeze(-1)  # (b, t*3)

        # tail_process（内联）
        mask = torch.ones(b, t, dtype=torch.float32, device=hidden.device)
        zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=hidden.device)
        ones_t = torch.ones_like(zeros_t)
        mask_1 = torch.cat([mask, zeros_t], dim=1)
        mask_2 = torch.cat([ones_t, mask], dim=1)
        tail_mask = mask_2 - mask_1
        tail_threshold = tail_mask * self.predictor.tail_threshold
        alphas = torch.cat([alphas, zeros_t], dim=1)
        alphas = alphas + tail_threshold

        zeros_hidden = torch.zeros((b, 1, d), dtype=hidden.dtype, device=hidden.device)
        hidden = torch.cat([hidden, zeros_hidden], dim=1)

        token_num = alphas.sum(dim=-1)

        # CIF 核心（向量化版本）
        from funasr.models.paraformer.cif_predictor import cif_v1_export
        acoustic_embeds, cif_peak = cif_v1_export(hidden, alphas, self.predictor.threshold)

        return acoustic_embeds, token_num, us_alphas, cif_peak


class DecoderWrapper(nn.Module):
    """Decoder 包装器：acoustic_embeds + encoder_out + bias_embed → logits"""

    def __init__(self, decoder, seaco_decoder=None, output_layer=None):
        super().__init__()
        self.decoder = decoder
        self.seaco_decoder = seaco_decoder
        self.output_layer = output_layer

    def forward(
        self,
        acoustic_embeds: torch.Tensor,
        acoustic_embeds_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ):
        # 主 decoder: return_hidden=True → hidden (batch, token_len, 512)
        dec_result = self.decoder(
            encoder_out,
            encoder_out_lens,
            acoustic_embeds,
            acoustic_embeds_lens,
            return_hidden=True,
        )
        if isinstance(dec_result, tuple):
            decoder_hidden = dec_result[0]
        else:
            decoder_hidden = dec_result

        # SeACo decoder 作为独立后处理（不包含在 decoder.onnx 中）
        # 有热词时在推理脚本中单独调用 seaco_decoder
        logits = self.decoder.output_layer(decoder_hidden)

        return logits


def export_encoder(pt_model, output_path: str, opset: int = 16):
    """导出 Encoder 子模型。"""
    print("\n[1/3] 导出 Encoder...")

    encoder = EncoderWrapper(pt_model.encoder)
    encoder.eval()

    # Dummy inputs（用 max seq_len 导出，确保 attention mask 覆盖所有可能长度）
    batch, seq_len, feat_dim = 1, 289, 560
    speech = torch.randn(batch, seq_len, feat_dim)
    speech_lengths = torch.tensor([seq_len], dtype=torch.long)

    torch.onnx.export(
        encoder,
        (speech, speech_lengths),
        output_path,
        opset_version=opset,
        input_names=["speech", "speech_lengths"],
        output_names=["encoder_out", "encoder_out_lens"],
        dynamic_axes={
            "speech": {0: "batch", 1: "seq_len"},
            "speech_lengths": {0: "batch"},
            "encoder_out": {0: "batch", 1: "enc_len"},
            "encoder_out_lens": {0: "batch"},
        },
    )

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  输出: {output_path} ({size_mb:.1f}MB)")
    print(f"  输入: speech(batch, seq_len, 560), speech_lengths(batch,)")
    print(f"  输出: encoder_out(batch, enc_len, 512), encoder_out_lens(batch,)")


def export_cif(pt_model, output_path: str, opset: int = 16):
    """导出 CIF Predictor 子模型。"""
    print("\n[2/3] 导出 CIF Predictor...")

    cif = CIFWrapper(pt_model.predictor)
    cif.eval()

    # Dummy inputs（batch=2 避免 trace 时维度压缩）
    batch, enc_len, hidden_dim = 2, 67, 512
    encoder_out = torch.randn(batch, enc_len, hidden_dim)
    encoder_out_lens = torch.tensor([enc_len, enc_len], dtype=torch.long)

    torch.onnx.export(
        cif,
        (encoder_out, encoder_out_lens),
        output_path,
        opset_version=opset,
        input_names=["encoder_out", "encoder_out_lens"],
        output_names=["acoustic_embeds", "token_num", "alphas", "cif_peak"],
        dynamic_axes={
            "encoder_out": {0: "batch", 1: "enc_len"},
            "encoder_out_lens": {0: "batch"},
            "acoustic_embeds": {0: "batch", 1: "token_len"},
            "token_num": {0: "batch"},
            "alphas": {0: "batch", 1: "enc_len"},
            "cif_peak": {0: "batch", 1: "enc_len"},
        },
    )

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  输出: {output_path} ({size_mb:.1f}MB)")
    print(f"  输入: encoder_out(batch, enc_len, 512), encoder_out_lens(batch,)")
    print(f"  输出: acoustic_embeds(batch, token_len, 512), token_num, alphas, cif_peak")


def export_decoder(pt_model, output_path: str, opset: int = 16):
    """导出 Decoder 子模型（使用 FunASR Export 类处理 SANM mask 对齐）。"""
    print("\n[3/3] 导出 Decoder...")

    from funasr.models.e_paraformer.decoder import ParaformerSANMDecoderExport

    # 获取 decoder 组件
    decoder = pt_model.decoder if hasattr(pt_model, 'decoder') else None
    seaco_decoder = pt_model.seaco_decoder if hasattr(pt_model, 'seaco_decoder') else None
    output_layer = pt_model.output_layer if hasattr(pt_model, 'output_layer') else None

    # 用 Export 版本替换（正确处理 SANM mask/padding）
    decoder_export = ParaformerSANMDecoderExport(decoder)
    decoder_export.eval()

    seaco_decoder_export = None
    if seaco_decoder is not None:
        seaco_decoder_export = ParaformerSANMDecoderExport(seaco_decoder)
        seaco_decoder_export.eval()

    dec_wrapper = DecoderWrapper(decoder_export, seaco_decoder_export, output_layer)
    dec_wrapper.eval()

    # Dummy inputs（batch=1）
    batch = 1
    token_len = 20
    enc_len = 67
    hidden_dim = 512
    num_hotwords = 4

    acoustic_embeds = torch.randn(batch, token_len, hidden_dim)
    acoustic_embeds_lens = torch.tensor([token_len], dtype=torch.long)
    encoder_out = torch.randn(batch, enc_len, hidden_dim)
    encoder_out_lens = torch.tensor([enc_len], dtype=torch.long)

    torch.onnx.export(
        dec_wrapper,
        (acoustic_embeds, acoustic_embeds_lens, encoder_out, encoder_out_lens),
        output_path,
        opset_version=opset,
        input_names=["acoustic_embeds", "acoustic_embeds_lens", "encoder_out", "encoder_out_lens"],
        output_names=["logits"],
        dynamic_axes={
            "acoustic_embeds": {0: "batch", 1: "token_len"},
            "acoustic_embeds_lens": {0: "batch"},
            "encoder_out": {0: "batch", 1: "enc_len"},
            "encoder_out_lens": {0: "batch"},
            "logits": {0: "batch", 1: "logits_len"},
        },
    )

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  输出: {output_path} ({size_mb:.1f}MB)")
    print(f"  输入: acoustic_embeds, acoustic_embeds_lens, encoder_out, encoder_out_lens")
    print(f"  输出: logits(batch, logits_len, 8404)")
    print(f"  注意: seaco_decoder 热词增强作为独立后处理，不包含在此模型中")


def export_bias_encoder(pt_model, output_path: str, opset: int = 16):
    """导出 Bias Encoder（热词编码器）。

    流程：hotword_ids → embed → LSTM → bias_embed
    """
    print("\n[4/4] 导出 Bias Encoder...")

    class BiasEncoderWrapper(nn.Module):
        def __init__(self, embed, bias_encoder):
            super().__init__()
            self.embed = embed
            self.bias_encoder = bias_encoder

        def forward(self, hotword_ids: torch.Tensor):
            # hotword_ids: (num_hotwords, max_len) int64
            # embed: token_ids → embeddings
            if hasattr(self.embed, 'embed'):
                # FunASR embed 可能是 Sequential(Embedding, PositionalEncoding, ...)
                hw_embed = self.embed.embed(hotword_ids)  # (num_hw, max_len, 512)
            else:
                hw_embed = self.embed(hotword_ids)

            # LSTM encoder
            lstm_out, _ = self.bias_encoder(hw_embed)  # (num_hw, max_len, 512)

            # 取最后一个时间步作为热词表示
            # 或者取均值 — 需要看 FunASR 原始实现
            # 这里取最后时间步（与 FunASR export 一致）
            bias_embed = lstm_out[:, -1:, :]  # (num_hw, 1, 512)

            return bias_embed

    # 获取 embed 层（从 decoder 中）
    embed = pt_model.decoder.embed if hasattr(pt_model.decoder, 'embed') else None
    bias_encoder = pt_model.bias_encoder

    if embed is None:
        print("  跳过：未找到 decoder.embed 模块")
        return

    wrapper = BiasEncoderWrapper(embed, bias_encoder)
    wrapper.eval()

    # Dummy input
    num_hotwords, max_len = 4, 5
    hotword_ids = torch.randint(0, 8404, (num_hotwords, max_len), dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (hotword_ids,),
        output_path,
        opset_version=opset,
        input_names=["hotword"],
        output_names=["hw_embed"],
        dynamic_axes={
            "hotword": {0: "num_hotwords", 1: "max_len"},
            "hw_embed": {0: "num_hotwords"},
        },
    )

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  输出: {output_path} ({size_mb:.1f}MB)")
    print(f"  输入: hotword(num_hotwords, max_len) int64")
    print(f"  输出: hw_embed(num_hotwords, 1, 512)")


def main():
    parser = argparse.ArgumentParser(description="SeACo-Paraformer 分段 ONNX 导出")
    parser.add_argument("--model-id", default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    parser.add_argument("--output-dir", default="./models/asr/split")
    parser.add_argument("--opset", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SeACo-Paraformer 分段 ONNX 导出")
    print("=" * 60)
    print(f"模型: {args.model_id}")
    print(f"输出: {output_dir}")
    print(f"opset: {args.opset}")

    # 加载模型
    print("\n加载 PyTorch 模型...")
    pt_model = load_model(args.model_id)

    # 打印模型结构
    print("\n模型子模块:")
    for name, module in pt_model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {param_count/1e6:.1f}M params")

    # 分别导出
    export_encoder(pt_model, str(output_dir / "encoder.onnx"), args.opset)
    export_cif(pt_model, str(output_dir / "cif.onnx"), args.opset)
    export_decoder(pt_model, str(output_dir / "decoder.onnx"), args.opset)
    export_bias_encoder(pt_model, str(output_dir / "model_eb.onnx"), args.opset)

    print("\n" + "=" * 60)
    print("导出完成！")
    print("=" * 60)
    print("\n下一步:")
    print(f"  # encoder → TRT fp16")
    print(f"  python scripts/convert_trt.py --input {output_dir}/encoder.onnx --precision fp16 --profile encoder")
    print(f"  # decoder → TRT fp16")
    print(f"  python scripts/convert_trt.py --input {output_dir}/decoder.onnx --precision fp16 --profile decoder")
    print(f"  # cif + model_eb 保持 ORT fp32（cif 含 NonZero，model_eb 轻量）")
    print(f"  # model_eb.onnx 复用 models/asr/fp32/model_eb.onnx")


if __name__ == "__main__":
    main()
