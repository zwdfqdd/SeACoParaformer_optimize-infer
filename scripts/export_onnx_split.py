"""
SeACo-Paraformer 分段 ONNX 导出脚本

基于独立的 seaco_paraformer 包（不依赖 FunASR 运行时），导出 5 个子模型：
1. encoder.onnx       — Encoder（speech → encoder_out）
2. cif.onnx           — CIF Predictor（encoder_out + mask → acoustic_embeds, token_num）
3. decoder.onnx       — Decoder + SeACo（acoustic_embeds + encoder_out + bias_embed → logits）
4. bias_encoder.onnx  — Bias Encoder（hotword_ids → hw_embed）
5. timestamp.onnx     — 字级时间戳（encoder_out + mask + token_num → us_alphas, us_cif_peak）
                        按需启用（ENABLE_WORD_TIMESTAMP），不影响主链路吞吐

用法：
    python scripts/export_onnx_split.py --output-dir ./models/asr/split
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seaco_paraformer.load_model import load_model
from seaco_paraformer.predictor import cif_v1_export


# ============================================================
# Encoder Wrapper
# ============================================================
class EncoderWrapper(nn.Module):
    """Encoder 导出 wrapper（去掉 speech_lengths，推理时无需 mask）。"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, speech: torch.Tensor):
        # 内部 dummy lengths（不影响计算，因为 encoder 已改为 None mask）
        b = speech.shape[0]
        dummy_lengths = torch.tensor([speech.shape[1]] * b, dtype=torch.long, device=speech.device)
        encoder_out, encoder_out_lens, _ = self.encoder(speech, dummy_lengths)
        return encoder_out


# ============================================================
# CIF Predictor Wrapper（使用向量化 CIF，TRT 兼容）
# ============================================================
class CIFWrapper(nn.Module):
    """CIF Predictor 导出 wrapper。

    替换原始 cif（含 for 循环）为 cif_v1_export（向量化），保持 TRT 兼容。
    输入：encoder_out (B, T, D) + mask (B, 1, T)
    输出：acoustic_embeds, token_num, alphas, cif_peak

    注：字级时间戳（upsample+blstm）已拆分为独立的 TimestampWrapper（timestamp.onnx），
        通过 ENABLE_WORD_TIMESTAMP 开关按需加载，避免拖累主链路吞吐。
    """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, encoder_out: torch.Tensor, mask: torch.Tensor):
        pred = self.predictor
        h = encoder_out
        b, t, d = h.shape

        # 1. CIF conv → relu → cif_output → sigmoid → relu(smooth)
        context = h.transpose(1, 2)
        queries = pred.pad(context)
        output = torch.relu(pred.cif_conv1d(queries))
        output_t = output.transpose(1, 2)
        cif_logit = pred.cif_output(output_t)  # (B, T, 1)
        alphas = torch.sigmoid(cif_logit)
        alphas = torch.nn.functional.relu(
            alphas * pred.smooth_factor - pred.noise_threshold
        )

        # 2. mask
        mask_t = mask.transpose(-1, -2).float()  # (B, T, 1)
        alphas = alphas * mask_t
        alphas = alphas.squeeze(-1)  # (B, T)
        mask_squeezed = mask_t.squeeze(-1)  # (B, T)

        # 3. tail_process（与 predictor.tail_process_fn 一致）
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
        acoustic_embeds, cif_peak = cif_v1_export(hidden, alphas, pred.threshold)

        return acoustic_embeds, token_num, alphas, cif_peak


# ============================================================
# Timestamp Wrapper（独立第 5 段，字级时间戳，按需启用）
# ============================================================
class TimestampWrapper(nn.Module):
    """
    字级时间戳导出 wrapper（独立 engine，ENABLE_WORD_TIMESTAMP 控制加载）。

    upsample_cnn + blstm + cif_output2 计算量较大，并入 CIF 会拖累吞吐
    （实测 2800→2000 req/s）。拆为独立 engine 后不启用时零成本。

    输入：encoder_out (B, T, D) + mask (B, 1, T) + token_num (B,)
    输出：us_alphas (B, T*up)、us_cif_peak (B, T*up)
    复用 predictor.get_upsample_timestamp（与 PT 验证同一份代码）。
    """

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, encoder_out: torch.Tensor, mask: torch.Tensor,
                token_num: torch.Tensor):
        return self.predictor.get_upsample_timestamp(
            encoder_out, mask=mask, token_num=torch.round(token_num)
        )


# ============================================================
# Decoder + SeACo Wrapper
# ============================================================
class DecoderWithSeACoWrapper(nn.Module):
    """Decoder + SeACo 完整推理 wrapper。

    输入：
        acoustic_embeds (B, N, D)
        token_num (B,)
        encoder_out (B, T, D)
        encoder_out_lens (B,)
        bias_embed (B, H, D)

    输出：
        logits (B, N, vocab) — log_softmax 合并后
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.seaco_decoder = model.seaco_decoder
        self.hotword_output_layer = model.hotword_output_layer
        self.NO_BIAS = model.NO_BIAS
        self.seaco_weight = model.seaco_weight

    def forward(
        self,
        acoustic_embeds: torch.Tensor,
        token_num: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        bias_embed: torch.Tensor,
    ):
        # 主 decoder（return logits + hidden）
        decoder_out, decoder_hidden, _ = self.decoder(
            encoder_out, encoder_out_lens,
            acoustic_embeds, token_num,
            return_hidden=True, return_both=True,
        )
        decoder_pred = torch.log_softmax(decoder_out, dim=-1)

        # SeACo decoder：bias_embed 作为 memory，分别用 acoustic_embeds 和 decoder_hidden 作为 query
        B, H, D = bias_embed.shape
        contextual_length = torch.full(
            (B,), H, dtype=torch.long, device=bias_embed.device
        )

        cif_attended, _ = self.seaco_decoder(
            bias_embed, contextual_length,
            acoustic_embeds, token_num,
        )
        dec_attended, _ = self.seaco_decoder(
            bias_embed, contextual_length,
            decoder_hidden, token_num,
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
        return final_logits


# ============================================================
# Bias Encoder Wrapper
# ============================================================
class BiasEncoderWrapper(nn.Module):
    """Bias Encoder 导出 wrapper。

    输入：hotword (num_hotwords, hw_len) — num_hotwords 个热词的 token IDs（已 pad）
    输出：hw_embed (hw_len, num_hotwords, D) — LSTM 全序列输出（外部按热词长度取最后时间步）

    简化：使用全序列 LSTM（不用 pack_padded_sequence，便于 ONNX 导出）。
    外部代码需按 hotword_lengths 取每个热词的最后有效时间步。
    """

    def __init__(self, model):
        super().__init__()
        self.embed = model.decoder.embed  # 共享 decoder 的 embedding
        self.bias_encoder = model.bias_encoder
        self.lstm_proj = model.lstm_proj

    def forward(self, hotword: torch.Tensor):
        # embed: (H, L) → (H, L, D)
        hw_embed = self.embed(hotword)

        # LSTM 全序列：batch_first=True
        rnn_output, _ = self.bias_encoder(hw_embed)  # (H, L, D) or (H, L, 2D) if bid

        if self.lstm_proj is not None:
            rnn_output = self.lstm_proj(rnn_output)

        # 转为 (L, H, D)（与 FunASR ContextualEmbedderExport 输出格式一致）
        return rnn_output.transpose(0, 1)


# ============================================================
# 导出函数
# ============================================================
def export_encoder(model, output_path: str, opset: int = 17, clamp_value: float = None):
    """导出 Encoder（仅 speech 输入）。

    Args:
        clamp_value: 残差 Add 后的 clamp 阈值。None 或 0 = 不 clamp（PT/fp32/ORT 模式，无损）。
                     fp16/int8 推荐 60000：encoder 后段层残差激活峰值高达 ~48万（远超 fp16 上限
                     65504），60000 贴近上限最大化保留信息；clamp=30000 裁剪过狠已弃用。
    """
    print("\n[1/4] 导出 Encoder...")
    if clamp_value is not None and clamp_value > 0:
        print(f"  注入 clamp_value={clamp_value} 到所有 EncoderLayerSANM")
        # 注入到所有 encoder layer（encoders0 + encoders）
        for layer in model.encoder.encoders0:
            layer.clamp_value = clamp_value
        for layer in model.encoder.encoders:
            layer.clamp_value = clamp_value
        model.encoder.clamp_value = clamp_value

    encoder = EncoderWrapper(model.encoder)
    encoder.eval()

    batch, seq_len, feat_dim = 1, 134, 560
    speech = torch.randn(batch, seq_len, feat_dim)

    torch.onnx.export(
        encoder,
        (speech,),
        output_path,
        opset_version=opset,
        input_names=["speech"],
        output_names=["encoder_out"],
        dynamic_axes={
            "speech": {0: "batch", 1: "seq_len"},
            "encoder_out": {0: "batch", 1: "enc_len"},
        },
    )
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  输出: {output_path} ({size_mb:.1f}MB)")


def export_cif(model, output_path: str, opset: int = 17):
    """导出 CIF Predictor（向量化版本，TRT 兼容）。"""
    print("\n[2/4] 导出 CIF Predictor...")
    cif_wrapper = CIFWrapper(model.predictor)
    cif_wrapper.eval()

    # 用真实 encoder 输出作为 dummy input
    with torch.no_grad():
        dummy_speech = torch.randn(1, 134, 560)
        dummy_lengths = torch.tensor([134], dtype=torch.long)
        enc_out, _, _ = model.encoder(dummy_speech, dummy_lengths)

    mask = torch.ones(1, 1, enc_out.shape[1])

    torch.onnx.export(
        cif_wrapper,
        (enc_out, mask),
        output_path,
        opset_version=opset,
        input_names=["encoder_out", "mask"],
        output_names=["acoustic_embeds", "token_num", "alphas", "cif_peak"],
        dynamic_axes={
            "encoder_out": {0: "batch", 1: "enc_len"},
            "mask": {0: "batch", 2: "enc_len"},
            "acoustic_embeds": {0: "batch", 1: "token_len"},
            "token_num": {0: "batch"},
            "alphas": {0: "batch", 1: "enc_len_p1"},
            "cif_peak": {0: "batch", 1: "enc_len_p1"},
        },
    )
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  输出: {output_path} ({size_mb:.1f}MB)")


def export_timestamp(model, output_path: str, opset: int = 17):
    """
    导出字级时间戳独立 engine（第 5 段，按需启用）。
    upsample_cnn + blstm + cif_output2 → us_alphas / us_cif_peak。
    """
    print("\n[5] 导出 Timestamp（字级时间戳，独立 engine）...")
    ts_wrapper = TimestampWrapper(model.predictor)
    ts_wrapper.eval()

    with torch.no_grad():
        dummy_speech = torch.randn(1, 134, 560)
        dummy_lengths = torch.tensor([134], dtype=torch.long)
        enc_out, _, _ = model.encoder(dummy_speech, dummy_lengths)
    mask = torch.ones(1, 1, enc_out.shape[1])
    token_num = torch.tensor([30.0], dtype=torch.float32)

    torch.onnx.export(
        ts_wrapper,
        (enc_out, mask, token_num),
        output_path,
        opset_version=opset,
        input_names=["encoder_out", "mask", "token_num"],
        output_names=["us_alphas", "us_cif_peak"],
        dynamic_axes={
            "encoder_out": {0: "batch", 1: "enc_len"},
            "mask": {0: "batch", 2: "enc_len"},
            "token_num": {0: "batch"},
            "us_alphas": {0: "batch", 1: "enc_len_up"},
            "us_cif_peak": {0: "batch", 1: "enc_len_up"},
        },
    )
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  输出: {output_path} ({size_mb:.1f}MB)")


def export_decoder(model, output_path: str, opset: int = 17):
    """导出 Decoder + SeACo。"""
    print("\n[3/4] 导出 Decoder + SeACo...")
    dec_wrapper = DecoderWithSeACoWrapper(model)
    dec_wrapper.eval()

    batch, token_len, enc_len, hidden_dim = 1, 61, 134, 512
    num_hotwords = 4

    acoustic_embeds = torch.randn(batch, token_len, hidden_dim)
    token_num = torch.tensor([token_len], dtype=torch.long)
    encoder_out = torch.randn(batch, enc_len, hidden_dim)
    encoder_out_lens = torch.tensor([enc_len], dtype=torch.long)
    bias_embed = torch.randn(batch, num_hotwords, hidden_dim)

    torch.onnx.export(
        dec_wrapper,
        (acoustic_embeds, token_num, encoder_out, encoder_out_lens, bias_embed),
        output_path,
        opset_version=opset,
        input_names=["acoustic_embeds", "token_num", "encoder_out", "encoder_out_lens", "bias_embed"],
        output_names=["logits"],
        dynamic_axes={
            "acoustic_embeds": {0: "batch", 1: "token_len"},
            "token_num": {0: "batch"},
            "encoder_out": {0: "batch", 1: "enc_len"},
            "encoder_out_lens": {0: "batch"},
            "bias_embed": {0: "batch", 1: "num_hotwords"},
            "logits": {0: "batch", 1: "logits_len"},
        },
    )
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  输出: {output_path} ({size_mb:.1f}MB)")


def export_bias_encoder(model, output_path: str, opset: int = 17):
    """导出 Bias Encoder（热词编码器）。"""
    print("\n[4/4] 导出 Bias Encoder...")
    bias_wrapper = BiasEncoderWrapper(model)
    bias_wrapper.eval()

    num_hotwords, hw_len = 4, 4
    hotword = torch.randint(0, 8404, (num_hotwords, hw_len), dtype=torch.long)

    torch.onnx.export(
        bias_wrapper,
        (hotword,),
        output_path,
        opset_version=opset,
        input_names=["hotword"],
        output_names=["hw_embed"],
        dynamic_axes={
            "hotword": {0: "num_hotwords", 1: "hw_len"},
            "hw_embed": {0: "hw_len", 1: "num_hotwords"},
        },
    )
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  输出: {output_path} ({size_mb:.1f}MB)")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SeACo-Paraformer 分段 ONNX 导出")
    parser.add_argument("--model-id", default="./models/asr/pt",
                        help="PT 模型本地目录路径（默认 ./models/asr/pt，不联网下载）")
    parser.add_argument("--output-dir", default="./models/asr/split")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset 版本（默认 17，启用原生 LayerNormalization 算子，"
                             "fp16 推理无需 fp32 fallback）")
    parser.add_argument("--clamp-value", type=float, default=60000.0,
                        help="encoder 残差 Add 后 clamp 阈值（默认 60000，fp16/int8 标准）。"
                             "背景：encoder 后段层(32-49)残差累积激活峰值高达 ~48万，远超 fp16 上限 65504。"
                             "clamp=60000 贴近 fp16 上限、最大化保留信息（仅极少数峰值点被裁，CER 影响极小）；"
                             "clamp=30000 裁剪过狠（后段裁到3万）导致截断输入解码错乱，已弃用。"
                             "传 0 禁用 clamp（仅 fp32/ORT 路径用，无损但 fp16 会溢出）。")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["encoder", "cif", "decoder", "bias_encoder", "timestamp"],
                        help="跳过指定模块")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SeACo-Paraformer 分段 ONNX 导出（基于独立包）")
    print("=" * 60)

    print("\n加载模型...")
    model = load_model(args.model_id)
    print("\n模型子模块:")
    for name, child in model.named_children():
        param_count = sum(p.numel() for p in child.parameters())
        print(f"  {name}: {type(child).__name__}, {param_count/1e6:.1f}M")

    if "encoder" not in args.skip:
        export_encoder(model, str(output_dir / "encoder.onnx"), args.opset, clamp_value=args.clamp_value)
    if "cif" not in args.skip:
        export_cif(model, str(output_dir / "cif.onnx"), args.opset)
    if "decoder" not in args.skip:
        export_decoder(model, str(output_dir / "decoder.onnx"), args.opset)
    if "bias_encoder" not in args.skip:
        export_bias_encoder(model, str(output_dir / "bias_encoder.onnx"), args.opset)
    if "timestamp" not in args.skip:
        export_timestamp(model, str(output_dir / "timestamp.onnx"), args.opset)

    print("\n" + "=" * 60)
    print("导出完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
