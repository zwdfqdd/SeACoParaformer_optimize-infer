"""
SeACo-Paraformer 分段 ONNX 导出脚本（v2 TRT 版本，不含热词）

从 PyTorch 模型分别导出三个子模型：
1. encoder.onnx — Encoder（speech → encoder_output）
2. cif.onnx — CIF Predictor（encoder_output → acoustic_embeds, token_num）
3. decoder.onnx — Decoder（acoustic_embeds + encoder_output → logits）

v2 TRT 版本不支持热词（seaco_decoder / model_eb 不导出）。
热词功能使用 v1 ORT 完整模型。

环境要求：转换容器（含 FunASR + PyTorch + onnx）

用法：
    python scripts/export_onnx_split.py --output-dir ./models/asr/split
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

    # Patch CIF（消除 Loop 算子）
    try:
        from funasr.models.paraformer.cif_predictor import cif_v1_export, cif_wo_hidden_v1
        import funasr.models.bicif_paraformer.cif_predictor as bicif_module
        bicif_module.cif_export = cif_v1_export
        bicif_module.cif_wo_hidden_export = cif_wo_hidden_v1
    except Exception as e:
        print(f"  警告：CIF patch 失败: {e}")

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

    pt_model = model.model
    pt_model.eval()
    return pt_model


class EncoderWrapper(nn.Module):
    """Encoder：使用 FunASR SANMEncoderExport 处理动态 mask。"""

    def __init__(self, encoder):
        super().__init__()
        from funasr.models.sanm.encoder import SANMEncoderExport
        self.encoder_export = SANMEncoderExport(encoder)

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        results = self.encoder_export(speech, speech_lengths)
        if isinstance(results, (tuple, list)):
            encoder_out = results[0]
            encoder_out_lens = results[1] if len(results) > 1 else speech_lengths
        else:
            encoder_out = results
            encoder_out_lens = speech_lengths
        return encoder_out, encoder_out_lens


class CIFWrapper(nn.Module):
    """CIF Predictor：encoder_output → acoustic_embeds, token_num, us_alphas, cif_peak"""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor):
        hidden = encoder_out
        b = hidden.shape[0]
        t = hidden.shape[1]
        d = hidden.shape[2]

        # Conv1D + Relu
        context = hidden.transpose(1, 2)
        queries = self.predictor.pad(context)
        output = torch.relu(self.predictor.cif_conv1d(queries))
        output = output.transpose(1, 2)

        # alphas
        alphas = torch.sigmoid(self.predictor.cif_output(output))
        alphas = alphas.squeeze(-1)

        # upsample: ConvTranspose1d → BLSTM → cif_output2
        _output = context
        if self.predictor.use_cif1_cnn:
            _output = output.transpose(1, 2)
        output2 = self.predictor.upsample_cnn(_output)
        output2 = output2.transpose(1, 2)
        output2, (_, _) = self.predictor.blstm(output2)
        us_alphas = torch.sigmoid(self.predictor.cif_output2(output2))
        us_alphas = us_alphas.squeeze(-1)

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
    """Decoder：acoustic_embeds + encoder_out → logits（不含热词增强）"""

    def __init__(self, decoder_export):
        super().__init__()
        self.decoder = decoder_export

    def forward(
        self,
        acoustic_embeds: torch.Tensor,
        acoustic_embeds_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ):
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

        logits = self.decoder.output_layer(decoder_hidden)
        return logits


def export_encoder(pt_model, output_path: str, opset: int = 16):
    """导出 Encoder。"""
    print("\n[1/3] 导出 Encoder...")

    encoder = EncoderWrapper(pt_model.encoder)
    encoder.eval()

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


def export_cif(pt_model, output_path: str, opset: int = 16):
    """导出 CIF Predictor。"""
    print("\n[2/3] 导出 CIF Predictor...")

    cif = CIFWrapper(pt_model.predictor)
    cif.eval()

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


def export_decoder(pt_model, output_path: str, opset: int = 16):
    """导出 Decoder（不含 seaco_decoder 热词增强）。"""
    print("\n[3/3] 导出 Decoder...")

    from funasr.models.e_paraformer.decoder import ParaformerSANMDecoderExport

    decoder_export = ParaformerSANMDecoderExport(pt_model.decoder)
    decoder_export.eval()

    dec_wrapper = DecoderWrapper(decoder_export)
    dec_wrapper.eval()

    batch = 1
    token_len = 20
    enc_len = 67
    hidden_dim = 512

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


def main():
    parser = argparse.ArgumentParser(description="SeACo-Paraformer 分段 ONNX 导出（v2，不含热词）")
    parser.add_argument("--model-id", default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    parser.add_argument("--output-dir", default="./models/asr/split")
    parser.add_argument("--opset", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SeACo-Paraformer 分段 ONNX 导出（v2，不含热词）")
    print("=" * 60)
    print(f"模型: {args.model_id}")
    print(f"输出: {output_dir}")
    print(f"opset: {args.opset}")

    print("\n加载 PyTorch 模型...")
    pt_model = load_model(args.model_id)

    print("\n模型子模块:")
    for name, module in pt_model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {param_count/1e6:.1f}M params")

    export_encoder(pt_model, str(output_dir / "encoder.onnx"), args.opset)
    export_cif(pt_model, str(output_dir / "cif.onnx"), args.opset)
    export_decoder(pt_model, str(output_dir / "decoder.onnx"), args.opset)

    print("\n" + "=" * 60)
    print("导出完成！")
    print("=" * 60)
    print("\n下一步（TRT 转换）:")
    print(f"  python scripts/convert_trt.py --input {output_dir}/encoder.onnx --precision fp16 --profile encoder")
    print(f"  python scripts/convert_trt.py --input {output_dir}/cif.onnx --precision fp16 --profile cif")
    print(f"  python scripts/convert_trt.py --input {output_dir}/decoder.onnx --precision fp16 --profile decoder")


if __name__ == "__main__":
    main()

