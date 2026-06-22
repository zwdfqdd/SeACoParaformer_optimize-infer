"""
截断 Encoder/Decoder 导出脚本

将 encoder 或 decoder 截断到前 N 层，导出为 ONNX。
使用 seaco_paraformer 原始模型。

用法：
    # encoder 截断到 40 层
    python scripts/export_encoder_truncated.py --num-layers 40

    # encoder 截断到 31 层
    python scripts/export_encoder_truncated.py --num-layers 31 --output ./models/asr/split/encoder_31layers.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seaco_paraformer.load_model import load_model


class TruncatedEncoderWrapper(nn.Module):
    """截断 Encoder：只保留前 num_layers 层。"""

    def __init__(self, encoder, num_layers: int):
        super().__init__()
        self.encoder = encoder

        # 截断
        encoders0 = encoder.encoders0
        encoders = encoder.encoders
        total_layers = len(encoders0) + len(encoders)
        print(f"  双段结构: encoders0={len(encoders0)} 层 + encoders={len(encoders)} 层 = {total_layers} 层")
        print(f"  截断到: {num_layers} 层")

        if num_layers <= len(encoders0):
            encoder.encoders0 = encoders0[:num_layers]
            encoder.encoders = encoders[:0]
        else:
            remain = num_layers - len(encoders0)
            encoder.encoders = encoders[:remain]

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        encoder_out, olens, _ = self.encoder(speech, speech_lengths)
        return encoder_out, speech_lengths


def main():
    parser = argparse.ArgumentParser(description="截断 Encoder 导出")
    parser.add_argument("--model-id", default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    parser.add_argument("--num-layers", type=int, default=40, help="保留的 encoder 层数")
    parser.add_argument("--output", default=None, help="输出路径")
    parser.add_argument("--opset", type=int, default=16)
    args = parser.parse_args()

    if args.output is None:
        output_dir = Path("./models/asr/split")
        output_path = str(output_dir / f"encoder_{args.num_layers}layers.onnx")
    else:
        output_path = args.output

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"截断 Encoder 导出（前 {args.num_layers} 层）")
    print("=" * 60)

    print("\n加载模型...")
    pt_model = load_model(args.model_id)

    print(f"\n构建截断 Encoder...")
    encoder = TruncatedEncoderWrapper(pt_model.encoder, args.num_layers)
    encoder.eval()

    print(f"\n导出 ONNX...")
    batch, seq_len, feat_dim = 1, 289, 560
    speech = torch.randn(batch, seq_len, feat_dim)
    speech_lengths = torch.tensor([seq_len], dtype=torch.long)

    torch.onnx.export(
        encoder,
        (speech, speech_lengths),
        output_path,
        opset_version=args.opset,
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
    print(f"\n  输出: {output_path} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
