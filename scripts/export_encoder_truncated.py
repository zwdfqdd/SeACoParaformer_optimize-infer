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
    parser.add_argument("--model-id", default="./models/asr/pt",
                        help="PT 模型本地目录路径（默认 ./models/asr/pt，不联网下载）")
    parser.add_argument("--num-layers", type=int, default=40, help="保留的 encoder 层数")
    parser.add_argument("--output", default=None, help="输出路径")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--compare", action="store_true",
                        help="导出后立即用 PT(截断版) vs ORT 对比同输入的 encoder_out，定位误差层")
    parser.add_argument("--cmp-frames", type=int, default=125, help="对比用的序列帧数")
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

    # 对比 PT(截断版) vs ORT，定位误差层
    if args.compare:
        import numpy as np
        import onnxruntime as ort
        print(f"\n{'='*60}")
        print(f"[对比] PT(前{args.num_layers}层) vs ORT，{args.cmp_frames} 帧随机输入")
        print(f"{'='*60}")
        torch.manual_seed(0)
        x = torch.randn(1, args.cmp_frames, 560)
        xl = torch.tensor([args.cmp_frames], dtype=torch.long)
        with torch.no_grad():
            pt_out, _ = encoder(x, xl)
        pt_out = pt_out.cpu().numpy()
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(["encoder_out"], {"speech": x.numpy(), "speech_lengths": xl.numpy()})[0]
        d = np.abs(pt_out - ort_out).max()
        print(f"  PT  out abs max={np.abs(pt_out).max():.4f}")
        print(f"  ORT out abs max={np.abs(ort_out).max():.4f}")
        print(f"  encoder_out 最大绝对误差: {d:.6f}  "
              f"{'✓ 一致（此截断点无问题）' if d < 1e-2 else '✗ 误差已出现（根因在前 '+str(args.num_layers)+' 层内）'}")


if __name__ == "__main__":
    main()
