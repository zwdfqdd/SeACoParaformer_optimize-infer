"""
截断 Decoder 导出脚本

将 decoder 截断到前 N 层（decoders 部分），导出为 ONNX。
使用 seaco_paraformer 原始模型。

Decoder 结构：
    decoders (16 层，含 self_attn + cross_attn + ffn)
    decoders3 (1 层，仅 ffn)
    after_norm → output_layer

截断只影响 decoders 部分，decoders3 和 output_layer 保留。

用法：
    # decoder 截断到 12 层
    python scripts/export_decoder_truncated.py --num-layers 12

    # decoder 截断到 8 层
    python scripts/export_decoder_truncated.py --num-layers 8 --output ./models/asr/split/decoder_8layers.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seaco_paraformer.load_model import load_model


class TruncatedDecoderWrapper(nn.Module):
    """截断 Decoder：只保留 decoders 的前 num_layers 层。"""

    def __init__(self, decoder, num_layers: int):
        super().__init__()
        self.decoder = decoder

        total_layers = len(decoder.decoders)
        print(f"  decoders: {total_layers} 层")
        print(f"  截断到: {num_layers} 层")

        decoder.decoders = decoder.decoders[:num_layers]

    def forward(
        self,
        acoustic_embeds: torch.Tensor,
        acoustic_embeds_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ):
        logits, _ = self.decoder(
            encoder_out, encoder_out_lens,
            acoustic_embeds, acoustic_embeds_lens,
        )
        return logits


def main():
    parser = argparse.ArgumentParser(description="截断 Decoder 导出")
    parser.add_argument("--model-id", default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    parser.add_argument("--num-layers", type=int, default=16, help="保留的 decoder 层数（decoders 部分）")
    parser.add_argument("--output", default=None, help="输出路径")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.output is None:
        output_dir = Path("./models/asr/split")
        output_path = str(output_dir / f"decoder_{args.num_layers}layers.onnx")
    else:
        output_path = args.output

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"截断 Decoder 导出（前 {args.num_layers} 层）")
    print("=" * 60)

    print("\n加载模型...")
    pt_model = load_model(args.model_id)

    print(f"\n构建截断 Decoder...")
    decoder = TruncatedDecoderWrapper(pt_model.decoder, args.num_layers)
    decoder.eval()

    print(f"\n导出 ONNX...")
    batch, token_len, enc_len, hidden_dim = 1, 20, 67, 512
    acoustic_embeds = torch.randn(batch, token_len, hidden_dim)
    acoustic_embeds_lens = torch.tensor([token_len], dtype=torch.long)
    encoder_out = torch.randn(batch, enc_len, hidden_dim)
    encoder_out_lens = torch.tensor([enc_len], dtype=torch.long)

    torch.onnx.export(
        decoder,
        (acoustic_embeds, acoustic_embeds_lens, encoder_out, encoder_out_lens),
        output_path,
        opset_version=args.opset,
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
    print(f"\n  输出: {output_path} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
