"""
Bias Encoder QDQ 量化导出（方案 1：Explicit Quantization）

Bias Encoder（热词编码 LSTM）输入是热词 token IDs (H, L)，自包含（无需上游 engine）。
校准数据用真实词表（默认 models/asr/hotwords.txt）编码出的 token 序列；
词表不足时用词表内 token 随机组合补足，覆盖不同热词长度。

⚠ 说明：
    bias_encoder 是 LSTM，INT8 量化收益主要在显存。LSTM 对量化相对鲁棒，
    但若热词修正精度下降，可回退 bias=fp16（trt_int8_enc 方案）。

用法：
    python scripts/export_bias_qdq.py \\
        --output ./models/asr/split/bias_encoder_qdq.onnx

导出后转 TRT（QDQ 不需要 calibrator）：
    python scripts/convert_trt.py --input ./models/asr/split/bias_encoder_qdq.onnx \\
        --precision int8 --profile bias \\
        --output ./models/asr/trt/2080_ti_bias_encoder_int8_qdq.engine
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_decoder_qdq import detect_backend


def build_bias_wrapper(model_id: str = None):
    """加载 BiasEncoderWrapper。"""
    from seaco_paraformer.load_model import load_model
    from scripts.export_onnx_split import BiasEncoderWrapper

    model = load_model(model_id) if model_id else load_model()
    wrapper = BiasEncoderWrapper(model)
    wrapper.eval()
    return wrapper


def collect_bias_inputs(
    hotword_file: str, tokens_path: str,
    calib_hw_len: int = 8, max_samples: int = 500,
) -> list[torch.Tensor]:
    """从词表编码出热词 token 序列作为校准输入（每条 shape (H, L)）。

    为覆盖不同热词数 / 长度，构造多个 batch（不同 H、不同有效长度）。
    """
    from src.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    tokenizer.load(tokens_path)

    words: list[str] = []
    if hotword_file and Path(hotword_file).exists():
        with open(hotword_file, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]

    # 编码为 token 序列
    encoded = [tokenizer.encode(w) for w in words]
    encoded = [e for e in encoded if e]

    # 词表过少时用已知 token 合成补充（覆盖不同长度）
    if len(encoded) < 8:
        rng = np.random.default_rng(0)
        vocab = max(tokenizer.vocab_size, 100)
        for _ in range(32):
            L = int(rng.integers(1, calib_hw_len + 1))
            encoded.append([int(rng.integers(3, vocab)) for _ in range(L)])

    samples: list[torch.Tensor] = []
    # 构造不同热词数的 batch（含 [sos] 哨兵行），覆盖 profile 范围
    rng = np.random.default_rng(1)
    hw_counts = [1, 4, 16, 64]
    for hc in hw_counts:
        for _ in range(max(1, max_samples // (len(hw_counts) * 4))):
            picked = [encoded[int(rng.integers(0, len(encoded)))] for _ in range(hc)]
            picked.append([1])  # [sos] 哨兵
            max_len = min(max(len(ids) for ids in picked), calib_hw_len)
            mat = np.zeros((len(picked), max_len), dtype=np.int64)
            for i, ids in enumerate(picked):
                v = min(len(ids), max_len)
                mat[i, :v] = ids[:v]
            samples.append(torch.from_numpy(mat).long())

    print(f"  校准样本: {len(samples)} 条（热词数覆盖 {hw_counts}）")
    return samples


def quantize_with_modelopt(wrapper, samples, output_path, opset, calib_hw_len,
                           exclude_patterns=None):
    import modelopt.torch.quantization as mtq

    print("[modelopt] 配置 INT8 量化（INT8_DEFAULT_CFG）...")
    config = mtq.INT8_DEFAULT_CFG

    if exclude_patterns:
        import copy
        config = copy.deepcopy(config)
        quant_cfg = config.setdefault("quant_cfg", {})
        for pat in exclude_patterns:
            quant_cfg[f"*{pat}*"] = {"enable": False}
        print(f"  排除量化的模块模式: {exclude_patterns}")

    def forward_loop(model):
        with torch.no_grad():
            for i, hw in enumerate(samples):
                model(hw)
                if (i + 1) % 50 == 0:
                    print(f"  校准进度: {i + 1}/{len(samples)}")

    print("[modelopt] 量化 + 校准中...")
    mtq.quantize(wrapper, config, forward_loop)

    try:
        mtq.print_quant_summary(wrapper)
    except Exception as e:
        print(f"  （print_quant_summary 跳过: {e}）")

    print(f"\n[modelopt] 导出 QDQ ONNX: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # dummy：(H, L) 热词 token（含哨兵行）
    dummy = torch.ones((4, calib_hw_len), dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), output_path,
            opset_version=opset,
            input_names=["hotword"],
            output_names=["hw_embed"],
            dynamic_axes={
                "hotword": {0: "num_hotwords", 1: "hw_len"},
                "hw_embed": {0: "hw_len", 1: "num_hotwords"},
            },
            do_constant_folding=False,
        )

    try:
        import onnx
        m = onnx.load(output_path)
        q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
        dq = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear")
        print(f"\n  QDQ 节点统计: QuantizeLinear={q}, DequantizeLinear={dq}")
        if q == 0:
            print("  ⚠ 未发现 QuantizeLinear，QDQ 导出可能失败（LSTM 量化支持有限）")
        else:
            print("  ✓ QDQ 节点已嵌入 ONNX")
    except Exception as e:
        print(f"  （ONNX QDQ 检查跳过: {e}）")


def main():
    parser = argparse.ArgumentParser(description="Bias Encoder QDQ 量化导出（方案 1）")
    parser.add_argument("--hotword-file", default="./models/asr/hotwords.txt",
                        help="校准用词表（编码出 token 序列）")
    parser.add_argument("--tokens-path", default="./models/asr/tokens.json")
    parser.add_argument("--output", default="./models/asr/split/bias_encoder_qdq.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--calib-hw-len", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--backend", default=None,
                        choices=["modelopt", "pytorch_quantization"])
    parser.add_argument("--exclude-patterns", nargs="*", default=[],
                        help="排除量化的模块名模式（保持 fp16）。"
                             "如 LSTM 量化导致热词精度下降，可加 bias_encoder。")
    parser.add_argument("--model-id", default=None,
                        help="PT 模型 ID 或本地目录路径（默认 ModelScope 在线）")
    args = parser.parse_args()

    backend = detect_backend(args.backend)
    print("=" * 60)
    print("Bias Encoder QDQ 量化导出（方案 1）")
    print(f"  量化库: {backend}")
    print(f"  校准词表: {args.hotword_file}")
    print(f"  输出: {args.output}")
    print("=" * 60)

    print("\n[1/3] 加载 bias encoder...")
    wrapper = build_bias_wrapper(args.model_id)

    print("\n[2/3] 编码热词生成校准输入...")
    samples = collect_bias_inputs(
        args.hotword_file, args.tokens_path, args.calib_hw_len, args.max_samples
    )

    print("\n[3/3] 量化 + 导出...")
    if backend == "modelopt":
        quantize_with_modelopt(
            wrapper, samples, args.output, args.opset, args.calib_hw_len,
            exclude_patterns=args.exclude_patterns,
        )
    else:
        sys.exit("bias QDQ 仅支持 modelopt 后端")

    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\n完成！QDQ ONNX: {args.output} ({size_mb:.1f}MB)")
    print("\n下一步转 TRT：")
    print(f"  python scripts/convert_trt.py --input {args.output} \\")
    print(f"      --precision int8 --profile bias \\")
    print(f"      --output ./models/asr/trt/2080_ti_bias_encoder_int8_qdq.engine")


if __name__ == "__main__":
    main()
