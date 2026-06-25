"""
ONNX fp32 → int8 动态量化脚本

将 fp32 ONNX 模型转换为 int8 动态量化模型，用于 CPU 推理。
- 量化 MatMul/Gemm 权重为 int8
- 无需校准数据集（动态量化，推理时动态计算 scale）
- 模型文件约缩小 75%
- 仅适用于 CPU 推理（CPUExecutionProvider）

用法：
    # 转换整个目录
    python scripts/convert_onnx_int8_dynamic.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/int8

    # 转换单个模型
    python scripts/convert_onnx_int8_dynamic.py --input ./models/asr/fp32/model.onnx --output ./models/asr/int8/model.onnx

    # 转换后验证（CPU）
    python scripts/verify_onnx.py --audio test_data/audio_16000_30s.wav --onnx-dir ./models/asr/int8 --device cpu
"""

import argparse
import sys
from pathlib import Path


def convert_to_int8(fp32_path: Path, output_path: Path):
    """将 fp32 ONNX 动态量化为 int8。"""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"转换: {fp32_path.name}")
    print(f"  输入: {fp32_path}")
    print(f"  输出: {output_path}")
    print(f"  量化类型: 动态量化 (int8, 仅 MatMul/Gemm)")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    int8_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  大小: {fp32_mb:.1f}MB → {int8_mb:.1f}MB ({int8_mb/fp32_mb*100:.0f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="ONNX fp32 → int8 动态量化")
    parser.add_argument("--input", default=None, help="单个 fp32 ONNX 文件路径")
    parser.add_argument("--output", default=None, help="输出 int8 文件路径")
    parser.add_argument("--input-dir", default=None, help="fp32 模型目录（转换目录下所有 .onnx）")
    parser.add_argument("--output-dir", default=None, help="int8 输出目录")
    args = parser.parse_args()

    print("=" * 60)
    print("ONNX fp32 → int8 动态量化")
    print("=" * 60)
    print()

    if args.input:
        fp32_path = Path(args.input)
        if not fp32_path.exists():
            sys.exit(f"错误：文件不存在: {args.input}")
        output_path = Path(args.output) if args.output else fp32_path.parent.parent / "int8" / fp32_path.name
        convert_to_int8(fp32_path, output_path)

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            sys.exit(f"错误：目录不存在: {args.input_dir}")
        output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / "int8"

        onnx_files = list(input_dir.glob("*.onnx"))
        if not onnx_files:
            sys.exit(f"错误：目录下无 .onnx 文件: {input_dir}")

        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"文件数: {len(onnx_files)}")
        print()

        for fp32_path in onnx_files:
            output_path = output_dir / fp32_path.name
            convert_to_int8(fp32_path, output_path)

    else:
        parser.print_help()
        sys.exit("\n错误：请指定 --input 或 --input-dir")

    print("量化完成！")
    print("注意：int8 模型仅适用于 CPU 推理（CPUExecutionProvider）")


if __name__ == "__main__":
    main()
