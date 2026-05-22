"""
ONNX → TensorRT Engine 转换脚本

使用 trtexec 命令行工具构建 TRT engine。
基于 TRT 10.6（nvcr.io/nvidia/tensorrt:24.11-py3）。

分段模型转换命令（推荐）：
    # fp32
    # 1. Encoder（604MB → ~317MB，计算量最大，加速收益最高）
    python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --profile encoder

    # 2. CIF Predictor（23MB → ~13MB，含 NonZero/CumSum）
    python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --profile cif

    # 3. Decoder（254MB，含 SANM Conv + SeACo Decoder）
    python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --profile decoder

    # 4. Bias Encoder（33MB → ~17MB，热词编码 LSTM）
    python scripts/convert_trt.py --input ./models/asr/split/model_eb.onnx --profile bias

    # fp16
    # 1. Encoder（604MB → ~317MB，计算量最大，加速收益最高）
    python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp16 --profile encoder

    # 2. CIF Predictor（23MB → ~13MB，含 NonZero/CumSum）
    python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp16 --profile cif

    # 3. Decoder（254MB，含 SANM Conv + SeACo Decoder）
    python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp16 --profile decoder

    # 4. Bias Encoder（33MB → ~17MB，热词编码 LSTM）
    python scripts/convert_trt.py --input ./models/asr/split/model_eb.onnx --precision fp16 --profile bias

完整模型转换（不拆分，可能遇到 Cask/NonZero 问题）：

    python scripts/convert_trt.py --input ./models/asr/fp32/model.onnx --precision fp16 --profile asr

其他选项：

    # 指定输出路径
    python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp16 --output ./models/asr/trt/a10_encoder_fp16.engine

    # 增大 workspace（大显存 GPU）
    python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp16 --workspace 4096
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import tensorrt as trt
    TRT_VERSION = trt.__version__
except ImportError:
    TRT_VERSION = "unknown (需要在 TRT 容器内运行)"


# ============================================================
# Dynamic Shape Profiles
# 维度按数据流推导：speech → encoder → cif → decoder
# 输入规格：speech min=(1,8,560) opt=(4,128,560) max=(8,289,560)
# ============================================================

# 完整模型 profile（不拆分时使用）
ASR_PROFILES = {
    "speech": {
        "min": (1, 8, 560),
        "opt": (1, 128, 560),
        "max": (8, 289, 560),
    },
    "speech_lengths": {
        "min": (1,),
        "opt": (1,),
        "max": (8,),
    },
    "bias_embed": {
        "min": (1, 1, 512),
        "opt": (1, 4, 512),
        "max": (8, 8, 512),
    },
}

# Encoder：speech(B,T,560) + speech_lengths(B,) → encoder_out(B,T',512)
# min seq_len=16（encoder 内部 reshape 要求最小值）
# opt batch=1（避免 batch*seq_len 与 hidden_dim 冲突）
ENCODER_PROFILES = {
    "speech": {
        "min": (1, 8, 560),
        "opt": (1, 128, 560),
        "max": (8, 289, 560),
    },
    "speech_lengths": {
        "min": (1,),
        "opt": (1,),
        "max": (8,),
    },
}

# CIF：encoder_out(B,T,512) → acoustic_embeds(B,N,512)
# T 与 encoder 输出一致
CIF_PROFILES = {
    "encoder_out": {
        "min": (1, 8, 512),
        "opt": (1, 128, 512),
        "max": (8, 289, 512),
    },
}

# Decoder：acoustic_embeds(B,N,512) + encoder_out(B,T,512) + bias_embed(B,H,512) → logits
# Decoder：acoustic_embeds(B,N,512) + encoder_out(B,T,512) → logits
# N(token数) ≈ T/5（CIF 压缩比），min 需 ≥ SANM conv kernel_size(~11)
DECODER_PROFILES = {
    "acoustic_embeds": {
        "min": (1, 2, 512),
        "opt": (1, 128, 512),
        "max": (8, 289, 512),
    },
    "acoustic_embeds_lens": {
        "min": (1,),
        "opt": (1,),
        "max": (8,),
    },
    "encoder_out": {
        "min": (1, 8, 512),
        "opt": (1, 128, 512),
        "max": (8, 289, 512),
    },
    "encoder_out_lens": {
        "min": (1,),
        "opt": (1,),
        "max": (8,),
    },
}

# Bias Encoder：hotword(H,L) → hw_embed(H,1,512)
BIAS_PROFILES = {
    "hotword": {
        "min": (1, 1),
        "opt": (1, 4),
        "max": (8, 8),
    },
}


def get_gpu_name() -> str:
    """获取当前 GPU 名称（用于 engine 文件命名）。"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            for prefix in ["nvidia ", "geforce ", "rtx ", "tesla "]:
                name = name.replace(prefix, "")
            return name.strip().replace(" ", "_")
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            name = result.stdout.strip().split("\n")[0].lower()
            for prefix in ["nvidia ", "geforce ", "rtx ", "tesla "]:
                name = name.replace(prefix, "")
            return name.strip().replace(" ", "_")
    except Exception:
        pass
    return "unknown_gpu"


def build_engine(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    profile_type: str = "asr",
):
    """
    使用 trtexec 构建 TensorRT engine。

    trtexec 优势：
    - 支持 --tacticSources=-EDGE_MASK_CONVOLUTIONS 精确禁用 Cask
    - 内存管理更好，不会因 NonZero 等算子 OOM
    - 自动处理 layer precision fallback
    """
    print(f"构建 TensorRT Engine (trtexec)")
    print(f"  ONNX: {onnx_path}")
    print(f"  输出: {output_path}")
    print(f"  精度: {precision}")
    print(f"  GPU: {get_gpu_name()}")
    print(f"  TensorRT: {TRT_VERSION}")
    print()

    # 构建 trtexec 命令
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        # 跳过构建后的推理验证
        "--skipInference",
    ]

    # 精度
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.extend(["--int8", "--fp16"])
    # fp32: 不加任何精度 flag

    # Dynamic shape profiles
    if profile_type == "encoder":
        profiles = ENCODER_PROFILES
    elif profile_type == "cif":
        profiles = CIF_PROFILES
    elif profile_type == "decoder":
        profiles = DECODER_PROFILES
    elif profile_type == "bias":
        profiles = BIAS_PROFILES
    else:
        profiles = ASR_PROFILES
    min_shapes = []
    opt_shapes = []
    max_shapes = []

    for name, p in profiles.items():
        min_str = "x".join(str(d) for d in p["min"])
        opt_str = "x".join(str(d) for d in p["opt"])
        max_str = "x".join(str(d) for d in p["max"])
        min_shapes.append(f"{name}:{min_str}")
        opt_shapes.append(f"{name}:{opt_str}")
        max_shapes.append(f"{name}:{max_str}")

    cmd.append(f"--minShapes={','.join(min_shapes)}")
    cmd.append(f"--optShapes={','.join(opt_shapes)}")
    cmd.append(f"--maxShapes={','.join(max_shapes)}")

    # 构建超时和日志
    cmd.append("--verbose")

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"  命令: {' '.join(cmd[:6])}...")
    print(f"  完整命令:")
    for c in cmd:
        print(f"    {c}")
    print()
    print("  构建中（可能需要 5-15 分钟）...")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 分钟超时
        )
    except FileNotFoundError:
        sys.exit("错误：trtexec 不在 PATH 中。请确认 TensorRT 已正确安装。")
    except subprocess.TimeoutExpired:
        sys.exit("错误：trtexec 构建超时（30分钟）")

    build_time = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"\n  trtexec 失败 (exit code={result.returncode})")
        # 输出最后 2000 字符的错误信息
        stderr = result.stderr or result.stdout or ""
        error_lines = [l for l in stderr.split("\n") if "[E]" in l or "Error" in l]
        if error_lines:
            print("  错误信息:")
            for line in error_lines[-10:]:
                print(f"    {line}")
        sys.exit("Engine 构建失败")

    if not os.path.exists(output_path):
        sys.exit(f"错误：engine 文件未生成: {output_path}")

    onnx_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    engine_mb = os.path.getsize(output_path) / (1024 * 1024)
    print()
    print(f"  构建完成！")
    print(f"    耗时: {build_time:.1f}s")
    print(f"    大小: ONNX {onnx_mb:.1f}MB → Engine {engine_mb:.1f}MB")
    print(f"    文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ONNX → TensorRT Engine 转换")
    parser.add_argument("--input", required=True, help="ONNX 模型路径")
    parser.add_argument("--output", default=None, help="输出 engine 路径（默认自动命名）")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"], help="推理精度（默认 fp32）")
    parser.add_argument("--profile", default="asr", choices=["asr", "encoder", "cif", "decoder", "bias"], help="shape profile 类型")
    args = parser.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"错误：文件不存在: {args.input}")

    # 自动生成输出路径
    if args.output is None:
        gpu_name = get_gpu_name()
        input_path = Path(args.input)
        model_name = input_path.stem
        output_dir = input_path.parent.parent / "trt"
        output_path = str(output_dir / f"{gpu_name}_{model_name}_{args.precision}.engine")
    else:
        output_path = args.output

    print("=" * 60)
    print("ONNX → TensorRT Engine 转换")
    print("=" * 60)
    print()

    build_engine(
        onnx_path=args.input,
        output_path=output_path,
        precision=args.precision,
        profile_type=args.profile
    )


if __name__ == "__main__":
    main()

