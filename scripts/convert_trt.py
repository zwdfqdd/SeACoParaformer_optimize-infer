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
    python scripts/convert_trt.py --input ./models/asr/split/bias_encoder.onnx --profile bias

    # fp16
    # 1. Encoder（604MB → ~317MB，计算量最大，加速收益最高）
    python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp16 --profile encoder

    # 2. CIF Predictor（23MB → ~13MB，含 NonZero/CumSum）
    python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp16 --profile cif

    # 3. Decoder（254MB，含 SANM Conv + SeACo Decoder）
    python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp16 --profile decoder

    # 4. Bias Encoder（33MB → ~17MB，热词编码 LSTM）
    python scripts/convert_trt.py --input ./models/asr/split/bias_encoder.onnx --precision fp16 --profile bias

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import tensorrt as trt
    TRT_VERSION = trt.__version__
except ImportError:
    TRT_VERSION = "unknown (需要在 TRT 容器内运行)"

# 统一参数源：从 config 动态生成分段 profile
try:
    from src.config import Settings as _Settings
except Exception:
    _Settings = None


# ============================================================
# Dynamic Shape Profiles
# 维度按数据流推导：speech → encoder → cif → decoder
# 全部由 src.config.Settings 动态生成（单一数据源，与 scheduler/导出严格一致）
# ============================================================

# 完整模型 profile（--profile asr，整体 model.onnx）：
# 不在此硬编码，统一由 config 动态生成（与分段 profile 同源，杜绝 batch/seq/热词维度漂移）。
# 历史硬编码值（batch=8/seq=289/热词=8）已废弃，与 config（batch=12/seq=134/热词=257）冲突。
def _build_asr_profiles() -> dict:
    """整体模型 profile：speech + speech_lengths + bias_embed，全部从 config 取值。"""
    if _Settings is None or not hasattr(_Settings, "get_trt_profiles"):
        raise RuntimeError(
            "无法导入 src.config.Settings 或缺少 get_trt_profiles——检测到过时 config.py。\n"
            "  请在项目根目录运行，并同步最新 src/config.py。"
        )
    s = _Settings
    mn, opt, mx = s.min_seq(), s.TRT_OPT_SEQ, s.TRT_MAX_SEQ
    ob, mb = s.opt_batch(), s.max_batch()
    fd, hd = s.FEAT_DIM, s.HIDDEN_DIM
    max_hw, opt_hw = s.MAX_HOTWORD_NUM + 1, s.OPT_HOTWORD_NUM  # 含 [sos] 哨兵 +1
    return {
        "speech": {"min": (1, mn, fd), "opt": (ob, opt, fd), "max": (mb, mx, fd)},
        "speech_lengths": {"min": (1,), "opt": (ob,), "max": (mb,)},
        "bias_embed": {"min": (1, 1, hd), "opt": (ob, opt_hw, hd), "max": (mb, max_hw, hd)},
    }


# 分段模型 profile（encoder/cif/decoder/bias）统一由 src.config.Settings.get_trt_profiles
# 动态生成，不在此硬编码，避免与 scheduler bucket/batch + 热词维度漂移。


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

    # Dynamic shape profiles：
    #   - 分段模型（encoder/cif/decoder/bias）：强制由 config 动态生成
    #     （单一数据源，与 scheduler bucket/batch + 热词维度严格一致，杜绝硬编码漂移）
    #   - 完整模型（asr）：用 ASR_PROFILES
    if profile_type in ("encoder", "cif", "decoder", "bias"):
        if _Settings is None:
            raise RuntimeError(
                "无法导入 src.config.Settings，分段 profile 必须由 config 生成。"
                "请在项目根目录运行，确保 src 可导入。"
            )
        if not hasattr(_Settings, "get_trt_profiles"):
            raise RuntimeError(
                "src.config.Settings 缺少 get_trt_profiles 方法——检测到过时的 config.py。\n"
                "  容器内的 src/config.py 是旧版本（TASK 4 之前），请同步最新代码：\n"
                "  重新构建镜像，或把最新 src/config.py 复制进容器后重试。"
            )
        profiles = _Settings.get_trt_profiles(profile_type)
    elif profile_type == "asr":
        profiles = _build_asr_profiles()
    else:
        raise ValueError(f"未知 profile_type: {profile_type}")
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

