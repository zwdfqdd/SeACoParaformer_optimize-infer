"""
ONNX → TensorRT Engine 转换脚本

将 ONNX fp32 模型转换为 TensorRT engine，支持 fp16/INT8 精度。
TRT engine 与 GPU 硬件绑定，不同 GPU 需分别构建。

环境要求：
    TensorRT 8.6.1 + CUDA 12.1
    pip install tensorrt==8.6.1

Dynamic Shape Profile（对齐 bucket 策略）：
    speech:         min=(1,34,560)   opt=(4,67,560)   max=(12,134,560)
    speech_lengths: min=(1,)         opt=(4,)         max=(12,)
    bias_embed:     min=(1,1,512)    opt=(4,4,512)    max=(12,50,512)

用法：
    # fp16 转换（推荐）
    python scripts/convert_trt.py --input ./models/asr/fp32/model.onnx --precision fp16

    # INT8 转换（需要校准缓存）
    python scripts/convert_trt.py --input ./models/asr/fp32/model.onnx --precision int8 --calib-cache ./models/asr/trt/calib.cache

    # 指定输出路径
    python scripts/convert_trt.py --input ./models/asr/fp32/model.onnx --precision fp16 --output ./models/asr/trt/a10_fp16.engine

    # 转换 bias encoder
    python scripts/convert_trt.py --input ./models/asr/fp32/model_eb.onnx --precision fp16 --profile bias
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    sys.exit("错误：需要安装 tensorrt，pip install tensorrt==8.6.1")

import numpy as np


# TRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================
# Dynamic Shape Profiles（对齐 scheduler bucket 策略）
# ============================================================
# ASR 主模型 profile
# bucket: 34/67/134 帧, batch: 1-12, feat_dim: 560
ASR_PROFILES = {
    "speech": {
        "min": (1, 34, 560),
        "opt": (4, 67, 560),
        "max": (12, 134, 560),
    },
    "speech_lengths": {
        "min": (1,),
        "opt": (4,),
        "max": (12,),
    },
    "bias_embed": {
        "min": (1, 1, 512),     # 无热词时传 1 个零向量占位
        "opt": (4, 4, 512),     # 常见：4 个热词
        "max": (12, 50, 512),   # 最大：50 个热词
    },
}

# Bias encoder profile（热词编码器，batch = hotword 数量）
BIAS_PROFILES = {
    "hotword": {
        "min": (1, 1),
        "opt": (10, 5),
        "max": (50, 20),
    },
}


def get_gpu_name() -> str:
    """获取当前 GPU 名称（用于 engine 文件命名）。"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            # 简化名称：NVIDIA GeForce RTX 2080 Ti → 2080ti
            name = name.lower()
            for prefix in ["nvidia ", "geforce ", "rtx ", "tesla "]:
                name = name.replace(prefix, "")
            name = name.strip().replace(" ", "_")
            return name
    except ImportError:
        pass

    # fallback: 从 nvidia-smi 获取
    try:
        import subprocess
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


def _force_fp32_layers(network):
    """
    强制特定层使用 fp32 精度，解决：
    1. LayerNorm fp16 溢出（TRT 警告：Running layernorm after self-attention in FP16 may cause overflow）
    2. FSMN Conv fp16 Cask 错误（Assertion isOpConsistent failed）

    策略：遍历所有层，将 LayerNorm 和 FSMN 相关 Conv 标记为 fp32。
    """
    fp32_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_name = layer.name.lower() if layer.name else ""

        # 强制 fp32 的条件：
        # 1. LayerNormalization 相关层（含 ReduceMean/Sub/Pow/Sqrt/Div 等 LN 子图）
        # 2. FSMN block 中的 Conv（名称含 fsmn）
        # 3. CIF predictor 相关（cumsum/threshold）
        force_fp32 = False

        if layer.type == trt.LayerType.NORMALIZATION:
            force_fp32 = True
        elif "layernorm" in layer_name or "layer_norm" in layer_name:
            force_fp32 = True
        elif "fsmn" in layer_name and ("conv" in layer_name or layer.type == trt.LayerType.CONVOLUTION):
            force_fp32 = True
        elif "cif" in layer_name:
            force_fp32 = True

        if force_fp32:
            layer.precision = trt.float32
            # 同时设置输出精度
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            fp32_count += 1

    print(f"    强制 fp32 层数: {fp32_count}")


def build_engine(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    profile_type: str = "asr",
    calib_cache: str = None,
    workspace_mb: int = 4096,
):
    """
    构建 TensorRT engine。

    参数：
        onnx_path: ONNX 模型路径
        output_path: 输出 engine 路径
        precision: fp16 或 int8
        profile_type: asr（主模型）或 bias（bias encoder）
        calib_cache: INT8 校准缓存路径（precision=int8 时必需）
        workspace_mb: 工作空间大小（MB）
    """
    print(f"构建 TensorRT Engine")
    print(f"  ONNX: {onnx_path}")
    print(f"  输出: {output_path}")
    print(f"  精度: {precision}")
    print(f"  GPU: {get_gpu_name()}")
    print(f"  TensorRT: {trt.__version__}")
    print(f"  Workspace: {workspace_mb}MB")
    print()

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 解析 ONNX
    print("  [1/4] 解析 ONNX 模型...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    错误: {parser.get_error(i)}")
            sys.exit("ONNX 解析失败")

    print(f"    输入: {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"    输出: {[network.get_output(i).name for i in range(network.num_outputs)]}")

    # 配置 builder
    print("  [2/4] 配置 builder...")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    # 精度设置
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("    警告：当前 GPU 不支持快速 fp16，性能可能不佳")
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        # 强制 LayerNorm 和 FSMN Conv 层使用 fp32（避免精度溢出和 Cask 错误）
        _force_fp32_layers(network)
        print("    启用 FP16（LayerNorm/FSMN Conv 强制 fp32）")
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            print("    警告：当前 GPU 不支持快速 int8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # INT8 回退到 FP16
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        _force_fp32_layers(network)
        print("    启用 INT8 + FP16 fallback（LayerNorm/FSMN Conv 强制 fp32）")

        if calib_cache and os.path.exists(calib_cache):
            config.int8_calibrator = CacheCalibrator(calib_cache)
            print(f"    校准缓存: {calib_cache}")
        else:
            print("    警告：无校准缓存，INT8 精度可能不佳")

    # Dynamic shape profile
    print("  [3/4] 设置 dynamic shape profile...")
    profiles = ASR_PROFILES if profile_type == "asr" else BIAS_PROFILES
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        name = input_tensor.name

        if name in profiles:
            p = profiles[name]
            profile.set_shape(name, p["min"], p["opt"], p["max"])
            print(f"    {name}: min={p['min']} opt={p['opt']} max={p['max']}")
        else:
            # 未在 profile 中定义的输入，使用模型中的固定 shape
            shape = input_tensor.shape
            # 替换动态维度为合理默认值
            min_shape = tuple(1 if s == -1 else s for s in shape)
            opt_shape = tuple(4 if s == -1 else s for s in shape)
            max_shape = tuple(12 if s == -1 else s for s in shape)
            profile.set_shape(name, min_shape, opt_shape, max_shape)
            print(f"    {name}: min={min_shape} opt={opt_shape} max={max_shape} (自动推断)")

    config.add_optimization_profile(profile)

    # 构建 engine
    print("  [4/4] 构建 engine（可能需要几分钟）...")
    t0 = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.perf_counter() - t0

    if serialized_engine is None:
        sys.exit("Engine 构建失败")

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    onnx_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    engine_mb = os.path.getsize(output_path) / (1024 * 1024)
    print()
    print(f"  构建完成！")
    print(f"    耗时: {build_time:.1f}s")
    print(f"    大小: ONNX {onnx_mb:.1f}MB → Engine {engine_mb:.1f}MB")
    print(f"    文件: {output_path}")


class CacheCalibrator(trt.IInt8EntropyCalibrator2):
    """从已有缓存文件加载 INT8 校准数据。"""

    def __init__(self, cache_path: str):
        super().__init__()
        self._cache_path = cache_path

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        return None

    def read_calibration_cache(self):
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self._cache_path, "wb") as f:
            f.write(cache)


def main():
    parser = argparse.ArgumentParser(description="ONNX → TensorRT Engine 转换")
    parser.add_argument("--input", required=True, help="ONNX 模型路径")
    parser.add_argument("--output", default=None, help="输出 engine 路径（默认自动命名）")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "int8"], help="推理精度")
    parser.add_argument("--profile", default="asr", choices=["asr", "bias"], help="shape profile 类型")
    parser.add_argument("--calib-cache", default=None, help="INT8 校准缓存路径")
    parser.add_argument("--workspace", type=int, default=4096, help="工作空间大小（MB）")
    args = parser.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"错误：文件不存在: {args.input}")

    # 自动生成输出路径
    if args.output is None:
        gpu_name = get_gpu_name()
        input_path = Path(args.input)
        model_name = input_path.stem  # model 或 model_eb
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
        profile_type=args.profile,
        calib_cache=args.calib_cache,
        workspace_mb=args.workspace,
    )


if __name__ == "__main__":
    main()
