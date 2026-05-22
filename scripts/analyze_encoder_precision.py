"""
Encoder 逐层精度分析脚本（基于 Polygraphy + 选择性层输出标记）

对比 ONNX fp32（baseline）与 TRT fp16 关键内部层输出，精确定位精度敏感层。
支持指定问题层 fallback fp32 后继续分析后续层的误差传播情况。

原理：
    1. 从 ONNX 图中提取 encoder 各 block 的关键输出节点名
    2. 只标记这些关键节点为输出（避免 mark all 导致 TRT 构建失败）
    3. 同时运行 ONNX Runtime（fp32）和 TRT（fp16/混合精度）
    4. 逐层对比输出，计算 max/mean abs diff、relative diff
    5. 支持迭代：指定问题层 fallback fp32，重新分析后续层

用法：
    # 第一轮：全 fp16 分析，找出所有问题层
    python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_30s.wav

    # 第二轮：将第一轮发现的问题层 fallback fp32，继续分析
    python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_30s.wav \
        --fp32-layers "/encoder_export/encoders/encoders.0/norm1/Add" \
                      "/encoder_export/encoders/encoders.0/self_attn/MatMul_1"

    # 从 JSON 报告中读取 fallback 层列表
    python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_30s.wav \
        --fp32-layers-from report_encoder_precision.json

    # 使用层名关键词批量 fallback（如所有 norm）
    python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_30s.wav \
        --fp32-pattern "norm1" "norm2"

依赖：
    pip install polygraphy onnxruntime-gpu tensorrt numpy onnx

注意：
    需要在 TRT 容器内运行（已安装 TensorRT 10.6 + Polygraphy）
"""

import argparse
import sys
import os
import re
import json
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import onnx
except ImportError:
    sys.exit("需要安装 onnx: pip install onnx")

try:
    from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
    from polygraphy.backend.trt import (
        TrtRunner,
        EngineFromNetwork,
        NetworkFromOnnxPath,
        CreateConfig,
        Profile,
    )
    from polygraphy.backend.onnx import OnnxFromPath, ModifyOutputs as OnnxModifyOutputs, BytesFromOnnx
    from polygraphy.backend.trt import ModifyNetworkOutputs as TrtModifyOutputs
    from polygraphy.comparator import Comparator
    from polygraphy.logger import G_LOGGER
except ImportError:
    sys.exit(
        "需要安装 polygraphy：pip install polygraphy\n"
        "建议在 TRT 容器内运行（nvcr.io/nvidia/tensorrt:24.11-py3）"
    )

try:
    import tensorrt as trt
except ImportError:
    sys.exit("需要安装 tensorrt")


# Encoder shape profile（与 convert_trt.py 一致）
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


def generate_dummy_input(seq_len: int = 67, batch: int = 1) -> dict:
    """生成模拟输入数据。"""
    np.random.seed(42)
    speech = np.random.randn(batch, seq_len, 560).astype(np.float32)
    speech_lengths = np.array([seq_len] * batch, dtype=np.int64)
    return {"speech": speech, "speech_lengths": speech_lengths}


def generate_input_from_audio(audio_path: str) -> dict:
    """从真实音频生成 encoder 输入（经过特征提取）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import soundfile as sf
    from src.feature_extractor import extract_features, load_cmvn

    pcm, sr = sf.read(audio_path, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]

    cmvn_path = os.path.join("./models/asr", "am.mvn")
    if not Path(cmvn_path).exists():
        sys.exit(f"CMVN 文件不存在: {cmvn_path}")

    cmvn_mean, cmvn_istd = load_cmvn(cmvn_path)
    features = extract_features(pcm, sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)

    speech = features[np.newaxis, :, :].astype(np.float32)
    speech_lengths = np.array([features.shape[0]], dtype=np.int64)
    print(f"  音频: {audio_path}, 时长: {len(pcm)/sr:.2f}s, 特征: {speech.shape}")
    return {"speech": speech, "speech_lengths": speech_lengths}


def classify_layer(name: str) -> str:
    """根据层名称分类。"""
    name_lower = name.lower()
    if "layernorm" in name_lower or "layer_norm" in name_lower or "/norm" in name_lower:
        return "LayerNorm"
    elif "softmax" in name_lower:
        return "Softmax"
    elif "matmul" in name_lower or "gemm" in name_lower:
        return "MatMul/Gemm"
    elif "add" in name_lower and "attention" not in name_lower:
        return "Add(残差)"
    elif "attention" in name_lower or "self_attn" in name_lower or "attn" in name_lower:
        return "Attention"
    elif "conv" in name_lower:
        return "Conv"
    elif "relu" in name_lower or "gelu" in name_lower or "silu" in name_lower:
        return "Activation"
    elif "mul" in name_lower:
        return "Mul"
    elif "reduce" in name_lower:
        return "Reduce"
    elif "embed" in name_lower or "pos" in name_lower:
        return "Embedding/Position"
    else:
        return "Other"


def extract_layer_index(name: str) -> tuple:
    """提取层名中的数字索引（近似拓扑顺序）。"""
    numbers = re.findall(r'\d+', name)
    return tuple(int(n) for n in numbers) if numbers else (9999,)


def load_fp32_layers_from_json(json_path: str) -> list[str]:
    """从之前的分析报告 JSON 中加载 fallback 层列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    layers = data.get("fallback_layers", [])
    print(f"  从 {json_path} 加载 {len(layers)} 个 fallback 层")
    return layers


def extract_key_outputs_from_onnx(onnx_path: str) -> list[str]:
    """
    从 ONNX 图中提取 encoder 各 block 的关键输出节点名。

    策略：提取每个 encoder layer 中以下关键节点的输出：
    - Add（残差连接输出，代表每个子层的最终输出）
    - MatMul（attention 计算的关键步骤）
    - Softmax（attention 权重）
    - Mul（scale 操作）
    - ReduceMean（LayerNorm 内部）
    - Conv（SANM 卷积）
    - 最终输出 encoder_out

    只选择 float 类型的输出（跳过 shape/index 等 int 输出）。
    """
    print("  从 ONNX 图中提取关键层输出节点...")
    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph

    # 收集所有节点输出名
    all_output_names = set()
    for node in graph.node:
        for output in node.output:
            if output:
                all_output_names.add(output)

    # 已有的图输出
    existing_outputs = {o.name for o in graph.output}

    # 关键节点类型（这些节点的输出对精度分析最有价值）
    key_op_types = {"Add", "MatMul", "Softmax", "Mul", "ReduceMean", "Conv", "Relu", "Sigmoid"}

    # 关键路径关键词（encoder block 内部）
    key_path_patterns = [
        "encoders",      # encoder layers
        "self_attn",     # self attention
        "feed_forward",  # FFN
        "norm",          # LayerNorm
        "conv",          # SANM conv
    ]

    key_outputs = []
    seen = set()

    for node in graph.node:
        # 只关注 encoder 相关节点
        node_name = node.name or ""
        is_encoder_node = any(p in node_name.lower() for p in key_path_patterns)

        if not is_encoder_node and node.op_type not in key_op_types:
            continue

        # 只选择关键 op 类型
        if node.op_type in key_op_types:
            for output in node.output:
                if output and output not in existing_outputs and output not in seen:
                    key_outputs.append(output)
                    seen.add(output)

    # 如果提取太少，放宽条件
    if len(key_outputs) < 10:
        print(f"  关键节点较少（{len(key_outputs)}），扩大搜索范围...")
        for node in graph.node:
            if "encoder" in (node.name or "").lower():
                for output in node.output:
                    if output and output not in existing_outputs and output not in seen:
                        key_outputs.append(output)
                        seen.add(output)

    print(f"  提取到 {len(key_outputs)} 个关键层输出节点")
    return key_outputs


def extract_key_outputs_by_block(onnx_path: str) -> dict[str, list[str]]:
    """
    按 encoder block 分组提取关键输出。

    返回: {block_name: [output_names]}
    """
    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph

    existing_outputs = {o.name for o in graph.output}

    blocks = {}  # block_id -> [output_names]

    for node in graph.node:
        node_name = node.name or ""

        # 匹配 encoder block: encoders.0, encoders.1, ...
        match = re.search(r'encoders[\./](\d+)', node_name)
        if match:
            block_id = f"encoder_block_{match.group(1)}"
        elif "encoder" in node_name.lower():
            block_id = "encoder_other"
        else:
            continue

        if block_id not in blocks:
            blocks[block_id] = []

        # 只取 Add（残差输出）和 MatMul（attention 核心）
        if node.op_type in {"Add", "MatMul", "Softmax", "Conv"}:
            for output in node.output:
                if output and output not in existing_outputs:
                    blocks[block_id].append(output)

    return blocks


def run_full_fp16_analysis(
    onnx_path: str,
    input_data: dict,
    threshold: float = 0.01,
    top_n: int = 50,
    output_file: str | None = None,
):
    """
    全 fp16 分析：对比 ONNX fp32 vs TRT fp16 关键内部层。

    使用选择性标记（而非 mark all）避免 TRT 构建失败。
    """
    print("=" * 70)
    print("Encoder fp16 逐层精度分析（选择性层输出标记）")
    print("=" * 70)
    print(f"  ONNX 模型:   {onnx_path}")
    print(f"  输入 shape:  speech={input_data['speech'].shape}")
    print(f"  误差阈值:   {threshold}")
    print(f"  时间:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    G_LOGGER.module_severity = G_LOGGER.WARNING

    # ====== 提取关键层输出 ======
    print("[1/4] 提取 encoder 关键层输出节点...")
    key_outputs = extract_key_outputs_from_onnx(onnx_path)

    if not key_outputs:
        sys.exit("未能提取到关键层输出节点，请检查 ONNX 模型结构")

    # ====== 构建 Profile ======
    profile = Profile()
    for name, shapes in ENCODER_PROFILES.items():
        profile.add(name, min=shapes["min"], opt=shapes["opt"], max=shapes["max"])

    # ====== ONNX Runner（标记关键层输出） ======
    print(f"\n[2/4] 构建 ONNX fp32 Runner（标记 {len(key_outputs)} 个关键层输出）...")
    onnx_loader = OnnxModifyOutputs(OnnxFromPath(onnx_path), outputs=key_outputs)
    onnx_runner = OnnxrtRunner(SessionFromOnnx(BytesFromOnnx(onnx_loader)))

    # ====== TRT Runner（fp16，标记关键层输出） ======
    print(f"[3/4] 构建 TRT fp16 Runner（标记 {len(key_outputs)} 个关键层输出）...")
    network_loader = NetworkFromOnnxPath(onnx_path)
    network_with_outputs = TrtModifyOutputs(network_loader, outputs=key_outputs)
    trt_runner = TrtRunner(
        EngineFromNetwork(
            network_with_outputs,
            config=CreateConfig(
                fp16=True,
                profiles=[profile],
            ),
        )
    )

    # ====== 运行对比 ======
    print("[4/4] 运行推理并收集层输出...")
    print()

    feed_dict = {k: v for k, v in input_data.items()}
    runners = [onnx_runner, trt_runner]
    run_results = Comparator.run(runners, data_loader=[feed_dict])

    # ====== 逐层误差计算 ======
    onnx_results = run_results[onnx_runner.name]
    trt_results = run_results[trt_runner.name]

    if not onnx_results or not trt_results:
        sys.exit("推理结果为空，请检查模型和输入")

    onnx_iter = onnx_results[0]
    trt_iter = trt_results[0]

    common_outputs = set(onnx_iter.keys()) & set(trt_iter.keys())
    print(f"  ONNX 输出层数: {len(onnx_iter)}")
    print(f"  TRT  输出层数: {len(trt_iter)}")
    print(f"  共有输出层数:  {len(common_outputs)}")
    print()

    layer_errors = compute_layer_errors(onnx_iter, trt_iter, common_outputs, threshold)

    # 生成报告
    sensitive_layers = [l for l in layer_errors if l["max_abs_diff"] > threshold]
    safe_layers = [l for l in layer_errors if l["max_abs_diff"] <= threshold]

    report_lines = generate_report(
        onnx_path, input_data, threshold, top_n,
        layer_errors, sensitive_layers, safe_layers, [],
        [], None, is_mixed=False,
    )

    report_text = "\n".join(report_lines)
    print(report_text)

    save_reports(output_file, report_text, onnx_path, input_data, threshold,
                 layer_errors, sensitive_layers, [], None)

    return not bool(sensitive_layers)


def run_mixed_precision_analysis(
    onnx_path: str,
    input_data: dict,
    fp32_layers: list[str],
    fp32_patterns: list[str] | None = None,
    threshold: float = 0.01,
    top_n: int = 50,
    output_file: str | None = None,
):
    """
    混合精度分析：将指定层 fallback fp32 后，对比剩余层精度。

    使用 TRT Python API 手动设置层精度，构建混合精度 engine。
    """
    print("=" * 70)
    print("Encoder 混合精度分析（指定层 fp32 fallback）")
    print("=" * 70)
    print(f"  ONNX 模型:     {onnx_path}")
    print(f"  输入 shape:    speech={input_data['speech'].shape}")
    print(f"  fp32 层数:     {len(fp32_layers)}")
    print(f"  fp32 模式:     {fp32_patterns if fp32_patterns else '无'}")
    print(f"  误差阈值:     {threshold}")
    print(f"  时间:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    G_LOGGER.module_severity = G_LOGGER.WARNING

    # ====== 提取关键层输出 ======
    print("[1/5] 提取 encoder 关键层输出节点...")
    key_outputs = extract_key_outputs_from_onnx(onnx_path)

    # ====== ONNX baseline ======
    print(f"\n[2/5] 构建 ONNX fp32 baseline（{len(key_outputs)} 个关键层输出）...")
    profile = Profile()
    for name, shapes in ENCODER_PROFILES.items():
        profile.add(name, min=shapes["min"], opt=shapes["opt"], max=shapes["max"])

    onnx_loader = OnnxModifyOutputs(OnnxFromPath(onnx_path), outputs=key_outputs)
    onnx_runner = OnnxrtRunner(SessionFromOnnx(BytesFromOnnx(onnx_loader)))

    # 运行 ONNX
    with onnx_runner:
        onnx_outputs = onnx_runner.infer(input_data)

    print(f"  ONNX 输出层数: {len(onnx_outputs)}")

    # ====== TRT 混合精度构建 ======
    print(f"\n[3/5] 构建 TRT 混合精度 engine...")
    import torch

    engine, trt_layer_names = build_trt_mixed_precision_engine(
        onnx_path, key_outputs, fp32_layers, fp32_patterns
    )

    if engine is None:
        sys.exit("混合精度 engine 构建失败")

    # ====== TRT 推理 ======
    print(f"\n[4/5] 运行 TRT 混合精度推理...")
    trt_outputs = run_trt_inference(engine, input_data)
    print(f"  TRT 输出层数: {len(trt_outputs)}")

    # ====== 逐层对比 ======
    print(f"\n[5/5] 逐层精度对比...")
    common_outputs = set(onnx_outputs.keys()) & set(trt_outputs.keys())
    print(f"  共有输出层: {len(common_outputs)}")

    layer_errors = compute_layer_errors(onnx_outputs, trt_outputs, common_outputs, threshold, fp32_layers, fp32_patterns)

    sensitive_layers = [l for l in layer_errors if l["max_abs_diff"] > threshold and not l.get("is_fp32_fallback")]
    safe_layers = [l for l in layer_errors if l["max_abs_diff"] <= threshold]

    report_lines = generate_report(
        onnx_path, input_data, threshold, top_n,
        layer_errors, sensitive_layers, safe_layers, [],
        fp32_layers, fp32_patterns, is_mixed=True,
    )

    report_text = "\n".join(report_lines)
    print(report_text)

    save_reports(output_file, report_text, onnx_path, input_data, threshold,
                 layer_errors, sensitive_layers, fp32_layers, fp32_patterns)

    return not bool(sensitive_layers)


def build_trt_mixed_precision_engine(
    onnx_path: str,
    key_outputs: list[str],
    fp32_layers: list[str],
    fp32_patterns: list[str] | None = None,
):
    """
    构建混合精度 TRT engine，标记关键层为输出。

    返回: (engine, all_layer_names)
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX 解析错误: {parser.get_error(i)}")
            return None, []

    # 标记关键层为输出
    key_output_set = set(key_outputs)
    marked_count = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            output = layer.get_output(j)
            if output and output.name in key_output_set and not output.is_network_output:
                network.mark_output(output)
                marked_count += 1

    print(f"  标记 {marked_count} 个关键层为输出")

    # 设置混合精度
    all_layer_names = []
    fp32_count = 0
    fp16_count = 0
    fp32_applied = []

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        all_layer_names.append(layer.name)

        should_fp32 = (
            layer.name in fp32_layers or
            any(p.lower() in layer.name.lower() for p in (fp32_patterns or []))
        )

        if should_fp32:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            fp32_count += 1
            fp32_applied.append(layer.name)
        else:
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float16)
            fp16_count += 1

    print(f"  网络总层数: {network.num_layers}")
    print(f"  混合精度: {fp32_count} 层 fp32, {fp16_count} 层 fp16")

    # 构建 config
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    trt_profile = builder.create_optimization_profile()
    for name, shapes in ENCODER_PROFILES.items():
        trt_profile.set_shape(name, shapes["min"], shapes["opt"], shapes["max"])
    config.add_optimization_profile(trt_profile)

    print(f"\n  构建混合精度 engine（可能需要 5-15 分钟）...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("  engine 构建失败")
        return None, all_layer_names

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print(f"  engine 构建成功")

    return engine, all_layer_names


def run_trt_inference(engine, input_data: dict) -> dict:
    """运行 TRT engine 推理，返回所有输出。"""
    import torch

    context = engine.create_execution_context()

    # 设置输入 shape
    for name, data in input_data.items():
        context.set_input_shape(name, data.shape)

    # 分配输入
    d_inputs = {}
    for name, data in input_data.items():
        t = torch.from_numpy(data).cuda().contiguous()
        d_inputs[name] = t
        context.set_tensor_address(name, t.data_ptr())

    # 分配输出
    d_outputs = {}
    output_names = []
    for i in range(engine.num_io_tensors):
        tname = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(tname)
        if mode == trt.TensorIOMode.OUTPUT:
            output_names.append(tname)
            shape = list(context.get_tensor_shape(tname))
            for j, s in enumerate(shape):
                if s <= 0:
                    shape[j] = 512  # 预分配
            t = torch.zeros(shape, dtype=torch.float32, device="cuda")
            d_outputs[tname] = t
            context.set_tensor_address(tname, t.data_ptr())

    # 执行
    stream = torch.cuda.Stream()
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    # 收集输出
    results = {}
    for tname, t in d_outputs.items():
        actual_shape = tuple(context.get_tensor_shape(tname))
        if all(s > 0 for s in actual_shape):
            slices = tuple(slice(0, s) for s in actual_shape)
            results[tname] = t[slices].cpu().numpy()
        else:
            results[tname] = t.cpu().numpy()

    return results


def compute_layer_errors(
    onnx_outputs: dict,
    trt_outputs: dict,
    common_outputs: set,
    threshold: float,
    fp32_layers: list[str] | None = None,
    fp32_patterns: list[str] | None = None,
) -> list[dict]:
    """计算各层误差。"""
    layer_errors = []

    for output_name in common_outputs:
        onnx_val = np.array(onnx_outputs[output_name], dtype=np.float64)
        trt_val = np.array(trt_outputs[output_name], dtype=np.float64)

        if onnx_val.shape != trt_val.shape:
            if len(onnx_val.shape) != len(trt_val.shape):
                continue
            min_shape = tuple(min(a, b) for a, b in zip(onnx_val.shape, trt_val.shape))
            slices = tuple(slice(0, s) for s in min_shape)
            onnx_val = onnx_val[slices]
            trt_val = trt_val[slices]

        if onnx_val.size == 0:
            continue

        abs_diff = np.abs(onnx_val - trt_val)
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))
        median_abs = float(np.median(abs_diff))

        denom = np.abs(onnx_val) + 1e-8
        rel_diff = abs_diff / denom
        max_rel = float(np.max(rel_diff))
        mean_rel = float(np.mean(rel_diff))

        exceed_ratio = float(np.mean(abs_diff > threshold))

        onnx_min, onnx_max = float(np.min(onnx_val)), float(np.max(onnx_val))
        trt_min, trt_max = float(np.min(trt_val)), float(np.max(trt_val))

        # 判断是否已 fallback
        is_fp32 = False
        if fp32_layers:
            is_fp32 = output_name in fp32_layers or any(
                p.lower() in output_name.lower() for p in (fp32_patterns or [])
            )

        layer_errors.append({
            "name": output_name,
            "category": classify_layer(output_name),
            "shape": list(onnx_val.shape),
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "median_abs_diff": median_abs,
            "max_rel_diff": max_rel,
            "mean_rel_diff": mean_rel,
            "exceed_ratio": exceed_ratio,
            "onnx_range": [onnx_min, onnx_max],
            "trt_range": [trt_min, trt_max],
            "is_fp32_fallback": is_fp32,
        })

    layer_errors.sort(key=lambda x: x["max_abs_diff"], reverse=True)
    return layer_errors


def generate_report(
    onnx_path, input_data, threshold, top_n,
    layer_errors, sensitive_layers, safe_layers, fp32_layers_result,
    fp32_layers_input, fp32_patterns, is_mixed,
):
    """生成分析报告。"""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    if is_mixed:
        lines.append("Encoder 混合精度分析报告（指定层 fp32 fallback 后）")
    else:
        lines.append("Encoder 全 fp16 精度分析报告")
    lines.append("=" * 70)
    lines.append(f"时间:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"ONNX:       {onnx_path}")
    lines.append(f"输入:       speech={list(input_data['speech'].shape)}")
    lines.append(f"阈值:       max_abs_diff > {threshold} 标记为精度敏感")
    lines.append(f"分析层数:   {len(layer_errors)}")
    lines.append("")

    if is_mixed:
        lines.append(f"已 fallback fp32 的层: {len(fp32_layers_input)} 个")
        if fp32_patterns:
            lines.append(f"fp32 模式匹配:        {fp32_patterns}")
        lines.append("")

    lines.append(f"仍然精度敏感的层: {len(sensitive_layers)} 个")
    lines.append(f"精度安全层:       {len(safe_layers)} 个")
    lines.append("")

    # 按类别统计
    category_stats = {}
    for layer in sensitive_layers:
        cat = layer["category"]
        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "max_error": 0}
        category_stats[cat]["count"] += 1
        category_stats[cat]["max_error"] = max(category_stats[cat]["max_error"], layer["max_abs_diff"])

    if category_stats:
        lines.append("-" * 70)
        lines.append("【按类别统计精度敏感层】")
        lines.append("-" * 70)
        sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]["max_error"], reverse=True)
        for cat, stats in sorted_cats:
            lines.append(f"  {cat:20s}: {stats['count']:3d} 个, 最大误差: {stats['max_error']:.6e}")
        lines.append("")

    # Top-N 表格
    show_n = min(top_n, len(layer_errors))
    lines.append("-" * 70)
    lines.append(f"【误差最大的 Top-{show_n} 层】")
    lines.append("-" * 70)
    lines.append("")
    header = f"{'#':<4} {'层名称':<48} {'类别':<14} {'max_abs':<12} {'mean_abs':<12} {'超阈值%':<7}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, layer in enumerate(layer_errors[:top_n], 1):
        name_short = layer["name"][:46] if len(layer["name"]) > 46 else layer["name"]
        exceed_pct = f"{layer['exceed_ratio']*100:.1f}%"
        lines.append(
            f"{i:<4} {name_short:<48} {layer['category']:<14} "
            f"{layer['max_abs_diff']:<12.4e} {layer['mean_abs_diff']:<12.4e} "
            f"{exceed_pct:<7}"
        )
    lines.append("")

    # 敏感层详情
    if sensitive_layers:
        lines.append("-" * 70)
        lines.append(f"【精度敏感层详情（共 {len(sensitive_layers)} 个）】")
        lines.append("-" * 70)
        for i, layer in enumerate(sensitive_layers[:top_n], 1):
            lines.append(f"\n  [{i}] {layer['name']}")
            lines.append(f"      类别:           {layer['category']}")
            lines.append(f"      shape:          {layer['shape']}")
            lines.append(f"      max_abs_diff:   {layer['max_abs_diff']:.6e}")
            lines.append(f"      mean_abs_diff:  {layer['mean_abs_diff']:.6e}")
            lines.append(f"      max_rel_diff:   {layer['max_rel_diff']:.6e}")
            lines.append(f"      超阈值比例:     {layer['exceed_ratio']*100:.2f}%")
            lines.append(f"      ONNX 值域:      [{layer['onnx_range'][0]:.4f}, {layer['onnx_range'][1]:.4f}]")
            lines.append(f"      TRT  值域:      [{layer['trt_range'][0]:.4f}, {layer['trt_range'][1]:.4f}]")
        lines.append("")

    # 误差传播分析
    lines.append("-" * 70)
    lines.append("【误差传播分析】")
    lines.append("-" * 70)
    sensitive_by_topo = sorted(sensitive_layers, key=lambda x: extract_layer_index(x["name"]))
    if sensitive_by_topo:
        lines.append("  误差最早出现的层（可能是源头）：")
        for i, layer in enumerate(sensitive_by_topo[:5], 1):
            lines.append(f"    {i}. {layer['name']}")
            lines.append(f"       类别: {layer['category']}, max_abs: {layer['max_abs_diff']:.4e}")
    else:
        lines.append("  无精度敏感层，误差已消除。")
    lines.append("")

    # 结论
    lines.append("=" * 70)
    lines.append("【结论与下一步】")
    lines.append("=" * 70)
    if not sensitive_layers:
        if is_mixed:
            lines.append("  当前混合精度配置已消除所有精度问题。")
            lines.append("  可以使用此配置构建最终 encoder engine。")
            lines.append("")
            lines.append("  构建命令：")
            lines.append("    python scripts/convert_trt_mixed.py \\")
            lines.append("      --onnx ./models/asr/split/encoder.onnx \\")
            lines.append("      --fp32-layers-from report_encoder_precision.json")
        else:
            lines.append("  所有关键层 fp16 精度合格，无需 fallback。")
    else:
        lines.append(f"  仍有 {len(sensitive_layers)} 个层精度超标。")
        lines.append("")
        lines.append("  建议下一步：将以下层追加到 fp32 fallback 列表，重新分析：")
        lines.append("")
        lines.append("    python scripts/analyze_encoder_precision.py \\")
        lines.append("      --audio test_data/audio_16000_30s.wav \\")
        if fp32_layers_input:
            lines.append(f"      --fp32-layers-from report_encoder_precision.json \\")
        else:
            lines.append(f"      --fp32-layers \\")
            for name in [l["name"] for l in sensitive_layers[:5]]:
                lines.append(f'        "{name}" \\')
            if len(sensitive_layers) > 5:
                lines.append(f"        # ... 共 {len(sensitive_layers)} 个（见 JSON）")
        lines.append("")
        if category_stats:
            top_cat = sorted(category_stats.items(), key=lambda x: x[1]["max_error"], reverse=True)[0][0]
            lines.append(f"  或按类别批量 fallback：--fp32-pattern \"{top_cat}\"")
    lines.append("")

    return lines


def save_reports(output_file, report_text, onnx_path, input_data, threshold,
                 layer_errors, sensitive_layers, fp32_layers_input, fp32_patterns):
    """保存文本和 JSON 报告。"""
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n文本报告已保存: {output_file}")

    json_path = output_file.replace(".txt", ".json") if output_file else "report_encoder_precision.json"

    # 合并已有 fallback 和新发现的敏感层
    all_fallback = list(fp32_layers_input or [])
    for l in sensitive_layers:
        if l["name"] not in all_fallback:
            all_fallback.append(l["name"])

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "onnx_path": onnx_path,
        "input_shape": {k: list(v.shape) for k, v in input_data.items()},
        "threshold": threshold,
        "total_layers_analyzed": len(layer_errors),
        "sensitive_count": len(sensitive_layers),
        "previous_fp32_layers": list(fp32_layers_input or []),
        "fp32_patterns": fp32_patterns or [],
        "new_sensitive_layers": [l["name"] for l in sensitive_layers],
        "fallback_layers": all_fallback,
        "sensitive_layers_detail": sensitive_layers,
        "all_layers": layer_errors,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"JSON 报告已保存: {json_path}")
    print(f"  fallback_layers 总计: {len(all_fallback)} 个")


def main():
    parser = argparse.ArgumentParser(
        description="Encoder 逐层精度分析（支持迭代 fallback）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 第一轮：全 fp16 分析
  python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_30s.wav

  # 第二轮：指定问题层 fallback fp32 后继续分析
  python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_30s.wav \\
      --fp32-layers "/encoder_export/encoders/encoders.0/norm1/Add"

  # 从上一轮 JSON 报告加载 fallback 列表
  python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_30s.wav \\
      --fp32-layers-from report_encoder_precision.json

  # 按类别批量 fallback
  python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_30s.wav \\
      --fp32-pattern "norm1" "norm2"
        """,
    )
    parser.add_argument(
        "--onnx", default="./models/asr/split/encoder.onnx",
        help="Encoder ONNX 模型路径",
    )
    parser.add_argument(
        "--audio", default=None,
        help="使用真实音频生成输入（推荐）",
    )
    parser.add_argument(
        "--seq-len", type=int, default=67,
        help="模拟输入序列长度（不指定 --audio 时使用）",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="batch size",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="精度误差阈值（默认 0.01）",
    )
    parser.add_argument(
        "--top", type=int, default=50,
        help="显示 Top-N 误差最大的层",
    )
    parser.add_argument(
        "--output", default="report_encoder_precision.txt",
        help="报告输出路径",
    )
    # ====== 混合精度参数 ======
    parser.add_argument(
        "--fp32-layers", nargs="*", default=None,
        help="指定需要 fallback fp32 的层名称列表",
    )
    parser.add_argument(
        "--fp32-layers-from", default=None,
        help="从 JSON 报告文件加载 fallback 层列表",
    )
    parser.add_argument(
        "--fp32-pattern", nargs="*", default=None,
        help="按关键词模式匹配需要 fallback 的层（如 norm1 norm2）",
    )
    args = parser.parse_args()

    # 检查 ONNX 文件
    if not Path(args.onnx).exists():
        sys.exit(f"ONNX 文件不存在: {args.onnx}")

    # 生成输入数据
    if args.audio:
        if not Path(args.audio).exists():
            sys.exit(f"音频文件不存在: {args.audio}")
        print(f"从音频生成输入: {args.audio}")
        input_data = generate_input_from_audio(args.audio)
    else:
        print(f"使用随机输入: batch={args.batch}, seq_len={args.seq_len}")
        input_data = generate_dummy_input(seq_len=args.seq_len, batch=args.batch)

    print()

    # 收集 fp32 fallback 层
    fp32_layers = []
    fp32_patterns = args.fp32_pattern

    if args.fp32_layers_from:
        if not Path(args.fp32_layers_from).exists():
            sys.exit(f"JSON 文件不存在: {args.fp32_layers_from}")
        fp32_layers.extend(load_fp32_layers_from_json(args.fp32_layers_from))

    if args.fp32_layers:
        fp32_layers.extend(args.fp32_layers)

    # 去重
    fp32_layers = list(dict.fromkeys(fp32_layers))

    # 决定运行模式
    if fp32_layers or fp32_patterns:
        print(f"模式: 混合精度分析（{len(fp32_layers)} 个指定层 + {len(fp32_patterns or [])} 个模式）")
        all_pass = run_mixed_precision_analysis(
            onnx_path=args.onnx,
            input_data=input_data,
            fp32_layers=fp32_layers,
            fp32_patterns=fp32_patterns,
            threshold=args.threshold,
            top_n=args.top,
            output_file=args.output,
        )
    else:
        print("模式: 全 fp16 分析（首轮，定位所有问题层）")
        all_pass = run_full_fp16_analysis(
            onnx_path=args.onnx,
            input_data=input_data,
            threshold=args.threshold,
            top_n=args.top,
            output_file=args.output,
        )

    if all_pass:
        print("\n结论：精度合格，无需额外 fallback。")
    else:
        print("\n结论：存在精度敏感层，请根据报告调整 fallback 配置后重新分析。")
        sys.exit(1)


if __name__ == "__main__":
    main()
