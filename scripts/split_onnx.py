"""
拆分 SeACo-Paraformer ONNX 模型为 encoder + decoder 两部分

拆分点：/encoder/after_norm/Add_1_output_0
- encoder 子图：speech, speech_lengths → encoder_output (hidden states)
- decoder 子图：encoder_output, speech_lengths, bias_embed → logits, token_num, us_alphas, us_cif_peak

encoder 用 TRT fp16 加速（无 NonZero 算子），decoder 用 ORT fp32（含 NonZero）。

用法：
    python scripts/split_onnx.py --input ./models/asr/fp32/model.onnx --output-dir ./models/asr/split
"""

import argparse
import sys
from pathlib import Path

try:
    import onnx
    from onnx import helper, TensorProto, shape_inference
except ImportError:
    sys.exit("需要安装 onnx: pip install onnx")

import numpy as np

# 拆分点 tensor 名称
SPLIT_TENSOR = "/encoder/after_norm/Add_1_output_0"


def split_model(input_path: str, output_dir: str):
    """拆分模型为 encoder + decoder。"""
    print(f"加载模型: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph

    print(f"总节点数: {len(graph.node)}")
    print(f"拆分点: {SPLIT_TENSOR}")
    print()

    # 分类节点：encoder vs decoder
    encoder_nodes = []
    decoder_nodes = []

    # 构建 tensor → producer 映射
    tensor_producers = {}  # tensor_name → node
    for node in graph.node:
        for out in node.output:
            tensor_producers[out] = node

    # 从拆分点反向追溯，找到所有 encoder 节点
    encoder_tensors = set()
    queue = [SPLIT_TENSOR]

    while queue:
        tensor = queue.pop()
        if tensor in encoder_tensors:
            continue
        encoder_tensors.add(tensor)
        if tensor in tensor_producers:
            node = tensor_producers[tensor]
            for inp in node.input:
                queue.append(inp)

    # 分类节点
    for node in graph.node:
        is_encoder = any(out in encoder_tensors for out in node.output)
        if is_encoder:
            encoder_nodes.append(node)
        else:
            decoder_nodes.append(node)

    print(f"Encoder 节点: {len(encoder_nodes)}")
    print(f"Decoder 节点: {len(decoder_nodes)}")

    # ============================================================
    # 构建 Encoder 子图
    # ============================================================
    print("\n构建 Encoder 子图...")

    # Encoder 输入：speech, speech_lengths
    encoder_inputs = []
    for inp in graph.input:
        if inp.name in ("speech", "speech_lengths"):
            encoder_inputs.append(inp)

    # Encoder 输出：拆分点 tensor
    # 推断 shape：(batch_size, feats_length, hidden_dim)
    # hidden_dim 通常是 512（Paraformer large）
    encoder_output = helper.make_tensor_value_info(
        SPLIT_TENSOR, TensorProto.FLOAT, ["batch_size", "encoder_length", 512]
    )

    # 收集 encoder 需要的 initializer
    encoder_tensor_names = set()
    for node in encoder_nodes:
        for inp in node.input:
            encoder_tensor_names.add(inp)
        for out in node.output:
            encoder_tensor_names.add(out)

    encoder_initializers = []
    for init in graph.initializer:
        if init.name in encoder_tensor_names:
            encoder_initializers.append(init)

    encoder_graph = helper.make_graph(
        encoder_nodes,
        "encoder",
        encoder_inputs,
        [encoder_output],
        initializer=encoder_initializers,
    )

    encoder_model = helper.make_model(
        encoder_graph,
        opset_imports=model.opset_import,
    )

    # ============================================================
    # 构建 Decoder 子图
    # ============================================================
    print("构建 Decoder 子图...")

    # Decoder 输入：encoder_output + speech_lengths + bias_embed
    decoder_inputs = [
        helper.make_tensor_value_info(
            SPLIT_TENSOR, TensorProto.FLOAT, ["batch_size", "encoder_length", 512]
        ),
    ]
    for inp in graph.input:
        if inp.name in ("speech_lengths", "bias_embed"):
            decoder_inputs.append(inp)

    # Decoder 输出：原始模型的所有输出
    decoder_outputs = list(graph.output)

    # 收集 decoder 需要的 initializer
    decoder_tensor_names = set()
    for node in decoder_nodes:
        for inp in node.input:
            decoder_tensor_names.add(inp)
        for out in node.output:
            decoder_tensor_names.add(out)

    decoder_initializers = []
    for init in graph.initializer:
        if init.name in decoder_tensor_names:
            decoder_initializers.append(init)

    decoder_graph = helper.make_graph(
        decoder_nodes,
        "decoder",
        decoder_inputs,
        decoder_outputs,
        initializer=decoder_initializers,
    )

    decoder_model = helper.make_model(
        decoder_graph,
        opset_imports=model.opset_import,
    )

    # ============================================================
    # 保存
    # ============================================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    encoder_path = output_path / "encoder.onnx"
    decoder_path = output_path / "decoder.onnx"

    print(f"\n保存 Encoder: {encoder_path}")
    onnx.save(encoder_model, str(encoder_path))
    enc_mb = encoder_path.stat().st_size / (1024 * 1024)
    print(f"  大小: {enc_mb:.1f}MB")

    print(f"保存 Decoder: {decoder_path}")
    onnx.save(decoder_model, str(decoder_path))
    dec_mb = decoder_path.stat().st_size / (1024 * 1024)
    print(f"  大小: {dec_mb:.1f}MB")

    orig_mb = Path(input_path).stat().st_size / (1024 * 1024)
    print(f"\n原始模型: {orig_mb:.1f}MB")
    print(f"拆分后: encoder {enc_mb:.1f}MB + decoder {dec_mb:.1f}MB = {enc_mb+dec_mb:.1f}MB")
    print("\n下一步:")
    print(f"  1. 转换 encoder 为 TRT: python scripts/convert_trt.py --input {encoder_path} --precision fp16 --profile encoder")
    print(f"  2. decoder 保持 ORT fp32 推理")


def main():
    parser = argparse.ArgumentParser(description="拆分 ONNX 模型为 encoder + decoder")
    parser.add_argument("--input", required=True, help="完整 ONNX 模型路径")
    parser.add_argument("--output-dir", default="./models/asr/split", help="输出目录")
    args = parser.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"错误：文件不存在: {args.input}")

    print("=" * 60)
    print("SeACo-Paraformer 模型拆分（encoder + decoder）")
    print("=" * 60)
    print()

    split_model(args.input, args.output_dir)


if __name__ == "__main__":
    main()
