"""
检查 ONNX 模型结构，找到 encoder/predictor/decoder 的边界节点。
用于确定拆分点。

用法：
    python scripts/inspect_onnx.py --input ./models/asr/fp32/model.onnx
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("需要安装 onnx: pip install onnx")


def main():
    parser = argparse.ArgumentParser(description="检查 ONNX 模型结构")
    parser.add_argument("--input", required=True, help="ONNX 模型路径")
    parser.add_argument("--detail", action="store_true", help="输出详细节点信息")
    args = parser.parse_args()

    model = onnx.load(args.input)
    graph = model.graph

    print("=" * 60)
    print(f"模型: {args.input}")
    print(f"opset: {[op.version for op in model.opset_import]}")
    print("=" * 60)

    # 输入输出
    print(f"\n输入 ({len(graph.input)}):")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape} ({inp.type.tensor_type.elem_type})")

    print(f"\n输出 ({len(graph.output)}):")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape} ({out.type.tensor_type.elem_type})")

    # 统计各模块节点数
    print(f"\n总节点数: {len(graph.node)}")
    module_counts = defaultdict(int)
    module_ops = defaultdict(set)

    for node in graph.node:
        name = node.name if node.name else ""
        # 提取模块前缀（如 /encoder/..., /predictor/..., /decoder/...）
        parts = name.split("/")
        if len(parts) >= 2:
            module = parts[1]  # encoder, predictor, decoder, seaco_decoder
        else:
            module = "_other"
        module_counts[module] += 1
        module_ops[module].add(node.op_type)

    print("\n模块节点统计:")
    for module, count in sorted(module_counts.items(), key=lambda x: -x[1]):
        ops = sorted(module_ops[module])
        print(f"  {module}: {count} 节点")
        if args.detail:
            print(f"    算子: {ops}")

    # 找 encoder 输出边界
    print("\n" + "=" * 60)
    print("寻找 encoder 输出边界...")
    print("=" * 60)

    # 找所有 encoder 最后一层的输出 tensor
    encoder_outputs = set()
    predictor_inputs = set()
    decoder_inputs = set()

    for node in graph.node:
        name = node.name if node.name else ""
        if "/encoder/" in name:
            for out in node.output:
                encoder_outputs.add(out)
        if "/predictor/" in name:
            for inp in node.input:
                predictor_inputs.add(inp)
        if "/decoder/" in name:
            for inp in node.input:
                decoder_inputs.add(inp)

    # encoder 输出 ∩ (predictor 输入 ∪ decoder 输入) = 拆分点
    split_tensors = encoder_outputs & (predictor_inputs | decoder_inputs)
    print(f"\nEncoder → Predictor/Decoder 边界 tensor ({len(split_tensors)}):")
    for t in sorted(split_tensors)[:20]:
        print(f"  {t}")

    # 检查 NonZero 节点位置
    print("\n" + "=" * 60)
    print("NonZero 节点位置:")
    print("=" * 60)
    for node in graph.node:
        if node.op_type == "NonZero":
            print(f"  {node.name}")
            print(f"    输入: {list(node.input)}")
            print(f"    输出: {list(node.output)}")


if __name__ == "__main__":
    main()
