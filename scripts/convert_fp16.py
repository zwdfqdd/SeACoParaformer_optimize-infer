"""
ONNX fp32 → fp16 转换脚本

将 fp32 ONNX 模型转换为 fp16 混合精度模型。
- keep_io_types=True：保持输入输出为 fp32
- op_block_list：保留精度敏感算子为 fp32
- 自动修复 Range 算子输入类型

SeACo-Paraformer 专用说明：
  CIF predictor 使用 cumsum + threshold 检测 peak，fp16 精度不足会导致：
  - cumsum 累积误差放大 → token 数量错误
  - threshold 比较翻转 → peak 位置偏移
  - decoder attention softmax 溢出 → 输出乱码
  使用 --preset paraformer 自动设置适合的 op_block_list。

用法：
    # 使用 paraformer 预设（推荐，GPU 推理安全）
    python scripts/convert_fp16.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/fp16 --preset paraformer

    # 转换单个模型
    python scripts/convert_fp16.py --input ./models/asr/fp32/model.onnx --output ./models/asr/fp16/model.onnx --preset paraformer

    # 自定义 op_block_list
    python scripts/convert_fp16.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/fp16 --op-block-list Range CumSum Softmax LayerNormalization

    # 最小 block list（仅 Range，GPU 上精度不足，仅 CPU 可用）
    python scripts/convert_fp16.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/fp16 --op-block-list Range
"""

import argparse
import sys
from pathlib import Path

# ============================================================
# 预设 op_block_list（保留 fp32 的算子集合）
# ============================================================
PRESETS = {
    # 最小集合：仅 Range（GPU 上 CIF 精度不足，不推荐）
    "minimal": ["Range"],

    # SeACo-Paraformer 推荐：仅对 MatMul/Conv 用 fp16，其余全部保留 fp32
    # 通过反向思维：block 所有非计算密集算子
    "paraformer": [
        "Range",
        "CumSum",
        "Sub",
        "Greater",
        "Where",
        "Softmax",
        "LayerNormalization",
        "ReduceMean",
        "Sqrt",
        "Div",
        "Add",
        "Mul",
        "Pow",
        "Exp",
        "Log",
        "Relu",
        "Sigmoid",
        "Tanh",
        "Slice",
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Reshape",
        "Transpose",
        "Concat",
        "Split",
        "Cast",
        "Shape",
        "ConstantOfShape",
        "Expand",
        "Tile",
        "Pad",
        "Clip",
        "Equal",
        "Less",
        "Not",
        "And",
        "Or",
        "Neg",
        "Abs",
        "Ceil",
        "Floor",
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        "TopK",
        "NonZero",
        "ScatterND",
        "GatherND",
        "GatherElements",
    ],

    # 激进模式：仅 MatMul 和 Conv 用 fp16（最大压缩，权重 fp16 但计算走 fp32）
    "weights_only": "SPECIAL_WEIGHTS_ONLY",
}


def convert_to_fp16(fp32_path: Path, output_path: Path, op_block_list: list[str]):
    """将 fp32 ONNX 转为 fp16。"""
    import onnx

    print(f"转换: {fp32_path.name}")
    print(f"  输入: {fp32_path}")
    print(f"  输出: {output_path}")
    print(f"  op_block_list: {op_block_list[:5]}{'...' if len(op_block_list) > 5 else ''} ({len(op_block_list)}个)")

    model = onnx.load(str(fp32_path))

    # 尝试多种转换方案
    model_fp16 = None

    # 方案一：onnxruntime.transformers.float16（支持子图）
    try:
        from onnxruntime.transformers import float16 as ort_float16

        model_fp16 = ort_float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            op_block_list=op_block_list,
            node_block_list=None,
        )
        print("  方案: onnxruntime.transformers.float16")
    except (ImportError, TypeError, Exception) as e:
        print(f"  方案一失败: {e}")

    # 方案二：onnxruntime.transformers.float16.convert_float_to_float16
    if model_fp16 is None:
        try:
            from onnxruntime.transformers.float16 import convert_float_to_float16

            model = onnx.load(str(fp32_path))
            model_fp16 = convert_float_to_float16(
                model,
                keep_io_types=True,
                op_block_list=op_block_list,
            )
            print("  方案: onnxruntime.transformers.float16 (v2)")
        except (ImportError, TypeError, Exception) as e:
            print(f"  方案二失败: {e}")

    # 方案三：onnxconverter-common
    if model_fp16 is None:
        try:
            from onnxconverter_common import float16

            model = onnx.load(str(fp32_path))

            # 收集子图内节点，避免 Sequence 类型冲突
            node_block_list = []
            all_blocked_ops = set(op_block_list)
            for node in model.graph.node:
                if node.op_type in ("Loop", "If", "Scan"):
                    for attr in node.attribute:
                        if attr.g:
                            for sub_node in attr.g.node:
                                if sub_node.name:
                                    node_block_list.append(sub_node.name)
                                all_blocked_ops.add(sub_node.op_type)

            model_fp16 = float16.convert_float_to_float16(
                model,
                keep_io_types=True,
                op_block_list=list(all_blocked_ops),
                node_block_list=node_block_list if node_block_list else None,
            )
            print("  方案: onnxconverter-common")
        except (ImportError, TypeError, Exception) as e:
            print(f"  方案三失败: {e}")
            sys.exit(f"错误：所有转换方案均失败")

    # 修复 Range 算子输入类型
    model_fp16 = _fix_range_inputs(model_fp16)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(output_path))

    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    fp16_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  大小: {fp32_mb:.1f}MB → {fp16_mb:.1f}MB ({fp16_mb/fp32_mb*100:.0f}%)")
    print()


def _fix_range_inputs(model):
    """
    修复 Range 算子输入类型。
    Range 只接受 int32/int64/float32，fp16 转换可能误将其输入转为 fp16。
    """
    from onnx import numpy_helper, TensorProto

    graph = model.graph

    range_input_names = set()
    for node in graph.node:
        if node.op_type == "Range":
            for inp in node.input:
                range_input_names.add(inp)

    if not range_input_names:
        return model

    fixed_count = 0

    # 修复 initializer
    for i, init in enumerate(graph.initializer):
        if init.name in range_input_names and init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype("float32")
            new_init = numpy_helper.from_array(arr, name=init.name)
            graph.initializer[i].CopyFrom(new_init)
            fixed_count += 1

    # 修复 graph input 类型声明
    for inp in graph.input:
        if inp.name in range_input_names:
            if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
                inp.type.tensor_type.elem_type = TensorProto.FLOAT
                fixed_count += 1

    # 修复 Constant 节点
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) > 0 and node.output[0] in range_input_names:
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype("float32")
                    new_tensor = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new_tensor)
                    fixed_count += 1

    if fixed_count > 0:
        print(f"  修复 Range 输入: {fixed_count} 处")

    return model


def convert_weights_only(fp32_path: Path, output_path: Path):
    """
    仅将大权重张量转为 fp16，保持所有计算图为 fp32。
    
    效果：模型文件约缩小 50%，但 GPU 推理时所有计算仍为 fp32（ORT 自动 cast）。
    适用于：显存受限但需要 fp32 精度的场景。
    """
    import onnx
    from onnx import numpy_helper, TensorProto

    print(f"转换（仅权重 fp16）: {fp32_path.name}")
    print(f"  输入: {fp32_path}")
    print(f"  输出: {output_path}")

    model = onnx.load(str(fp32_path))
    graph = model.graph

    converted_count = 0
    total_saved_bytes = 0

    for i, init in enumerate(graph.initializer):
        # 只转换 fp32 且大于 1024 元素的张量（小张量不值得转）
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            if arr.size > 1024:
                original_bytes = arr.nbytes
                arr_fp16 = arr.astype("float16")
                new_init = numpy_helper.from_array(arr_fp16, name=init.name)
                graph.initializer[i].CopyFrom(new_init)
                converted_count += 1
                total_saved_bytes += original_bytes - arr_fp16.nbytes

    # 注意：不修改图的计算节点，ORT 会在运行时自动 Cast fp16 权重回 fp32
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
    fp16_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  转换权重: {converted_count} 个张量")
    print(f"  大小: {fp32_mb:.1f}MB → {fp16_mb:.1f}MB ({fp16_mb/fp32_mb*100:.0f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="ONNX fp32 → fp16 转换")
    parser.add_argument("--input", default=None, help="单个 fp32 ONNX 文件路径")
    parser.add_argument("--output", default=None, help="输出 fp16 文件路径")
    parser.add_argument("--input-dir", default=None, help="fp32 模型目录（转换目录下所有 .onnx）")
    parser.add_argument("--output-dir", default=None, help="fp16 输出目录")
    parser.add_argument("--preset", default=None, choices=list(PRESETS.keys()),
                        help="预设 op_block_list（推荐 paraformer）")
    parser.add_argument("--op-block-list", nargs="+", default=None,
                        help="自定义保留 fp32 的算子列表（与 --preset 互斥）")
    args = parser.parse_args()

    # 确定 op_block_list
    if args.preset:
        if args.preset == "weights_only":
            use_weights_only = True
            op_block_list = []
            print("使用预设: weights_only（仅权重转 fp16，计算保持 fp32）")
        else:
            use_weights_only = False
            op_block_list = PRESETS[args.preset]
            print(f"使用预设: {args.preset}")
    elif args.op_block_list:
        use_weights_only = False
        op_block_list = args.op_block_list
    else:
        # 默认使用 weights_only 预设（最安全）
        use_weights_only = True
        op_block_list = []
        print("未指定预设，默认使用 weights_only（仅权重转 fp16，GPU 推理安全）")

    print("=" * 60)
    print("ONNX fp32 → fp16 转换")
    print("=" * 60)
    print(f"op_block_list: {op_block_list}")
    print()

    if args.input:
        # 单文件转换
        fp32_path = Path(args.input)
        if not fp32_path.exists():
            sys.exit(f"错误：文件不存在: {args.input}")
        output_path = Path(args.output) if args.output else fp32_path.parent.parent / "fp16" / fp32_path.name
        if use_weights_only:
            convert_weights_only(fp32_path, output_path)
        else:
            convert_to_fp16(fp32_path, output_path, op_block_list)

    elif args.input_dir:
        # 目录批量转换
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            sys.exit(f"错误：目录不存在: {args.input_dir}")
        output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / "fp16"

        onnx_files = list(input_dir.glob("*.onnx"))
        if not onnx_files:
            sys.exit(f"错误：目录下无 .onnx 文件: {input_dir}")

        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"文件数: {len(onnx_files)}")
        print()

        for fp32_path in onnx_files:
            output_path = output_dir / fp32_path.name
            if use_weights_only:
                convert_weights_only(fp32_path, output_path)
            else:
                convert_to_fp16(fp32_path, output_path, op_block_list)

    else:
        parser.print_help()
        sys.exit("\n错误：请指定 --input 或 --input-dir")

    print("转换完成！")


if __name__ == "__main__":
    main()
