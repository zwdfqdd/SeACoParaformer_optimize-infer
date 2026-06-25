"""
SeACo-Paraformer 模型 ONNX 导出脚本

流程：
1. 使用 FunASR AutoModel.export() 导出 fp32 ONNX
2. 使用 onnxconverter-common mixed precision 转为 fp16
   - keep_io_types=True 保持输入输出为 fp32
   - op_block_list 保留精度敏感算子为 fp32
3. opset_version=16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def export_fp32_onnx(model_id: str, output_dir: Path, opset_version: int = 16):
    """使用 seaco_paraformer 加载模型并导出 fp32 ONNX（v1 整体导出）。"""
    from seaco_paraformer.load_model import load_model

    export_dir = output_dir / "fp32"
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] 加载模型: {model_id}")
    pt_model = load_model(model_id)

    print(f"[2/2] 导出 fp32 ONNX (opset_version={opset_version})...")

    import torch

    # 导出完整模型（encoder + predictor + decoder 一体）
    batch, seq_len, feat_dim = 1, 289, 560
    speech = torch.randn(batch, seq_len, feat_dim)
    speech_lengths = torch.tensor([seq_len], dtype=torch.long)

    output_path = export_dir / "model.onnx"
    torch.onnx.export(
        pt_model,
        (speech, speech_lengths),
        str(output_path),
        opset_version=opset_version,
        input_names=["speech", "speech_lengths"],
        output_names=["logits", "token_num"],
        dynamic_axes={
            "speech": {0: "batch", 1: "seq_len"},
            "speech_lengths": {0: "batch"},
            "logits": {0: "batch", 1: "token_len"},
            "token_num": {0: "batch"},
        },
    )

    onnx_files = list(export_dir.rglob("*.onnx"))
    if not onnx_files:
        print("错误：未找到导出的 ONNX 文件")
        sys.exit(1)

    print(f"   导出完成: {[f.name for f in onnx_files]}")
    return onnx_files


def convert_to_fp16(
    fp32_onnx_path: Path,
    output_path: Path,
    op_block_list: list[str],
):
    """将 fp32 ONNX 转为 fp16（使用 onnxruntime float16 工具，支持子图处理）。"""
    import onnx

    print(f"   转换 fp16: {fp32_onnx_path.name}")

    try:
        # 方案一：使用 onnxruntime.transformers.float16（支持子图）
        from onnxruntime.transformers import float16 as ort_float16

        model = onnx.load(str(fp32_onnx_path))
        model_fp16 = ort_float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            op_block_list=op_block_list,
            node_block_list=None,
        )
    except (ImportError, TypeError):
        # 方案二：使用 float16_converter 的 auto_mixed_precision
        try:
            from onnxruntime.transformers.onnx_model import OnnxModel
            from onnxruntime.transformers.float16 import convert_float_to_float16

            model = onnx.load(str(fp32_onnx_path))
            model_fp16 = convert_float_to_float16(
                model,
                keep_io_types=True,
                op_block_list=op_block_list,
            )
        except (ImportError, TypeError):
            # 方案三：onnxconverter-common + 禁用子图内所有节点
            from onnxconverter_common import float16

            model = onnx.load(str(fp32_onnx_path))

            # 收集所有子图内节点名称作为 node_block_list
            node_block_list = []
            for node in model.graph.node:
                if node.op_type in ("Loop", "If", "Scan"):
                    for attr in node.attribute:
                        if attr.g:
                            for sub_node in attr.g.node:
                                if sub_node.name:
                                    node_block_list.append(sub_node.name)

            # 收集子图内所有 op_type
            all_blocked_ops = set(op_block_list)
            for node in model.graph.node:
                if node.op_type in ("Loop", "If", "Scan"):
                    for attr in node.attribute:
                        if attr.g:
                            for sub_node in attr.g.node:
                                all_blocked_ops.add(sub_node.op_type)

            model_fp16 = float16.convert_float_to_float16(
                model,
                keep_io_types=True,
                op_block_list=list(all_blocked_ops),
                node_block_list=node_block_list if node_block_list else None,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 修复 Range 算子输入类型：Range 不支持 fp16 输入，需将其输入常量从 fp16 转回 fp32
    model_fp16 = _fix_range_inputs(model_fp16)

    onnx.save(model_fp16, str(output_path))

    fp32_mb = fp32_onnx_path.stat().st_size / (1024 * 1024)
    fp16_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   {fp32_mb:.1f}MB → {fp16_mb:.1f}MB ({fp16_mb/fp32_mb*100:.0f}%)")


def _fix_range_inputs(model):
    """
    修复 fp16 模型中 Range 算子的输入类型。
    Range 算子只接受 int32/int64/float32，但 fp16 转换可能将其输入常量转为 fp16。
    解决方案：找到所有 Range 节点，将其 fp16 输入的 initializer 转回 float32。
    """
    import onnx
    from onnx import numpy_helper, TensorProto

    graph = model.graph

    # 收集所有 Range 节点的输入名称
    range_input_names = set()
    for node in graph.node:
        if node.op_type == "Range":
            for inp in node.input:
                range_input_names.add(inp)

    if not range_input_names:
        return model

    # 修复 initializer 中的 fp16 → fp32
    for i, init in enumerate(graph.initializer):
        if init.name in range_input_names and init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype("float32")
            new_init = numpy_helper.from_array(arr, name=init.name)
            graph.initializer[i].CopyFrom(new_init)

    # 修复 graph input 中的类型声明
    for inp in graph.input:
        if inp.name in range_input_names:
            if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
                inp.type.tensor_type.elem_type = TensorProto.FLOAT

    # 修复 Constant 节点输出的类型（如果 Range 输入来自 Constant 节点）
    for node in graph.node:
        if node.op_type == "Constant" and len(node.output) > 0 and node.output[0] in range_input_names:
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype("float32")
                    new_tensor = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new_tensor)

    print(f"   修复 Range 输入: {len(range_input_names)} 个输入已转回 float32")
    return model


def main():
    parser = argparse.ArgumentParser(description="SeACo-Paraformer ONNX 导出")
    parser.add_argument("--model-id", default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    parser.add_argument("--output-dir", default="./models/asr")
    parser.add_argument("--opset-version", type=int, default=16)
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--op-block-list", nargs="+",
                        default=["Range"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print("=" * 60)
    print("SeACo-Paraformer ONNX 导出")
    print("=" * 60)
    print(f"模型: {args.model_id}")
    print(f"输出目录: {output_dir}")
    print(f"opset_version: {args.opset_version}")
    print()

    fp32_files = export_fp32_onnx(args.model_id, output_dir, args.opset_version)

    if not args.skip_fp16:
        print(f"\n转换 fp16（保留算子: {args.op_block_list}）")
        fp16_dir = output_dir / "fp16"
        for fp32_path in fp32_files:
            convert_to_fp16(fp32_path, fp16_dir / fp32_path.name, args.op_block_list)

    print("\n导出完成！")


if __name__ == "__main__":
    main()
