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


def export_fp32_onnx(model_id: str, output_dir: Path, opset_version: int = 16):
    """使用 FunASR AutoModel.export() 导出 fp32 ONNX 模型。"""
    from funasr import AutoModel

    export_dir = output_dir / "fp32"
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] 加载模型: {model_id}")

    # 确保所有 Export 类被注册
    try:
        import funasr.models.bicif_paraformer.cif_predictor  # noqa: F401
        import funasr.models.contextual_paraformer.export_meta  # noqa: F401
    except ImportError:
        pass

    # 手动注册 CifPredictorV3Export
    from funasr.register import tables
    from funasr.models.bicif_paraformer.cif_predictor import CifPredictorV3Export
    tables.predictor_classes["CifPredictorV3Export"] = CifPredictorV3Export

    # Monkey-patch: 将带 Loop 的 cif_export 替换为向量化的 cif_v1_export
    from funasr.models.paraformer.cif_predictor import cif_v1_export, cif_wo_hidden_v1
    import funasr.models.bicif_paraformer.cif_predictor as bicif_module
    bicif_module.cif_export = cif_v1_export
    bicif_module.cif_wo_hidden_export = cif_wo_hidden_v1

    model = AutoModel(
        model=model_id,
        model_revision="v2.0.4",
        device="cpu",
        disable_update=True,
    )

    print(f"[2/2] 导出 fp32 ONNX (opset_version={opset_version})...")
    model.export(
        type="onnx",
        quantize=False,
        opset_version=opset_version,
        output_dir=str(export_dir),
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
    onnx.save(model_fp16, str(output_path))

    fp32_mb = fp32_onnx_path.stat().st_size / (1024 * 1024)
    fp16_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   {fp32_mb:.1f}MB → {fp16_mb:.1f}MB ({fp16_mb/fp32_mb*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="SeACo-Paraformer ONNX 导出")
    parser.add_argument("--model-id", default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    parser.add_argument("--output-dir", default="./models/asr")
    parser.add_argument("--opset-version", type=int, default=16)
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--op-block-list", nargs="+",
                        default=["LayerNormalization", "Softmax", "ReduceMean", "BatchNormalization",
                                 "Range", "Where", "Gather", "Loop", "SequenceInsert",
                                 "SequenceAt", "SequenceConstruct", "ConcatFromSequence", "SplitToSequence",
                                 "Slice", "Unsqueeze", "Squeeze", "Shape", "NonZero", "ConstantOfShape"])
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
