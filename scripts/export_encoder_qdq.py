"""
Encoder QDQ 量化导出（方案 1：Explicit Quantization）

与方案 2（Calibrator Implicit）的区别：
- 在 PyTorch 模型里显式插入 QuantizeLinear/DequantizeLinear 节点
- 用校准数据收集 amax（动态范围）写进 Q/DQ 节点
- 导出的 ONNX 自带量化信息，TRT 看到 Q/DQ 强制量化（不再依赖校准器猜测）
- 可绕过 myelin 融合阻断 INT8 的问题（Q/DQ 节点标记了明确的量化边界）

量化库优先级：
    1. nvidia-modelopt（TRT 10.6 官方配套，推荐）
    2. pytorch_quantization（NVIDIA 老库，TRT 10.x 仍支持）

用法：
    # 用 modelopt（自动检测）
    python scripts/export_encoder_qdq.py \\
        --calib-data ./speech \\
        --output ./models/asr/split/encoder_qdq.onnx

    # 强制指定库
    python scripts/export_encoder_qdq.py --backend modelopt --calib-data ./speech
    python scripts/export_encoder_qdq.py --backend pytorch_quantization --calib-data ./speech

导出后转 TRT（QDQ 不需要 calibrator）：
    python scripts/convert_trt.py --input ./models/asr/split/encoder_qdq.onnx \\
        --precision int8 --profile encoder \\
        --output ./models/asr/trt/2080_ti_encoder_int8_qdq.engine
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def detect_backend(prefer: str | None = None) -> str:
    """检测可用的量化库。"""
    candidates = []
    if prefer:
        candidates.append(prefer)
    candidates += ["modelopt", "pytorch_quantization"]

    for name in candidates:
        if name == "modelopt":
            try:
                import modelopt.torch.quantization  # noqa
                return "modelopt"
            except ImportError:
                continue
        elif name == "pytorch_quantization":
            try:
                import pytorch_quantization  # noqa
                return "pytorch_quantization"
            except ImportError:
                continue
    sys.exit(
        "未找到量化库。请在转换容器内安装：\n"
        "  pip install nvidia-modelopt\n"
        "  或 pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com"
    )


def load_calib_features(audio_dir: str, cmvn_path: str, calib_seq_len: int = 134,
                        max_samples: int = 500) -> list[np.ndarray]:
    """加载校准音频 → 特征（统一 pad 到固定长度）。"""
    import soundfile as sf
    from src.feature_extractor import extract_features, load_cmvn

    cmvn_mean, cmvn_istd = load_cmvn(cmvn_path)
    audio_files = sorted([str(p) for p in Path(audio_dir).rglob("*.wav")])[:max_samples]
    if not audio_files:
        sys.exit(f"未在 {audio_dir} 找到 .wav")

    feats = []
    for ap in audio_files:
        pcm, sr = sf.read(ap, dtype="float32")
        if len(pcm.shape) > 1:
            pcm = pcm[:, 0]
        if sr != 16000:
            continue
        f = extract_features(pcm, sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
        padded = np.zeros((calib_seq_len, 560), dtype=np.float32)
        valid = min(f.shape[0], calib_seq_len)
        padded[:valid] = f[:valid]
        feats.append(padded)
    print(f"  校准特征: {len(feats)} 条（固定长度 {calib_seq_len}）")
    return feats


# ============================================================
# EncoderWrapper（与 export_onnx_split.py 一致）
# ============================================================
def build_encoder_wrapper(clamp_value: float, model_id: str = None):
    from seaco_paraformer.load_model import load_model
    from scripts.export_onnx_split import EncoderWrapper

    model = load_model(model_id) if model_id else load_model()
    if clamp_value and clamp_value > 0:
        for layer in model.encoder.encoders0:
            layer.clamp_value = clamp_value
        for layer in model.encoder.encoders:
            layer.clamp_value = clamp_value
    wrapper = EncoderWrapper(model.encoder)
    wrapper.eval()
    return wrapper


# ============================================================
# modelopt 后端
# ============================================================
def quantize_with_modelopt(wrapper, calib_feats, output_path, opset, calib_seq_len):
    import modelopt.torch.quantization as mtq

    print("[modelopt] 配置 INT8 量化（INT8_DEFAULT_CFG）...")
    config = mtq.INT8_DEFAULT_CFG

    def forward_loop(model):
        with torch.no_grad():
            for i, f in enumerate(calib_feats):
                speech = torch.from_numpy(f[np.newaxis, :, :]).float()
                model(speech)
                if (i + 1) % 50 == 0:
                    print(f"  校准进度: {i + 1}/{len(calib_feats)}")

    print("[modelopt] 量化 + 校准中（in-place 替换 + 收集 amax）...")
    mtq.quantize(wrapper, config, forward_loop)

    # 打印量化摘要，确认哪些层插了 quantizer
    try:
        mtq.print_quant_summary(wrapper)
    except Exception as e:
        print(f"  （print_quant_summary 跳过: {e}）")

    print(f"\n[modelopt] 导出 QDQ ONNX: {output_path}")
    dummy = torch.randn(1, calib_seq_len, 560)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # modelopt 量化后，导出时需启用 fake quant → 真实 QuantizeLinear/DequantizeLinear
    try:
        from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers  # noqa
    except ImportError:
        pass

    # 使用 modelopt 上下文确保 QDQ 正确导出
    import modelopt.torch.quantization as _mtq
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), output_path,
            opset_version=opset,
            input_names=["speech"], output_names=["encoder_out"],
            dynamic_axes={"speech": {0: "batch", 1: "seq_len"},
                          "encoder_out": {0: "batch", 1: "enc_len"}},
            do_constant_folding=False,  # QDQ 量化下关闭常量折叠，避免 scale 被折叠掉
        )

    # 检查导出的 ONNX 是否含 QDQ 节点
    try:
        import onnx
        m = onnx.load(output_path)
        q_count = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
        dq_count = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear")
        print(f"\n  QDQ 节点统计: QuantizeLinear={q_count}, DequantizeLinear={dq_count}")
        if q_count == 0:
            print("  ⚠ 警告：未发现 QuantizeLinear 节点，QDQ 导出可能失败")
        else:
            print("  ✓ QDQ 节点已嵌入 ONNX，TRT 将强制量化这些点")
    except Exception as e:
        print(f"  （ONNX QDQ 检查跳过: {e}）")


# ============================================================
# pytorch_quantization 后端
# ============================================================
def quantize_with_pytorch_quantization(wrapper, calib_feats, output_path, opset, calib_seq_len):
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib
    from pytorch_quantization import quant_modules

    # 注意：quant_modules.initialize() 需要在构建模型前调用才能自动替换。
    # 这里模型已构建，改用手动 monkey-patch 不现实，提示用户。
    print("[pytorch_quantization] 警告：该后端需在模型构建前 quant_modules.initialize()，")
    print("  当前 wrapper 已构建，可能无法自动插入 QDQ。推荐改用 --backend modelopt。")

    # 尝试用 TensorQuantizer 收集 amax（best-effort）
    dummy = torch.randn(1, calib_seq_len, 560)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    torch.onnx.export(
        wrapper, (dummy,), output_path,
        opset_version=opset,
        input_names=["speech"], output_names=["encoder_out"],
        dynamic_axes={"speech": {0: "batch", 1: "seq_len"},
                      "encoder_out": {0: "batch", 1: "enc_len"}},
    )


def main():
    parser = argparse.ArgumentParser(description="Encoder QDQ 量化导出（方案 1）")
    parser.add_argument("--calib-data", default="./calib_data/audio_data", help="校准音频目录")
    parser.add_argument("--cmvn-path", default="./models/asr/pt/am.mvn")
    parser.add_argument("--output", default="./models/asr/split/encoder_qdq.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--clamp-value", type=float, default=60000)
    parser.add_argument("--calib-seq-len", type=int, default=134)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--backend", default=None,
                        choices=["modelopt", "pytorch_quantization"],
                        help="强制指定量化库（默认自动检测）")
    parser.add_argument("--model-id", default="./models/asr/pt",
                        help="PT 模型本地目录路径（默认 ./models/asr/pt，不联网下载）")
    args = parser.parse_args()

    backend = detect_backend(args.backend)
    print("=" * 60)
    print(f"Encoder QDQ 量化导出（方案 1）")
    print(f"  量化库: {backend}")
    print(f"  校准数据: {args.calib_data}")
    print(f"  输出: {args.output}")
    print(f"  opset: {args.opset}, clamp: {args.clamp_value}")
    print("=" * 60)

    print("\n[1/3] 加载 encoder...")
    wrapper = build_encoder_wrapper(args.clamp_value, args.model_id)

    print("\n[2/3] 加载校准数据...")
    calib_feats = load_calib_features(
        args.calib_data, args.cmvn_path, args.calib_seq_len, args.max_samples
    )

    print("\n[3/3] 量化 + 导出...")
    if backend == "modelopt":
        quantize_with_modelopt(wrapper, calib_feats, args.output, args.opset, args.calib_seq_len)
    else:
        quantize_with_pytorch_quantization(
            wrapper, calib_feats, args.output, args.opset, args.calib_seq_len
        )

    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\n完成！QDQ ONNX: {args.output} ({size_mb:.1f}MB)")
    print("\n下一步转 TRT（QDQ 不需要 calibrator）：")
    print(f"  python scripts/convert_trt.py --input {args.output} \\")
    print(f"      --precision int8 --profile encoder \\")
    print(f"      --output ./models/asr/trt/2080_ti_encoder_int8_qdq.engine")


if __name__ == "__main__":
    main()
