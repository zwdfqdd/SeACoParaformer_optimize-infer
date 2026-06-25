"""
CIF Predictor QDQ 量化导出（方案 1：Explicit Quantization）

CIF 的输入是 encoder 输出 (B,T,512) + mask (B,1,T)，校准数据需先用 fp16 encoder
engine 对校准音频跑出 encoder_out。

⚠ 数值敏感说明：
    CIF 的核心 cif_v1_export 含 cumsum/sigmoid 累加，对量化误差敏感（见 plan.md）。
    本脚本默认只量化 CIF 的卷积/线性权重（cif_conv1d / cif_output），
    cumsum 等累加路径不在 PyTorch 可量化模块内，天然保持 fp32/fp16。
    若量化后精度不达标，可用 --exclude-patterns 进一步排除 cif_conv1d/cif_output。

流程：
    1. 加载 PyTorch CIFWrapper（独立包）
    2. 用 fp16 encoder engine 对校准音频跑出 encoder_out
    3. modelopt INT8 量化 + 校准（收集 amax）
    4. 导出含 QDQ 节点的 ONNX

用法：
    python scripts/export_cif_qdq.py \\
        --encoder-engine ./models/asr/trt/2080_ti_encoder_fp16.engine \\
        --output ./models/asr/split/cif_qdq.onnx

导出后转 TRT（QDQ 不需要 calibrator）：
    python scripts/convert_trt.py --input ./models/asr/split/cif_qdq.onnx \\
        --precision int8 --profile cif \\
        --output ./models/asr/trt/2080_ti_cif_int8_qdq.engine
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 复用 decoder QDQ 脚本里的自包含单 engine 推理器
from scripts.export_decoder_qdq import _SimpleTRT, detect_backend


def build_cif_wrapper(model_id: str = None):
    """加载 CIFWrapper。"""
    from seaco_paraformer.load_model import load_model
    from scripts.export_onnx_split import CIFWrapper

    model = load_model(model_id) if model_id else load_model()
    wrapper = CIFWrapper(model.predictor)
    wrapper.eval()
    return wrapper


def collect_cif_inputs(
    audio_dir: str, cmvn_path: str, encoder_engine: str,
    calib_enc_len: int = 134, max_samples: int = 500,
) -> list[dict]:
    """用 fp16 encoder 跑出 cif 输入（encoder_out + mask，固定 shape）。"""
    import soundfile as sf
    from src.feature_extractor import extract_features, load_cmvn

    cmvn_mean, cmvn_istd = load_cmvn(cmvn_path)
    encoder = _SimpleTRT(encoder_engine)

    audio_files = sorted([str(p) for p in Path(audio_dir).rglob("*.wav")])[:max_samples]
    if not audio_files:
        sys.exit(f"未在 {audio_dir} 找到 .wav")

    samples = []
    for ap in audio_files:
        pcm, sr = sf.read(ap, dtype="float32")
        if len(pcm.shape) > 1:
            pcm = pcm[:, 0]
        if sr != 16000:
            continue

        features = extract_features(pcm, sample_rate=sr,
                                     cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
        enc_len = calib_enc_len
        padded = np.zeros((1, enc_len, 560), dtype=np.float32)
        valid = min(features.shape[0], enc_len)
        padded[0, :valid, :] = features[:valid]

        enc_inputs = {"speech": padded}
        if "speech_lengths" in encoder.input_names:
            enc_inputs["speech_lengths"] = np.array([enc_len], dtype=np.int64)
        enc_out = encoder.infer(enc_inputs)
        encoder_out = enc_out["encoder_out"]  # (1, enc_len, 512)

        # mask：有效帧为 1
        mask = np.zeros((1, 1, enc_len), dtype=np.float32)
        mask[0, 0, :valid] = 1.0

        samples.append({
            "encoder_out": torch.from_numpy(encoder_out).float(),
            "mask": torch.from_numpy(mask).float(),
        })

    print(f"  校准样本: {len(samples)} 条（enc_len={calib_enc_len}）")
    return samples


def quantize_with_modelopt(wrapper, samples, output_path, opset, calib_enc_len,
                           exclude_patterns=None):
    import modelopt.torch.quantization as mtq

    print("[modelopt] 配置 INT8 量化（INT8_DEFAULT_CFG）...")
    config = mtq.INT8_DEFAULT_CFG

    if exclude_patterns:
        import copy
        config = copy.deepcopy(config)
        quant_cfg = config.setdefault("quant_cfg", {})
        for pat in exclude_patterns:
            quant_cfg[f"*{pat}*"] = {"enable": False}
        print(f"  排除量化的模块模式: {exclude_patterns}")

    def forward_loop(model):
        with torch.no_grad():
            for i, s in enumerate(samples):
                model(s["encoder_out"], s["mask"])
                if (i + 1) % 50 == 0:
                    print(f"  校准进度: {i + 1}/{len(samples)}")

    print("[modelopt] 量化 + 校准中...")
    mtq.quantize(wrapper, config, forward_loop)

    try:
        mtq.print_quant_summary(wrapper)
    except Exception as e:
        print(f"  （print_quant_summary 跳过: {e}）")

    print(f"\n[modelopt] 导出 QDQ ONNX: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    dummy = (
        torch.randn(1, calib_enc_len, 512),       # encoder_out
        torch.ones(1, 1, calib_enc_len),          # mask
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, output_path,
            opset_version=opset,
            input_names=["encoder_out", "mask"],
            output_names=["acoustic_embeds", "token_num", "alphas", "cif_peak"],
            dynamic_axes={
                "encoder_out": {0: "batch", 1: "enc_len"},
                "mask": {0: "batch", 2: "enc_len"},
                "acoustic_embeds": {0: "batch", 1: "token_len"},
                "token_num": {0: "batch"},
            },
            do_constant_folding=False,
        )

    try:
        import onnx
        m = onnx.load(output_path)
        q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
        dq = sum(1 for n in m.graph.node if n.op_type == "DequantizeLinear")
        print(f"\n  QDQ 节点统计: QuantizeLinear={q}, DequantizeLinear={dq}")
        if q == 0:
            print("  ⚠ 未发现 QuantizeLinear，QDQ 导出可能失败")
        else:
            print("  ✓ QDQ 节点已嵌入 ONNX")
    except Exception as e:
        print(f"  （ONNX QDQ 检查跳过: {e}）")


def main():
    parser = argparse.ArgumentParser(description="CIF QDQ 量化导出（方案 1）")
    parser.add_argument("--calib-data", default="./calib_data/audio_data")
    parser.add_argument("--cmvn-path", default="./models/asr/am.mvn")
    parser.add_argument("--encoder-engine", required=True, help="上游 encoder fp16 engine")
    parser.add_argument("--output", default="./models/asr/split/cif_qdq.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--calib-enc-len", type=int, default=134)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--backend", default=None,
                        choices=["modelopt", "pytorch_quantization"])
    parser.add_argument("--exclude-patterns", nargs="*", default=[],
                        help="排除量化的模块名模式（保持 fp16）。"
                             "CIF cumsum 路径天然不量化；若精度不达标可加 cif_conv1d/cif_output。")
    parser.add_argument("--model-id", default=None,
                        help="PT 模型 ID 或本地目录路径（默认 ModelScope 在线）")
    args = parser.parse_args()

    backend = detect_backend(args.backend)
    print("=" * 60)
    print("CIF QDQ 量化导出（方案 1）")
    print(f"  量化库: {backend}")
    print(f"  校准数据: {args.calib_data}")
    print(f"  上游 encoder: {args.encoder_engine}")
    print(f"  输出: {args.output}")
    print("=" * 60)

    print("\n[1/3] 加载 cif...")
    wrapper = build_cif_wrapper(args.model_id)

    print("\n[2/3] 用 fp16 encoder 生成 cif 校准输入...")
    samples = collect_cif_inputs(
        args.calib_data, args.cmvn_path, args.encoder_engine,
        args.calib_enc_len, args.max_samples,
    )

    print("\n[3/3] 量化 + 导出...")
    if backend == "modelopt":
        quantize_with_modelopt(
            wrapper, samples, args.output, args.opset, args.calib_enc_len,
            exclude_patterns=args.exclude_patterns,
        )
    else:
        sys.exit("cif QDQ 仅支持 modelopt 后端")

    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\n完成！QDQ ONNX: {args.output} ({size_mb:.1f}MB)")
    print("\n下一步转 TRT：")
    print(f"  python scripts/convert_trt.py --input {args.output} \\")
    print(f"      --precision int8 --profile cif \\")
    print(f"      --output ./models/asr/trt/2080_ti_cif_int8_qdq.engine")


if __name__ == "__main__":
    main()
