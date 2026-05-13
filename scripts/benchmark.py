"""
推理性能基准测试脚本

对比 PT / ONNX fp32 / ONNX fp16 三种推理方式的性能。
模型 warmup 后多次推理，统计各阶段耗时。

指标：
- 数据加载耗时
- 特征提取耗时
- 模型推理耗时
- 总耗时
- RTF (Real-Time Factor) = 推理耗时 / 音频时长
- RTX (加速比) = 音频时长 / 推理耗时

用法：
    python scripts/benchmark.py --audio test_data/audio_16000_30s.wav --runs 10
    python scripts/benchmark.py --audio test_data/audio_16000_30s.wav --runs 10 --onnx-fp32-dir ./models/asr/fp32 --onnx-fp16-dir ./models/asr/fp16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi


# ============================================================
# 特征提取（与 src/feature_extractor.py 一致）
# ============================================================
SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
LFR_M = 7
LFR_N = 6


def extract_features_torchaudio(pcm, cmvn_mean=None, cmvn_istd=None):
    """PCM → fbank → LFR → CMVN"""
    waveform = torch.from_numpy(pcm).float() * (1 << 15)
    waveform = waveform.unsqueeze(0)

    mat = kaldi.fbank(
        waveform,
        num_mel_bins=NUM_MEL_BINS,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=SAMPLE_RATE,
        snip_edges=True,
    )

    # LFR
    T = mat.shape[0]
    T_lfr = int(np.ceil(T / LFR_N))
    left_padding = mat[0].repeat((LFR_M - 1) // 2, 1)
    mat = torch.vstack((left_padding, mat))
    T2 = mat.shape[0]
    feat_dim = mat.shape[-1]

    last_idx = (T2 - LFR_M) // LFR_N + 1
    num_padding = LFR_M - (T2 - last_idx * LFR_N)
    if num_padding > 0:
        num_padding = int(
            (2 * LFR_M - 2 * T2 + (T_lfr - 1 + last_idx) * LFR_N) / 2 * (T_lfr - last_idx)
        )
        if num_padding > 0:
            mat = torch.vstack([mat] + [mat[-1:]] * num_padding)

    strides = (LFR_N * feat_dim, 1)
    sizes = (T_lfr, LFR_M * feat_dim)
    lfr = mat.as_strided(sizes, strides).clone().float()

    # CMVN
    if cmvn_mean is not None and cmvn_istd is not None:
        lfr = (lfr + torch.from_numpy(cmvn_mean).float()) * torch.from_numpy(cmvn_istd).float()

    return lfr.numpy().astype(np.float32)


def load_cmvn(cmvn_path):
    """加载 CMVN（对齐官方格式）。"""
    import re
    with open(cmvn_path, "r") as f:
        content = f.read()
    brackets = re.findall(r'\[(.*?)\]', content, re.DOTALL)
    lines_data = []
    for bracket in brackets:
        values = []
        for token in bracket.split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        if len(values) > 10:
            lines_data.append(np.array(values, dtype=np.float32))
    if len(lines_data) >= 2:
        return lines_data[0], lines_data[1]
    raise ValueError(f"无法解析 CMVN: {cmvn_path}")


# ============================================================
# 推理函数
# ============================================================
def infer_pt(model, audio_data, warmup=3, runs=10):
    """PyTorch 推理性能测试。"""
    # warmup
    for _ in range(warmup):
        model.generate(input=audio_data, batch_size_s=300)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.generate(input=audio_data, batch_size_s=300)
        times.append(time.perf_counter() - t0)

    return times


def infer_onnx(sess, features, warmup=3, runs=10):
    """ONNX Runtime 推理性能测试（仅模型推理部分）。"""
    inputs = sess.get_inputs()
    feats = features[np.newaxis, :, :].astype(np.float32)
    feats_len = np.array([features.shape[0]], dtype=np.int32)

    feed = {}
    for inp in inputs:
        if inp.name == "speech":
            feed[inp.name] = feats
        elif inp.name == "speech_lengths":
            feed[inp.name] = feats_len
        elif "bias_embed" in inp.name:
            embed_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else 512
            batch_size = feats.shape[0]
            feed[inp.name] = np.zeros((batch_size, 1, embed_dim), dtype=np.float32)
        else:
            feed[inp.name] = np.zeros((1,), dtype=np.int32)

    # 单次测试确认模型可推理
    try:
        test_out = sess.run(None, feed)
        print(f"    [验证] 单次推理成功, 输出 shape: {test_out[0].shape}")
    except Exception as e:
        print(f"    [验证] 单次推理失败: {e}")
        return []

    # warmup
    for _ in range(warmup):
        sess.run(None, feed)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append(time.perf_counter() - t0)

    return times


# ============================================================
# 主逻辑
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="推理性能基准测试")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--model-id", default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    parser.add_argument("--onnx-fp32-dir", default="./models/asr/fp32")
    parser.add_argument("--onnx-fp16-dir", default="./models/asr/fp16")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    parser.add_argument("--runs", type=int, default=10, help="测试次数")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"错误：音频不存在: {args.audio}")

    # 加载音频
    print("=" * 60)
    print("SeACo-Paraformer 推理性能基准测试")
    print("=" * 60)

    t_load = time.perf_counter()
    pcm, sr = sf.read(args.audio, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    load_time = time.perf_counter() - t_load
    audio_duration = len(pcm) / sr
    print(f"音频: {args.audio}, 时长: {audio_duration:.2f}s")
    print(f"数据加载耗时: {load_time*1000:.1f}ms")
    print(f"预热次数: {args.warmup}, 测试次数: {args.runs}")
    print()

    results = {}

    # ============================================================
    # 1. PT 推理
    # ============================================================
    print("-" * 40)
    print("[1/3] PyTorch 推理")
    print("-" * 40)
    try:
        from funasr import AutoModel
        pt_model = AutoModel(
            model=args.model_id,
            model_revision="v2.0.4",
            device="cpu",
            disable_update=True,
        )
        pt_times = infer_pt(pt_model, pcm, warmup=args.warmup, runs=args.runs)
        pt_avg = np.mean(pt_times)
        pt_std = np.std(pt_times)
        pt_rtf = pt_avg / audio_duration
        pt_rtx = audio_duration / pt_avg
        print(f"  平均耗时: {pt_avg*1000:.1f}ms ± {pt_std*1000:.1f}ms")
        print(f"  RTF: {pt_rtf:.4f}")
        print(f"  RTX: {pt_rtx:.2f}x")
        results["pt"] = {"avg_ms": pt_avg*1000, "std_ms": pt_std*1000, "rtf": pt_rtf, "rtx": pt_rtx}
        del pt_model
    except Exception as e:
        print(f"  跳过: {e}")

    # ============================================================
    # 2. ONNX fp32 推理
    # ============================================================
    print()
    print("-" * 40)
    print("[2/3] ONNX fp32 推理")
    print("-" * 40)
    fp32_dir = Path(args.onnx_fp32_dir)
    fp32_model = fp32_dir / "model.onnx"
    if fp32_model.exists():
        # 配置文件统一在 models/asr（fp32_dir 的父目录）
        config_dir = fp32_dir.parent
        cmvn_mean, cmvn_istd = None, None
        cmvn_path = config_dir / "am.mvn"
        if cmvn_path.exists():
            cmvn_mean, cmvn_istd = load_cmvn(str(cmvn_path))
            print(f"  CMVN: {cmvn_path}")

        # 特征提取计时
        feat_times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            features = extract_features_torchaudio(pcm, cmvn_mean, cmvn_istd)
            feat_times.append(time.perf_counter() - t0)
        feat_avg = np.mean(feat_times)
        print(f"  特征提取: {feat_avg*1000:.1f}ms (shape={features.shape})")

        # 模型推理（禁用内存模式，避免动态 shape 缓存冲突）
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        # 自动选择设备：有 GPU 用 GPU
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            exec_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            exec_providers = ["CPUExecutionProvider"]
        sess_fp32 = ort.InferenceSession(str(fp32_model), sess_options, providers=exec_providers)
        print(f"  设备: {sess_fp32.get_providers()[0]}")
        fp32_times = infer_onnx(sess_fp32, features, warmup=args.warmup, runs=args.runs)
        fp32_avg = np.mean(fp32_times)
        fp32_std = np.std(fp32_times)
        total_avg = feat_avg + fp32_avg
        fp32_rtf = total_avg / audio_duration
        fp32_rtx = audio_duration / total_avg
        print(f"  模型推理: {fp32_avg*1000:.1f}ms ± {fp32_std*1000:.1f}ms")
        print(f"  总耗时(特征+推理): {total_avg*1000:.1f}ms")
        print(f"  RTF: {fp32_rtf:.4f}")
        print(f"  RTX: {fp32_rtx:.2f}x")
        results["onnx_fp32"] = {
            "feat_ms": feat_avg*1000, "infer_ms": fp32_avg*1000,
            "total_ms": total_avg*1000, "std_ms": fp32_std*1000,
            "rtf": fp32_rtf, "rtx": fp32_rtx,
        }
        del sess_fp32
    else:
        print(f"  跳过: {fp32_model} 不存在")

    # ============================================================
    # 3. ONNX fp16 推理
    # ============================================================
    print()
    print("-" * 40)
    print("[3/3] ONNX fp16 推理")
    print("-" * 40)
    fp16_dir = Path(args.onnx_fp16_dir)
    fp16_model = fp16_dir / "model.onnx"
    if fp16_model.exists():
        # 复用特征（配置文件统一在 models/asr）
        if features is None:
            config_dir = fp16_dir.parent
            cmvn_mean, cmvn_istd = None, None
            cmvn_path = config_dir / "am.mvn"
            if cmvn_path.exists():
                cmvn_mean, cmvn_istd = load_cmvn(str(cmvn_path))
            features = extract_features_torchaudio(pcm, cmvn_mean, cmvn_istd)

        sess_options_fp16 = ort.SessionOptions()
        sess_options_fp16.enable_mem_pattern = False
        sess_options_fp16.enable_cpu_mem_arena = False
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            exec_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            exec_providers = ["CPUExecutionProvider"]
        sess_fp16 = ort.InferenceSession(str(fp16_model), sess_options_fp16, providers=exec_providers)
        print(f"  设备: {sess_fp16.get_providers()[0]}")
        fp16_times = infer_onnx(sess_fp16, features, warmup=args.warmup, runs=args.runs)
        fp16_avg = np.mean(fp16_times)
        fp16_std = np.std(fp16_times)
        total_avg_fp16 = feat_avg + fp16_avg
        fp16_rtf = total_avg_fp16 / audio_duration
        fp16_rtx = audio_duration / total_avg_fp16
        print(f"  模型推理: {fp16_avg*1000:.1f}ms ± {fp16_std*1000:.1f}ms")
        print(f"  总耗时(特征+推理): {total_avg_fp16*1000:.1f}ms")
        print(f"  RTF: {fp16_rtf:.4f}")
        print(f"  RTX: {fp16_rtx:.2f}x")
        results["onnx_fp16"] = {
            "feat_ms": feat_avg*1000, "infer_ms": fp16_avg*1000,
            "total_ms": total_avg_fp16*1000, "std_ms": fp16_std*1000,
            "rtf": fp16_rtf, "rtx": fp16_rtx,
        }
        del sess_fp16
    else:
        print(f"  跳过: {fp16_model} 不存在")

    # ============================================================
    # 汇总
    # ============================================================
    print()
    print("=" * 60)
    print("性能对比汇总")
    print("=" * 60)
    print(f"{'方案':<12} {'推理(ms)':<12} {'总耗时(ms)':<12} {'RTF':<10} {'RTX':<10}")
    print("-" * 56)
    if "pt" in results:
        r = results["pt"]
        print(f"{'PT':<12} {r['avg_ms']:<12.1f} {r['avg_ms']:<12.1f} {r['rtf']:<10.4f} {r['rtx']:<10.2f}")
    if "onnx_fp32" in results:
        r = results["onnx_fp32"]
        print(f"{'ONNX fp32':<12} {r['infer_ms']:<12.1f} {r['total_ms']:<12.1f} {r['rtf']:<10.4f} {r['rtx']:<10.2f}")
    if "onnx_fp16" in results:
        r = results["onnx_fp16"]
        print(f"{'ONNX fp16':<12} {r['infer_ms']:<12.1f} {r['total_ms']:<12.1f} {r['rtf']:<10.4f} {r['rtx']:<10.2f}")

    # 保存结果
    output_path = Path("benchmark_results.json")
    results["audio_duration_s"] = audio_duration
    results["runs"] = args.runs
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
