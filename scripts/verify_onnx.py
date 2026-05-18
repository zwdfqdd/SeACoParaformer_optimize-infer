"""
ONNX 模型精度验证脚本（导出环境）

对比方式：
- PT 推理：FunASR AutoModel.generate()（基准）
- ONNX 推理：onnxruntime + 内联特征提取 + tokenizer（模拟线上部署）

运行环境：转换容器内（含 funasr + onnxruntime）
不依赖 src/ 目录。

用法：
    python scripts/verify_onnx.py --audio test.wav
    python scripts/verify_onnx.py --audio test.wav --onnx-dir ./models/asr/fp32 --device cuda
    python scripts/verify_onnx.py --audio test.wav --onnx-dir ./models/asr/int8 --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


# ============================================================
# 内联特征提取（使用 torchaudio，对齐官方 WavFrontend）
# ============================================================
SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
LFR_M = 7
LFR_N = 6


def _extract_features(pcm, cmvn_mean=None, cmvn_istd=None):
    """PCM → fbank → LFR → CMVN（使用 torchaudio.compliance.kaldi.fbank）"""
    import torch
    import torchaudio.compliance.kaldi as kaldi

    # 官方缩放
    waveform = torch.from_numpy(pcm).float() * (1 << 15)
    waveform = waveform.unsqueeze(0)

    # Kaldi fbank
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

    # LFR（严格对齐官方 apply_lfr）
    import torch as _torch
    mat_t = _torch.from_numpy(mat.numpy() if hasattr(mat, 'numpy') else mat).float()
    T = mat_t.shape[0]
    T_lfr = int(np.ceil(T / LFR_N))
    left_padding = mat_t[0].repeat((LFR_M - 1) // 2, 1)
    mat_t = _torch.vstack((left_padding, mat_t))
    T2 = mat_t.shape[0]
    feat_dim = mat_t.shape[-1]

    last_idx = (T2 - LFR_M) // LFR_N + 1
    num_padding = LFR_M - (T2 - last_idx * LFR_N)
    if num_padding > 0:
        num_padding = int(
            (2 * LFR_M - 2 * T2 + (T_lfr - 1 + last_idx) * LFR_N) / 2 * (T_lfr - last_idx)
        )
        if num_padding > 0:
            mat_t = _torch.vstack([mat_t] + [mat_t[-1:]] * num_padding)

    strides = (LFR_N * feat_dim, 1)
    sizes = (T_lfr, LFR_M * feat_dim)
    lfr = mat_t.as_strided(sizes, strides).clone().float()

    # CMVN
    if cmvn_mean is not None and cmvn_istd is not None:
        cmvn_mean_t = torch.from_numpy(cmvn_mean).float()
        cmvn_istd_t = torch.from_numpy(cmvn_istd).float()
        lfr = (lfr + cmvn_mean_t) * cmvn_istd_t

    return lfr.numpy().astype(np.float32)


def _load_cmvn(cmvn_path):
    """
    加载 CMVN 文件。
    支持 FunASR am.mvn 格式（Kaldi 风格）：
        <AddShift> ... [ val1 val2 ... val560 ]
        <Rescale> ... [ val1 val2 ... val560 ]
    """
    path = Path(cmvn_path)
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        return np.array(data["mean"], dtype=np.float32), np.array(data["istd"], dtype=np.float32)
    if path.suffix == ".npy":
        data = np.load(path)
        return data[0].astype(np.float32), data[1].astype(np.float32)

    # Kaldi/FunASR 文本格式 am.mvn
    # 提取方括号内的浮点数，只取 560 维的块
    with open(path, "r") as f:
        content = f.read()

    import re
    brackets = re.findall(r'\[(.*?)\]', content, re.DOTALL)
    lines_data = []
    for bracket in brackets:
        values = []
        for token in bracket.split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        if len(values) > 10:  # 过滤掉短块（如 [0]）
            lines_data.append(np.array(values, dtype=np.float32))

    if len(lines_data) >= 2:
        mean = lines_data[0]   # AddShift → shift (560维)
        istd = lines_data[1]   # Rescale → scale (560维)
        return mean, istd

    raise ValueError(f"无法解析 CMVN 文件: {cmvn_path}, 解析到 {len(lines_data)} 个有效块")


# ============================================================
# 内联 Tokenizer
# ============================================================
def _load_tokenizer(vocab_path):
    path = Path(vocab_path)
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            max_id = max(data.values())
            tokens = [""] * (max_id + 1)
            for t, i in data.items():
                tokens[i] = t
            return tokens
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                tokens.append(parts[0])
    return tokens


def _decode_tokens(token_ids, token_list):
    special = {0, 1, 2}
    parts = []
    for tid in token_ids:
        tid = int(tid)
        if tid in special or tid < 0 or tid >= len(token_list):
            continue
        t = token_list[tid]
        if t.startswith("<") and t.endswith(">"):
            continue
        parts.append(t)
    text = "".join(parts).replace("▁", " ")
    return " ".join(text.split())


# ============================================================
# 主逻辑
# ============================================================
def load_audio(audio_path):
    pcm, sr = sf.read(audio_path, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    print(f"音频: {audio_path}, 时长: {len(pcm)/sr:.2f}s, 采样率: {sr}")
    return pcm, sr


def infer_pytorch(model_id, audio_data):
    from funasr import AutoModel
    print("\n[PT 推理] (FunASR AutoModel)")
    model = AutoModel(model=model_id, model_revision="v2.0.4", device="cpu", disable_update=True)
    result = model.generate(input=audio_data, batch_size_s=300)
    text = result[0]["text"] if result else ""
    # 去空格，与 ONNX 解码结果格式对齐
    text = text.replace(" ", "")
    print(f"  结果: {text}")
    return text


def infer_onnx(onnx_dir, audio_data, sr, device="cpu"):
    """
    ONNX 推理：内联 _extract_features + onnxruntime + 内联 tokenizer。
    模拟线上部署的完整路径，不依赖 FunASR。
    """
    print(f"\n[ONNX 推理] (自实现前端 + onnxruntime, device={device})")
    onnx_dir = Path(onnx_dir)

    # 查找 onnx 模型文件
    onnx_path = onnx_dir / "model.onnx"
    if not onnx_path.exists():
        onnx_files = list(onnx_dir.glob("*.onnx"))
        onnx_path = onnx_files[0] if onnx_files else None
        if not onnx_path:
            sys.exit(f"错误：{onnx_dir} 下未找到 .onnx 文件")
    print(f"  模型: {onnx_path}")

    # 配置文件统一在 models/asr 下（onnx_dir 的父目录）
    config_dir = onnx_dir.parent

    # 加载 CMVN
    cmvn_mean, cmvn_istd = None, None
    cmvn_path = config_dir / "am.mvn"
    if cmvn_path.exists():
        cmvn_mean, cmvn_istd = _load_cmvn(str(cmvn_path))
        print(f"  CMVN: {cmvn_path}")

    # 特征提取
    features = _extract_features(audio_data, cmvn_mean, cmvn_istd)
    print(f"  特征: shape={features.shape}")

    # ORT 推理
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = False
    sess_options.enable_cpu_mem_arena = False

    if device == "cuda":
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(onnx_path), sess_options, providers=providers)
    actual_provider = sess.get_providers()[0]
    print(f"  设备: {device} (provider: {actual_provider})")
    inputs = sess.get_inputs()
    print(f"  模型输入: {[(i.name, i.shape, i.type) for i in inputs]}")

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
            feed[inp.name] = np.zeros((1, 1, embed_dim), dtype=np.float32)
        else:
            feed[inp.name] = np.zeros((1,), dtype=np.int32)

    outputs = sess.run(None, feed)
    logits = outputs[0]
    print(f"  输出: shape={logits.shape}")

    # 解码
    token_ids = np.argmax(logits[0], axis=-1)
    token_list = None
    config_dir_tokens = onnx_dir.parent
    for name in ["tokens.json", "tokens.txt"]:
        p = config_dir_tokens / name
        if p.exists():
            token_list = _load_tokenizer(str(p))
            break

    if token_list:
        text = _decode_tokens(token_ids, token_list)
    else:
        text = f"[无词表, ids前10: {token_ids[:10].tolist()}]"

    # 去空格对齐
    text = text.replace(" ", "")
    print(f"  结果: {text}")
    return text


def compute_cer(reference, hypothesis):
    ref, hyp = list(reference), list(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0
    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.int32)
    for i in range(len(ref)+1): d[i][0] = i
    for j in range(len(hyp)+1): d[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i][j] = d[i-1][j-1] if ref[i-1] == hyp[j-1] else min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    return d[len(ref)][len(hyp)] / len(ref)


def main():
    parser = argparse.ArgumentParser(description="ONNX 精度验证（导出环境）")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--model-id", default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    parser.add_argument("--onnx-dir", default="./models/asr/fp32", help="ONNX 模型目录（含 model.onnx；am.mvn 和 tokens.json 在父目录 models/asr 下）")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="ONNX 推理设备（默认 cpu）")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"错误：音频不存在: {args.audio}")
    if not Path(args.onnx_dir).exists():
        sys.exit(f"错误：目录不存在: {args.onnx_dir}")

    print("=" * 50)
    print("SeACo-Paraformer 精度验证")
    print("=" * 50)

    pcm, sr = load_audio(args.audio)
    pt_text = infer_pytorch(args.model_id, pcm)
    onnx_text = infer_onnx(args.onnx_dir, pcm, sr, device=args.device)

    print("\n" + "=" * 50)
    print("对比结果")
    print("=" * 50)
    print(f"  PT:   {pt_text}")
    print(f"  ONNX: {onnx_text}")

    if pt_text == onnx_text:
        print("  ✅ 文本完全一致 (CER=0%)")
        sys.exit(0)

    cer = compute_cer(pt_text, onnx_text)
    print(f"  CER: {cer:.4f} ({cer*100:.2f}%)")

    if cer <= 0.01:
        print("  ✅ 通过 (CER ≤ 1%)")
    elif cer <= 0.05:
        print("  ⚠️ 警告 (CER ≤ 5%)")
    else:
        print("  ❌ 失败 (CER > 5%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
