"""
音频特征提取模块（对齐 FunASR WavFrontend 官方实现）

使用 torchaudio.compliance.kaldi.fbank 确保与训练时特征完全一致。

流程：
1. PCM * 32768（int16 缩放）
2. torchaudio.compliance.kaldi.fbank（hamming 窗，80-dim，25ms/10ms，dither=0）
3. LFR（左填充 3 帧，堆叠 7 帧跳 6 帧）→ 560 维
4. CMVN：(input + shift) * scale

依赖：torch, torchaudio, numpy
"""

import re
from pathlib import Path

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi


# ============================================================
# 参数（对齐官方 WavFrontend 默认配置）
# ============================================================
SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
NUM_MEL_BINS = 80
LFR_M = 7
LFR_N = 6
DITHER = 0.0  # 推理时关闭抖动


# ============================================================
# 公开接口
# ============================================================
def extract_features(
    pcm: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    cmvn_mean: np.ndarray | None = None,
    cmvn_istd: np.ndarray | None = None,
) -> np.ndarray:
    """
    完整特征提取：PCM → fbank → LFR → CMVN → (time, 560)
    对齐 FunASR WavFrontend 官方实现。
    """
    # Step 0: 官方缩放 PCM * 32768
    waveform = torch.from_numpy(pcm).float() * (1 << 15)
    waveform = waveform.unsqueeze(0)  # (1, samples)

    # Step 1: Kaldi fbank（与官方完全一致）
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=NUM_MEL_BINS,
        frame_length=FRAME_LENGTH_MS,
        frame_shift=FRAME_SHIFT_MS,
        dither=DITHER,
        energy_floor=0.0,
        window_type="hamming",
        sample_frequency=sample_rate,
        snip_edges=True,
    )  # (num_frames, 80)

    # 防御：极短音频（< 1 帧，snip_edges 下 <25ms）可能产生 0 帧。
    # 正常管线有 VAD min-speech(250ms) + 桶下限(2s) 双重兜底，此处仅防直接调用崩溃。
    if mat.shape[0] == 0:
        return np.zeros((0, NUM_MEL_BINS * LFR_M), dtype=np.float32)

    # Step 2: LFR
    lfr_feats = _apply_lfr(mat, LFR_M, LFR_N)  # (T_lfr, 560)

    # Step 3: CMVN
    if cmvn_mean is not None and cmvn_istd is not None:
        cmvn_mean_t = torch.from_numpy(cmvn_mean).float()
        cmvn_istd_t = torch.from_numpy(cmvn_istd).float()
        lfr_feats = (lfr_feats + cmvn_mean_t) * cmvn_istd_t

    return lfr_feats.numpy().astype(np.float32)


def _apply_lfr(inputs: torch.Tensor, lfr_m: int, lfr_n: int) -> torch.Tensor:
    """
    LFR 变换（严格对齐官方 apply_lfr 实现）。
    """
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))

    # 左填充（复制第一帧）
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))

    T = inputs.shape[0]
    feat_dim = inputs.shape[-1]

    # 计算右填充（对齐官方逻辑）
    last_idx = (T - lfr_m) // lfr_n + 1
    num_padding = lfr_m - (T - last_idx * lfr_n)
    if num_padding > 0:
        num_padding = int(
            (2 * lfr_m - 2 * T + (T_lfr - 1 + last_idx) * lfr_n) / 2 * (T_lfr - last_idx)
        )
        if num_padding > 0:
            inputs = torch.vstack([inputs] + [inputs[-1:]] * num_padding)

    # as_strided 堆叠（对齐官方）
    strides = (lfr_n * feat_dim, 1)
    sizes = (T_lfr, lfr_m * feat_dim)
    lfr_outputs = inputs.as_strided(sizes, strides)

    return lfr_outputs.clone().float()


# ============================================================
# CMVN 加载（对齐官方 load_cmvn）
# ============================================================
def load_cmvn(cmvn_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    加载 CMVN 文件（对齐官方 load_cmvn 解析逻辑）。

    返回: (shift, scale)，即 am.mvn 中 AddShift 和 Rescale 的值。
    """
    path = Path(cmvn_path)

    if path.suffix == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        return np.array(data["mean"], dtype=np.float32), np.array(data["istd"], dtype=np.float32)

    if path.suffix == ".npy":
        data = np.load(path)
        return data[0].astype(np.float32), data[1].astype(np.float32)

    # Kaldi/FunASR 文本格式（对齐官方 load_cmvn）
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    means_list = []
    vars_list = []

    for i in range(len(lines)):
        line_item = lines[i].split()
        if not line_item:
            continue
        if line_item[0] == "<AddShift>":
            next_line = lines[i + 1].split()
            if next_line[0] == "<LearnRateCoef>":
                means_list = next_line[3:-1]  # 跳过前3个token，去掉末尾 ]
        elif line_item[0] == "<Rescale>":
            next_line = lines[i + 1].split()
            if next_line[0] == "<LearnRateCoef>":
                vars_list = next_line[3:-1]

    if not means_list or not vars_list:
        # fallback：正则提取方括号内容，取 560 维的两个块
        content = "".join(lines)
        brackets = re.findall(r'\[(.*?)\]', content, re.DOTALL)
        # 过滤掉短块，只取 560 维的
        valid_brackets = []
        for b in brackets:
            vals = b.split()
            if len(vals) > 10:  # 560 维的块
                valid_brackets.append(vals)
        if len(valid_brackets) >= 2:
            means_list = valid_brackets[0]
            vars_list = valid_brackets[1]

    mean = np.array([float(x) for x in means_list], dtype=np.float32)
    istd = np.array([float(x) for x in vars_list], dtype=np.float32)

    if mean.shape[0] == 0 or istd.shape[0] == 0:
        raise ValueError(f"CMVN 解析失败: {cmvn_path}")

    return mean, istd
