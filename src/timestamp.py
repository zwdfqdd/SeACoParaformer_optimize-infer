"""
字级时间戳后处理（对齐 FunASR ts_prediction_lfr6_standard）

CIF timestamp head 输出 us_alphas（上采样 CIF 权重）+ us_cif_peak（fire 峰值），
本模块将其转换为每个 token 的 [start_ms, end_ms]。

算法（FunASR bicif_paraformer 标准）：
    - fire_place = where(us_cif_peak >= 1-1e-4) + force_time_shift
    - fire 数应 = token 数 + 1；不符则按 token 数归一化后重算
    - 相邻 fire 之间作为前一 token 的时长（不重叠）
    - 单 token 超 MAX_TOKEN_DURATION 帧 → 截断（超出部分算静音）
    - 首尾静音按 START_END_THRESHOLD 判定
    - TIME_RATE = 10 * LFR_N(6) / 1000 / upsample_times（每 fire 帧秒数）
"""
import numpy as np


def _cif_wo_hidden_numpy(alphas: np.ndarray, threshold: float) -> np.ndarray:
    """向量化 cif_wo_hidden（numpy 版，与 predictor._cif_wo_hidden_v1 一致）。"""
    cum = np.cumsum(alphas)
    floor_cum = np.floor(cum / threshold)
    prev = np.zeros_like(floor_cum)
    prev[1:] = floor_cum[:-1]
    fired = (floor_cum > prev).astype(np.float32)
    frac = cum - floor_cum * threshold
    return fired * 1.0 + (1.0 - fired) * frac


def compute_word_timestamps(
    us_alphas: np.ndarray,
    us_cif_peak: np.ndarray,
    num_tokens: int,
    upsample_times: int = 3,
    offset_ms: int = 0,
    force_time_shift: float = -1.5,
    max_token_duration_frames: int = 12,
    start_end_threshold: int = 5,
) -> list[tuple[int, int]]:
    """
    从 timestamp head 输出计算每个 token 的 [start_ms, end_ms]。

    Args:
        us_alphas: (T_up,) 上采样 CIF 权重（单条，已去 batch 维）
        us_cif_peak: (T_up,) fire 峰值
        num_tokens: 该 chunk 的有效 token 数（decoder 输出，不含特殊 token）
        upsample_times: 上采样倍数（模型 upsample_times，本模型 3）
        offset_ms: chunk 在原始音频中的起始毫秒（对齐全局时间轴）
        force_time_shift: fire 位置整体偏移（官方 -1.5）
        max_token_duration_frames: 单 token 最大 fire 帧数
        start_end_threshold: 首尾静音判定帧数阈值

    Returns:
        list of (start_ms, end_ms)，长度对齐 num_tokens（不足则截断/补齐末尾）
    """
    if num_tokens <= 0 or us_alphas is None or len(us_alphas) == 0:
        return []

    TIME_RATE = 10.0 * 6 / 1000 / upsample_times  # 每 fire 帧秒数
    alphas = us_alphas.astype(np.float64).copy()
    peaks = us_cif_peak.astype(np.float64).copy()

    fire_place = np.where(peaks >= 1.0 - 1e-4)[0] + force_time_shift
    # fire 数应 = token 数 + 1；不符则按 token 数归一化后重算
    if len(fire_place) != num_tokens + 1:
        total = alphas.sum()
        if total > 0:
            alphas = alphas / (total / (num_tokens + 1))
        peaks = _cif_wo_hidden_numpy(alphas, 1.0 - 1e-4)
        fire_place = np.where(peaks >= 1.0 - 1e-4)[0] + force_time_shift

    num_frames = peaks.shape[0]
    ts_list: list[list[float]] = []
    is_sil: list[bool] = []

    # 首静音
    if len(fire_place) > 0 and fire_place[0] > start_end_threshold:
        ts_list.append([0.0, fire_place[0] * TIME_RATE])
        is_sil.append(True)

    # 逐 token（相邻 fire 间隔 = 前一 token 时长）
    for i in range(len(fire_place) - 1):
        dur = fire_place[i + 1] - fire_place[i]
        if max_token_duration_frames < 0 or dur <= max_token_duration_frames:
            ts_list.append([fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE])
            is_sil.append(False)
        else:
            _split = fire_place[i] + max_token_duration_frames
            ts_list.append([fire_place[i] * TIME_RATE, _split * TIME_RATE])
            is_sil.append(False)
            ts_list.append([_split * TIME_RATE, fire_place[i + 1] * TIME_RATE])
            is_sil.append(True)

    # 尾静音
    if len(fire_place) > 0 and num_frames - fire_place[-1] > start_end_threshold:
        _end = (num_frames + fire_place[-1]) * 0.5
        if ts_list:
            ts_list[-1][1] = _end * TIME_RATE
        ts_list.append([_end * TIME_RATE, num_frames * TIME_RATE])
        is_sil.append(True)
    elif ts_list:
        ts_list[-1][1] = num_frames * TIME_RATE

    # 只取非静音 token 时间戳 + chunk 偏移（秒→毫秒）
    res: list[tuple[int, int]] = []
    for ts, sil in zip(ts_list, is_sil):
        if sil:
            continue
        start_ms = int(ts[0] * 1000) + offset_ms
        end_ms = int(ts[1] * 1000) + offset_ms
        res.append((start_ms, end_ms))
    return res
