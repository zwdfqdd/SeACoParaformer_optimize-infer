"""
Timestamp 验证脚本

对比两种字级时间戳方案：
  --method main    ：main head alphas 反推（60ms 粒度，粗，字间停顿归前字）
  --method alphas2 ：upsample timestamp head（20ms 粒度，对齐 FunASR 官方 ts_prediction）

alphas2 方案（推荐）流程：
  1. predictor.get_upsample_timestamp → us_alphas(上采样 upsample_times=3) + us_cif_peak(fire峰值)
  2. ts_prediction_lfr6_standard 后处理：
     - fire_place = where(peak>=1-1e-4) + force_time_shift
     - 相邻 fire 之间作为前一 token 的时长（不重叠）
     - 单 token 超 MAX_TOKEN_DURATION 截断补静音
     - 首尾静音处理
  3. TIME_RATE = 10 * LFR_N(6) / 1000 / upsample_times

打印字符 + 时间戳表，保存 JSON 供对比。
"""
import argparse
import os
import sys
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seaco_paraformer.load_model import load_model
from seaco_paraformer.utils import make_pad_mask
from src.feature_extractor import extract_features, load_cmvn
from src.tokenizer import Tokenizer


FRAME_MS = 60  # main head：每 encoder 帧 60ms（LFR_N=6 × 10ms）


def ts_prediction_lfr6_standard(
    us_alphas: np.ndarray,
    us_peaks: np.ndarray,
    char_list: list[str],
    upsample_times: int = 5,
    force_time_shift: float = -1.5,
    max_token_duration_frames: int = 12,
    start_end_threshold: int = 5,
) -> list[list[int]]:
    """
    移植 FunASR ts_prediction_lfr6_standard（去掉 sil_in_str 文本拼接，只返回时间戳）。

    Args:
        us_alphas: (T_up,) 上采样 CIF 权重
        us_peaks: (T_up,) fire 峰值（>=1-1e-4 处 fire）
        char_list: token 字符列表（不含 <eos>）
        upsample_times: 上采样倍数（本模型 5）
        force_time_shift: fire 位置整体偏移（官方 -1.5）
        max_token_duration_frames: 单 token 最大帧数（超则截断补静音）
        start_end_threshold: 首尾静音判定帧数阈值

    Returns:
        list of [start_ms, end_ms]，长度 = len(char_list)（不含静音）
    """
    if not len(char_list):
        return []
    TIME_RATE = 10.0 * 6 / 1000 / upsample_times  # 每 fire 帧的秒数

    alphas = us_alphas.copy()
    peaks = us_peaks.copy()

    fire_place = np.where(peaks >= 1.0 - 1e-4)[0] + force_time_shift
    # fire 数应 = token 数 + 1；不符则按 token 数重新归一化后再检测
    if len(fire_place) != len(char_list) + 1:
        total = alphas.sum()
        if total > 0:
            alphas = alphas / (total / (len(char_list) + 1))
        # 重新 cif_wo_hidden（numpy 版）
        cum = np.cumsum(alphas)
        thr = 1.0 - 1e-4
        floor_cum = np.floor(cum / thr)
        prev = np.zeros_like(floor_cum)
        prev[1:] = floor_cum[:-1]
        fired = (floor_cum > prev).astype(np.float32)
        frac = cum - floor_cum * thr
        peaks = fired * 1.0 + (1.0 - fired) * frac
        fire_place = np.where(peaks >= 1.0 - 1e-4)[0] + force_time_shift

    num_frames = peaks.shape[0]
    timestamp_list = []
    new_char_list = []

    # 首静音
    if len(fire_place) > 0 and fire_place[0] > start_end_threshold:
        timestamp_list.append([0.0, fire_place[0] * TIME_RATE])
        new_char_list.append("<sil>")

    # 逐 token 时间戳（相邻 fire 之间 = 前一 token 时长）
    for i in range(len(fire_place) - 1):
        new_char_list.append(char_list[i] if i < len(char_list) else "?")
        dur = fire_place[i + 1] - fire_place[i]
        if max_token_duration_frames < 0 or dur <= max_token_duration_frames:
            timestamp_list.append([fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE])
        else:
            # 超长：截断到 max，剩余作静音
            _split = fire_place[i] + max_token_duration_frames
            timestamp_list.append([fire_place[i] * TIME_RATE, _split * TIME_RATE])
            timestamp_list.append([_split * TIME_RATE, fire_place[i + 1] * TIME_RATE])
            new_char_list.append("<sil>")

    # 尾静音
    if len(fire_place) > 0 and num_frames - fire_place[-1] > start_end_threshold:
        _end = (num_frames + fire_place[-1]) * 0.5
        if timestamp_list:
            timestamp_list[-1][1] = _end * TIME_RATE
        timestamp_list.append([_end * TIME_RATE, num_frames * TIME_RATE])
        new_char_list.append("<sil>")
    elif timestamp_list:
        timestamp_list[-1][1] = num_frames * TIME_RATE

    # 只返回非静音 token 的时间戳（ms）
    res = []
    for char, ts in zip(new_char_list, timestamp_list):
        if char != "<sil>":
            res.append([int(ts[0] * 1000), int(ts[1] * 1000)])
    return res


def main_head_timestamps(alphas: np.ndarray, token_ids: list[int], threshold: float) -> list[tuple[int, int]]:
    """main head alphas 反推（旧方案，60ms 粒度，对照用）。"""
    cum = np.cumsum(alphas)
    floor_cum = np.floor(cum / threshold)
    floor_diff = np.empty_like(floor_cum)
    floor_diff[0] = floor_cum[0]
    floor_diff[1:] = floor_cum[1:] - floor_cum[:-1]
    peaks = np.where(floor_diff > 0)[0]
    total = len(alphas)
    out = []
    for i, pos in enumerate(peaks):
        start = int(pos * FRAME_MS)
        end = int(peaks[i + 1] * FRAME_MS) if i + 1 < len(peaks) else int(total * FRAME_MS)
        out.append((start, end))
    return out


def main():
    parser = argparse.ArgumentParser(description="Timestamp 验证脚本")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model-id", default="./models/asr/pt")
    parser.add_argument("--config-dir", default="./models/asr/pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--method", default="alphas2", choices=["main", "alphas2"],
                        help="main=旧60ms粒度，alphas2=官方upsample时间戳")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"Timestamp 验证（method={args.method}, device={args.device}）")
    print("=" * 60)

    cmvn_mean, cmvn_istd = load_cmvn(os.path.join(args.config_dir, "am.mvn"))
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(args.config_dir, "tokens.json"))

    pcm, sr = sf.read(args.audio, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    features = extract_features(pcm, sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
    audio_duration = len(pcm) / sr
    print(f"\n音频: {args.audio}  时长: {audio_duration:.2f}s  特征帧: {features.shape[0]}")

    model = load_model(model_id=args.model_id, device=args.device)
    model.eval()
    speech = torch.from_numpy(features).unsqueeze(0).float().to(args.device)
    speech_lengths = torch.tensor([features.shape[0]], dtype=torch.long).to(args.device)

    with torch.no_grad():
        encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        pred_out = model.predictor(encoder_out, None, encoder_out_mask, ignore_id=-1)
        acoustic_embeds, token_num, alphas, cif_peak = pred_out[:4]

        decoder_out, _, _ = model.decoder(
            encoder_out, encoder_out_lens, acoustic_embeds, token_num,
            return_hidden=True, return_both=True,
        )
        token_ids = decoder_out[0].argmax(dim=-1).cpu().numpy()

    # 过滤特殊 token 得字符列表
    token_list = tokenizer._token_list
    char_list = []
    valid_token_ids = []
    for tid in token_ids.tolist():
        if tid <= 2:  # blank/sos/eos
            continue
        char = token_list[tid] if 0 <= tid < len(token_list) else f"?{tid}"
        char_list.append(char)
        valid_token_ids.append(tid)

    text = tokenizer.decode(token_ids)
    print(f"识别结果: {text}")
    print(f"token 数（含特殊）: {len(token_ids)}  有效字符数: {len(char_list)}")

    if args.method == "alphas2":
        with torch.no_grad():
            tn = torch.round(token_num).to(encoder_out.device)
            us_alphas, us_cif_peak = model.predictor.get_upsample_timestamp(
                encoder_out, encoder_out_mask, tn
            )
        us_alphas_np = us_alphas[0].cpu().numpy()
        us_peaks_np = us_cif_peak[0].cpu().numpy()
        up = int(model.predictor.upsample_times)
        print(f"\nupsample_times={up}  us_alphas 长度={len(us_alphas_np)}  "
              f"fire 数={int((us_peaks_np >= 1.0 - 1e-4).sum())}  期望={len(char_list)}+1")
        ts = ts_prediction_lfr6_standard(
            us_alphas_np, us_peaks_np, char_list, upsample_times=up
        )
        words = [{"text": c, "start_ms": t[0], "end_ms": t[1]} for c, t in zip(char_list, ts)]
    else:
        threshold = float(model.predictor.threshold)
        alphas_np = alphas[0].cpu().numpy()
        ts = main_head_timestamps(alphas_np, valid_token_ids, threshold)
        words = [{"text": c, "start_ms": t[0], "end_ms": t[1]} for c, t in zip(char_list, ts)]

    # 打印
    print("\n" + "=" * 60)
    print(f"{'idx':<4} {'char':<8} {'start_ms':<10} {'end_ms':<10} {'dur_ms':<8}")
    print("-" * 60)
    prev_end = -1
    overlap_cnt = 0
    long_cnt = 0
    for i, w in enumerate(words):
        dur = w["end_ms"] - w["start_ms"]
        flag = ""
        if w["start_ms"] < prev_end:
            flag += " OVERLAP"
            overlap_cnt += 1
        if dur > 500:
            flag += " LONG"
            long_cnt += 1
        print(f"{i:<4} {w['text']:<8} {w['start_ms']:<10} {w['end_ms']:<10} {dur:<8}{flag}")
        prev_end = w["end_ms"]

    print("-" * 60)
    print(f"字数={len(words)}  重叠={overlap_cnt}  超长(>500ms)={long_cnt}")

    import json
    out_path = os.path.splitext(args.audio)[0] + f"_ts_{args.method}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"audio": args.audio, "method": args.method, "text": text,
                   "words": words}, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
