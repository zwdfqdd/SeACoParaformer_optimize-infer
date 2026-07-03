"""
Timestamp 验证脚本

用 PT 模型跑一段音频，dump CIF 输出的 cif_peak 反推每个 token 的时间戳，
并打印字符 + 时间戳表，供人工听感对照识别精度。

流程：
    1. 加载音频 → 特征提取
    2. PT encoder + predictor 拿到 encoder_out + cif_peak
    3. cif_peak 找 fire 位置 → 每个 token 的 encoder 帧索引
    4. encoder 帧 60ms/帧（LFR_M=7, LFR_N=6, fbank shift 10ms）→ token 时间戳
    5. decoder 解码得到 text，与时间戳一一对应

字级/词级时间戳精度评估：
    - 中文单字一般 100-300ms 时长
    - 若 cif_peak 精度足够（帧级 = 60ms），可满足字幕/搜索场景
    - 若不够，可导出 alphas2 (upsample 4x, 15ms/帧) 更细粒度
"""
import argparse
import os
import sys
import numpy as np
import torch
import soundfile as sf

# 允许 scripts/ 直接执行 python scripts/verify_timestamp.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seaco_paraformer.load_model import load_model
from src.feature_extractor import extract_features, load_cmvn
from src.tokenizer import Tokenizer


# encoder 每帧对应的音频时长（ms）：
# LFR 堆叠 M=7 帧, stride=6 帧 → 每 encoder 帧 = 6 × 10ms(fbank shift) = 60ms
FRAME_MS = 60


def cif_peak_to_timestamps(cif_peak: np.ndarray, token_num: int) -> list[tuple[int, int]]:
    """
    从 cif_peak 反推每个 token 的 (start_ms, end_ms)。

    cif_peak: (T,) float 数组，>0 表示该 encoder 帧 fire 了一个 token
    token_num: 预期 token 数量（用于对齐/校验）

    返回：List[(start_ms, end_ms)]，长度 = 实际 fire 数量
    """
    peak_positions = np.where(cif_peak > 0)[0]
    timestamps = []
    for i, pos in enumerate(peak_positions):
        # 单 token 时间范围：本次 fire 到下次 fire 之间
        start_ms = int(pos * FRAME_MS)
        if i + 1 < len(peak_positions):
            end_ms = int(peak_positions[i + 1] * FRAME_MS)
        else:
            # 最后一个 token：延伸到 encoder 末尾
            end_ms = int((len(cif_peak)) * FRAME_MS)
        timestamps.append((start_ms, end_ms))
    return timestamps


def main():
    parser = argparse.ArgumentParser(description="Timestamp 验证脚本")
    parser.add_argument("--audio", required=True, help="音频文件路径（16kHz 单声道）")
    parser.add_argument("--model-id", default="./models/asr/pt",
                        help="PT 模型路径（默认 ./models/asr/pt）")
    parser.add_argument("--config-dir", default="./models/asr/pt",
                        help="CMVN + tokenizer 配置目录")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"Timestamp 验证（PT 模型，device={args.device}）")
    print("=" * 60)

    # 加载配置
    cmvn_mean, cmvn_istd = load_cmvn(os.path.join(args.config_dir, "am.mvn"))
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(args.config_dir, "tokens.json"))

    # 加载音频与特征
    pcm, sr = sf.read(args.audio, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    features = extract_features(pcm, sample_rate=sr,
                                cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
    audio_duration = len(pcm) / sr
    print(f"\n音频: {args.audio}")
    print(f"  时长: {audio_duration:.2f}s, 特征帧: {features.shape[0]}")

    # 加载模型
    print("\n加载模型...")
    model = load_model(model_id=args.model_id, device=args.device)
    model.eval()

    speech = torch.from_numpy(features).unsqueeze(0).float().to(args.device)
    speech_lengths = torch.tensor([features.shape[0]], dtype=torch.long).to(args.device)

    # Encoder + Predictor
    print("\n推理...")
    with torch.no_grad():
        # encoder 前向
        encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)
        # predictor（含 CIF）: 返回 (acoustic_embeds, token_num, alphas, cif_peak, token_num2)
        from seaco_paraformer.utils import make_pad_mask
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        pred_out = model.predictor(encoder_out, None, encoder_out_mask, ignore_id=-1)
        acoustic_embeds, token_num, alphas, cif_peak = pred_out[:4]

        # decoder 解码得到文本（不用 SeACo，只看主 decoder logits）
        decoder_out, _, _ = model.decoder(
            encoder_out, encoder_out_lens,
            acoustic_embeds, token_num,
            return_hidden=True, return_both=True,
        )
        token_ids = decoder_out[0].argmax(dim=-1).cpu().numpy()

    text = tokenizer.decode(token_ids)
    print(f"  识别结果: {text}")

    # cif_peak 反推时间戳
    cif_peak_np = cif_peak[0].cpu().numpy()  # (T,)
    token_num_val = int(token_num[0].item())
    print(f"\ncif_peak 分析:")
    print(f"  encoder 帧数: {len(cif_peak_np)}")
    print(f"  预测 token_num（float）: {alphas[0].sum().item():.2f}")
    print(f"  round(token_num): {token_num_val}")
    print(f"  cif_peak > 0 帧数: {int((cif_peak_np > 0).sum())}")
    print(f"  cif_peak 峰值: min={cif_peak_np[cif_peak_np > 0].min():.3f} "
          f"max={cif_peak_np.max():.3f}")

    timestamps = cif_peak_to_timestamps(cif_peak_np, token_num_val)

    # 字符 + 时间戳表
    print("\n" + "=" * 60)
    print("字级时间戳（cif_peak → encoder 帧 × 60ms）:")
    print("=" * 60)
    print(f"{'idx':<4} {'token':<8} {'char':<6} {'start_ms':<10} {'end_ms':<10} {'dur_ms':<8}")
    print("-" * 60)

    # 解码得到 token 字符：直接 argmax 后逐个转（避免 tokenizer.decode 的空格/合并规则）
    id_to_token = getattr(tokenizer, "_id2token", None)
    if id_to_token is None:
        id_to_token = getattr(tokenizer, "id2token", {})

    # 取 decoder 输出的 token（filter <sos>/<eos>/<blank>）
    token_ids_list = token_ids.tolist()
    # 与 timestamps 对齐取前 N 个（N = min(len(timestamps), len(token_ids)) 兼容长度不一致）
    n = min(len(timestamps), len(token_ids_list))
    for i in range(n):
        tid = int(token_ids_list[i])
        if tid <= 2:  # <blank>=0, <sos>=1, <eos>=2
            char = f"<{['blank','sos','eos'][tid]}>"
        else:
            char = id_to_token.get(tid, f"?{tid}") if isinstance(id_to_token, dict) else \
                   (id_to_token[tid] if tid < len(id_to_token) else f"?{tid}")
        start_ms, end_ms = timestamps[i]
        print(f"{i:<4} {tid:<8} {char:<6} {start_ms:<10} {end_ms:<10} {end_ms-start_ms:<8}")

    if len(timestamps) != len(token_ids_list):
        print(f"\n[警告] cif_peak fire 数({len(timestamps)}) != decoder 输出 token 数"
              f"({len(token_ids_list)})，对齐可能不完全")

    # 保存 JSON 便于后续对比
    import json
    out_json = {
        "audio": args.audio,
        "duration_s": round(audio_duration, 3),
        "text": text,
        "encoder_frames": len(cif_peak_np),
        "frame_ms": FRAME_MS,
        "token_num_predicted": round(alphas[0].sum().item(), 3),
        "token_num_rounded": token_num_val,
        "fire_count": len(timestamps),
        "decoder_tokens": len(token_ids_list),
        "words": [
            {
                "idx": i,
                "token_id": int(token_ids_list[i]) if i < len(token_ids_list) else None,
                "start_ms": timestamps[i][0],
                "end_ms": timestamps[i][1],
            }
            for i in range(n)
        ],
    }
    out_path = os.path.splitext(args.audio)[0] + "_timestamp.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
