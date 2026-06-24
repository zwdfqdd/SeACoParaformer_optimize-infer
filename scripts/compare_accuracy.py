"""
TRT 精度对比工具（fp32 vs fp16 vs INT8）

用同一组音频跑 4 段串联推理，按精度方案统计 CER（字符错误率）。
基准是 fp32（ORT/TRT 同等行为），其他方案与基准比对。

用法：
    # 单条音频对比
    python scripts/compare_accuracy.py --audio test_data/audio_16000_10s.wav \\
        --schemes fp32 fp16 int8

    # 批量对比（输入一个音频目录）
    python scripts/compare_accuracy.py --audio-dir ./speech \\
        --schemes fp16 int8 --baseline fp32

    # 含热词
    python scripts/compare_accuracy.py --audio test_data/audio_16000_10s.wav \\
        --schemes fp32 fp16 int8 --hotwords 埃文 账号

输出：
    每个音频每个方案的识别文本 + CER（相对 baseline）。
    最后给出汇总表。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cer(ref: str, hyp: str) -> float:
    """字符错误率（Levenshtein 距离 / ref 字符数）。"""
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m] / n


def find_engine(engine_dir: Path, module: str, precision: str) -> str | None:
    """按 {gpu}_{module}_{precision}.engine 命名严格查找。"""
    if not engine_dir.is_dir():
        return None

    def _match(filename: str, mod: str, prec: str) -> bool:
        if not filename.endswith(".engine"):
            return False
        stem = filename[:-7]
        suffix = f"_{prec}"
        if not stem.endswith(suffix):
            return False
        stem_no_prec = stem[:-len(suffix)]
        if not stem_no_prec.endswith(f"_{mod}"):
            return False
        if mod == "encoder":
            base = stem_no_prec[:-(len(mod) + 1)]
            if base.endswith("_bias") or base == "bias":
                return False
        return True

    for f in engine_dir.iterdir():
        if _match(f.name, module, precision):
            return str(f)
    return None


def build_engine_set(engine_dir: Path, scheme: str) -> dict | None:
    """
    根据精度方案构造 4 段 engine 路径。

    scheme:
      fp32   — 全 fp32
      fp16   — 全 fp16
      int8   — encoder/decoder INT8 + cif/bias fp16
      int8_encoder — 仅 encoder INT8，其余 fp16（v2 阶段 2 第一阶段验证）
    """
    if scheme == "fp32":
        return {
            "encoder": find_engine(engine_dir, "encoder", "fp32"),
            "cif": find_engine(engine_dir, "cif", "fp32"),
            "decoder": find_engine(engine_dir, "decoder", "fp32"),
            "bias_encoder": find_engine(engine_dir, "bias_encoder", "fp32"),
        }
    if scheme == "fp16":
        return {
            "encoder": find_engine(engine_dir, "encoder", "fp16"),
            "cif": find_engine(engine_dir, "cif", "fp16"),
            "decoder": find_engine(engine_dir, "decoder", "fp16"),
            "bias_encoder": find_engine(engine_dir, "bias_encoder", "fp16"),
        }
    if scheme == "int8_encoder":
        return {
            "encoder": find_engine(engine_dir, "encoder", "int8"),
            "cif": find_engine(engine_dir, "cif", "fp16"),
            "decoder": find_engine(engine_dir, "decoder", "fp16"),
            "bias_encoder": find_engine(engine_dir, "bias_encoder", "fp16"),
        }
    if scheme == "int8":
        return {
            "encoder": find_engine(engine_dir, "encoder", "int8"),
            "cif": find_engine(engine_dir, "cif", "fp16"),
            "decoder": find_engine(engine_dir, "decoder", "int8"),
            "bias_encoder": find_engine(engine_dir, "bias_encoder", "fp16"),
        }
    raise ValueError(f"未知精度方案: {scheme}")


def infer_audio(
    audio_path: str,
    engines: dict,
    cmvn_mean, cmvn_istd, tokenizer,
    hotwords: list[str] | None = None,
) -> tuple[str, float]:
    """对单条音频跑完整推理，返回 (text, infer_ms)。"""
    import time as _time

    from src.feature_extractor import extract_features
    from src.trt_engine import _TRTInferencer  # type: ignore

    pcm, sr = sf.read(audio_path, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    features = extract_features(pcm, sample_rate=sr,
                                 cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
    speech = features[np.newaxis, :, :].astype(np.float32)

    encoder = _TRTInferencer(engines["encoder"])
    cif = _TRTInferencer(engines["cif"])
    decoder = _TRTInferencer(engines["decoder"])
    bias_encoder = _TRTInferencer(engines["bias_encoder"]) if engines.get("bias_encoder") else None

    # 热词
    bias_embed = None
    if hotwords and bias_encoder is not None:
        encoded = [tokenizer.encode(hw) for hw in hotwords if hw]
        encoded.append([1])  # SeACo 哨兵
        max_len = max(len(ids) for ids in encoded)
        hotword_ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        for i, ids in enumerate(encoded):
            hotword_ids[i, :len(ids)] = ids
        bias_out = bias_encoder.infer({"hotword": hotword_ids})
        hw_embed = bias_out["hw_embed"]
        hotword_lengths = (hotword_ids != 0).sum(axis=1) - 1
        hotword_lengths[-1] = 0
        hotword_lengths = np.clip(hotword_lengths, 0, None)
        hw_embed_t = hw_embed.transpose(1, 0, 2)
        bias_list = [hw_embed_t[i, hotword_lengths[i], :] for i in range(len(encoded))]
        bias_embed = np.stack(bias_list, axis=0)[np.newaxis, :, :].astype(np.float32)

    t0 = _time.perf_counter()
    enc_inputs = {"speech": speech}
    if "speech_lengths" in encoder.input_names:
        enc_inputs["speech_lengths"] = np.array([features.shape[0]], dtype=np.int64)
    enc_out = encoder.infer(enc_inputs)
    encoder_out = enc_out["encoder_out"]

    mask = np.ones((1, 1, encoder_out.shape[1]), dtype=np.float32)
    cif_out = cif.infer({"encoder_out": encoder_out, "mask": mask})
    acoustic_embeds = cif_out["acoustic_embeds"]
    token_num = int(np.round(cif_out["token_num"].flatten()[0]))
    if token_num == 0:
        return "", 0.0
    acoustic_embeds = acoustic_embeds[:, :token_num, :]

    dec_bias = bias_embed if bias_embed is not None else np.zeros((1, 1, 512), dtype=np.float32)
    dec_inputs = {}
    if "acoustic_embeds" in decoder.input_names:
        dec_inputs["acoustic_embeds"] = acoustic_embeds.astype(np.float32)
    if "encoder_out" in decoder.input_names:
        dec_inputs["encoder_out"] = encoder_out.astype(np.float32)
    if "bias_embed" in decoder.input_names:
        dec_inputs["bias_embed"] = dec_bias.astype(np.float32)
    if "token_num" in decoder.input_names:
        dec_inputs["token_num"] = np.array([token_num], dtype=np.int64)
    if "encoder_out_lens" in decoder.input_names:
        dec_inputs["encoder_out_lens"] = np.array([encoder_out.shape[1]], dtype=np.int64)
    dec_out = decoder.infer(dec_inputs)
    logits = dec_out["logits"]
    infer_ms = (_time.perf_counter() - t0) * 1000

    token_ids = np.argmax(logits[0], axis=-1)
    text = tokenizer.decode(token_ids)
    return text, infer_ms


def main():
    parser = argparse.ArgumentParser(description="TRT 精度对比")
    parser.add_argument("--audio", default=None, help="单条音频")
    parser.add_argument("--audio-dir", default=None, help="音频目录（批量）")
    parser.add_argument("--engine-dir", default="./models/asr/trt")
    parser.add_argument("--config-dir", default="./models/asr")
    parser.add_argument("--schemes", nargs="+", default=["fp32", "fp16"],
                        choices=["fp32", "fp16", "int8_encoder", "int8"])
    parser.add_argument("--baseline", default="fp32",
                        choices=["fp32", "fp16", "int8_encoder", "int8"],
                        help="作为 CER 计算基准的方案")
    parser.add_argument("--hotwords", nargs="*", default=None)
    args = parser.parse_args()

    if not args.audio and not args.audio_dir:
        sys.exit("需要 --audio 或 --audio-dir")

    # 收集音频列表
    audios: list[str] = []
    if args.audio:
        audios.append(args.audio)
    if args.audio_dir:
        audios.extend(sorted([str(p) for p in Path(args.audio_dir).rglob("*.wav")]))
    print(f"音频数：{len(audios)}")

    # 加载配置
    from src.feature_extractor import load_cmvn
    from src.tokenizer import Tokenizer
    cmvn_mean, cmvn_istd = load_cmvn(os.path.join(args.config_dir, "am.mvn"))
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(args.config_dir, "tokens.json"))

    # 检查 engine 完整性
    engine_dir = Path(args.engine_dir)
    scheme_engines = {}
    for s in args.schemes:
        eng = build_engine_set(engine_dir, s)
        if eng is None or not eng.get("encoder") or not eng.get("cif") or not eng.get("decoder"):
            print(f"跳过 {s}（engine 不齐全）: {eng}")
            continue
        scheme_engines[s] = eng

    if args.baseline not in scheme_engines:
        sys.exit(f"baseline {args.baseline} 的 engine 不齐全，无法对比")

    # 推理
    results: dict[str, dict[str, tuple[str, float]]] = {s: {} for s in scheme_engines}
    for audio in audios:
        print(f"\n音频: {audio}")
        for s, eng in scheme_engines.items():
            text, ms = infer_audio(audio, eng, cmvn_mean, cmvn_istd, tokenizer,
                                    hotwords=args.hotwords)
            results[s][audio] = (text, ms)
            print(f"  [{s:14s}] {ms:6.1f}ms  {text}")

    # 汇总 CER
    print("\n" + "=" * 80)
    print(f"汇总（baseline = {args.baseline}）")
    print("=" * 80)
    print(f"{'scheme':<14} {'avg_ms':>10} {'avg_cer':>10}")
    print("-" * 80)
    for s in args.schemes:
        if s not in results:
            continue
        cers = []
        mss = []
        for audio in audios:
            ref, _ = results[args.baseline].get(audio, ("", 0))
            hyp, ms = results[s].get(audio, ("", 0))
            cers.append(cer(ref, hyp))
            mss.append(ms)
        avg_cer = sum(cers) / len(cers) if cers else 0
        avg_ms = sum(mss) / len(mss) if mss else 0
        print(f"{s:<14} {avg_ms:>10.1f} {avg_cer*100:>9.2f}%")


if __name__ == "__main__":
    main()
