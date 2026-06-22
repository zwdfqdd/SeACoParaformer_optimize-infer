"""
SeACo-Paraformer PT 推理验证（使用独立的 seaco_paraformer 包，不依赖 FunASR 运行时）

验证从 FunASR 训练源码抽取的 seaco_paraformer 包能否正确推理（含热词）。

用法：
    python tests/test_pt_inference_v2.py --audio test_data/audio_16000_10s.wav
    python tests/test_pt_inference_v2.py --audio test_data/audio_16000_10s.wav --hotwords 埃文 账号
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seaco_paraformer.load_model import load_model
from src.feature_extractor import extract_features, load_cmvn
from src.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="SeACo-Paraformer PT 推理验证（独立包）")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--config-dir", default="./models/asr", help="配置文件目录")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto: 有 GPU 用 GPU，否则用 CPU")
    args = parser.parse_args()

    # 自动选择设备
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，回退到 CPU")
        args.device = "cpu"

    if not Path(args.audio).exists():
        sys.exit(f"音频不存在: {args.audio}")

    print("=" * 60)
    print(f"SeACo-Paraformer PT 推理验证（独立包，device={args.device}）")
    print("=" * 60)

    # 加载配置
    cmvn_mean, cmvn_istd = load_cmvn(os.path.join(args.config_dir, "am.mvn"))
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(args.config_dir, "tokens.json"))

    # 特征提取
    pcm, sr = sf.read(args.audio, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    features = extract_features(pcm, sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
    audio_duration = len(pcm) / sr
    print(f"\n音频: {args.audio}, 时长: {audio_duration:.2f}s, 特征: {features.shape}")

    # 加载模型
    print("\n加载模型...")
    model = load_model(device=args.device)
    print(f"  encoder: {type(model.encoder).__module__}.{type(model.encoder).__name__}")
    print(f"  predictor: {type(model.predictor).__module__}.{type(model.predictor).__name__}")
    print(f"  decoder: {type(model.decoder).__module__}.{type(model.decoder).__name__}")
    print(f"  seaco_decoder layers: {len(model.seaco_decoder.decoders)}")

    speech = torch.from_numpy(features).unsqueeze(0).float().to(args.device)
    speech_lengths = torch.tensor([features.shape[0]], dtype=torch.long).to(args.device)

    # 1. 不含热词推理
    print("\n[1] 不含热词推理...")
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, token_num = model(speech, speech_lengths)
    t1 = time.perf_counter()
    token_ids = logits[0].argmax(dim=-1).cpu().numpy()
    text_no_hw = tokenizer.decode(token_ids)
    print(f"  token_num: {int(token_num.item())}")
    print(f"  结果: {text_no_hw}")
    print(f"  耗时: {(t1-t0)*1000:.0f}ms")

    # 2. 含热词推理
    if args.hotwords:
        print(f"\n[2] 含热词推理: {args.hotwords}")

        # 编码热词为 token ID 列表
        hw_list = [tokenizer.encode(hw) for hw in args.hotwords if hw]
        hw_list.append([model.sos])  # NO_BIAS 标记
        print(f"  hw_list: {hw_list}")

        t0 = time.perf_counter()
        with torch.no_grad():
            logits, token_num = model.inference(
                speech, speech_lengths,
                hw_list=hw_list,
                nfilter=50,
            )
        t1 = time.perf_counter()
        token_ids = logits[0].argmax(dim=-1).cpu().numpy()
        text_hw = tokenizer.decode(token_ids)
        print(f"  token_num: {int(token_num.item())}")
        print(f"  结果: {text_hw}")
        print(f"  耗时: {(t1-t0)*1000:.0f}ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
