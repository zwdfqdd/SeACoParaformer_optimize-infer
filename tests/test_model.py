"""
模型直接推理测试脚本

跳过 FastAPI 服务，直接加载 ONNX 模型进行推理：
1. 加载音频 → PCM
2. 特征提取（torchaudio kaldi fbank + LFR + CMVN）
3. ONNX 模型推理（model.onnx）
4. Tokenizer 解码 → 文本

用于验证模型加载、特征提取、推理、解码全链路是否正常。

用法：
    python tests/test_model.py --audio test_data/audio_16000_30s.wav
    python tests/test_model.py --audio test_data/audio_16000_30s.wav --model-dir ./models/asr/int8 --device cpu
    python tests/test_model.py --audio test_data/audio_16000_30s.wav --hotwords 张三 李四
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_extractor import extract_features, load_cmvn
from src.tokenizer import Tokenizer


def load_onnx_session(model_path: str, device: str = "auto"):
    """加载 ONNX 模型。"""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = False
    sess_options.enable_cpu_mem_arena = False

    if device == "auto":
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            device = "cuda"
        else:
            device = "cpu"

    if device == "cuda":
        exec_providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
    else:
        exec_providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, sess_options, providers=exec_providers)
    return session, device


def main():
    parser = argparse.ArgumentParser(description="模型直接推理测试")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频路径")
    parser.add_argument("--model-dir", default="./models/asr/fp32", help="ONNX 模型目录（fp32 或 int8）")
    parser.add_argument("--config-dir", default="./models/asr/pt", help="配置文件目录（am.mvn, tokens.json）")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="推理设备")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"错误：音频文件不存在: {args.audio}")

    model_path = os.path.join(args.model_dir, "model.onnx")
    bias_model_path = os.path.join(args.model_dir, "model_eb.onnx")
    cmvn_path = os.path.join(args.config_dir, "am.mvn")
    vocab_path = os.path.join(args.config_dir, "tokens.json")
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(args.config_dir, "tokens.txt")

    print("=" * 60)
    print("SeACo-Paraformer 模型直接推理测试")
    print("=" * 60)
    print()

    # ====== Step 1: 加载音频 ======
    print("[1/5] 加载音频...")
    t0 = time.perf_counter()
    pcm, sr = sf.read(str(audio_path), dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    audio_duration_s = len(pcm) / sr
    t1 = time.perf_counter()
    print(f"  文件: {args.audio}")
    print(f"  采样率: {sr} Hz")
    print(f"  时长: {audio_duration_s:.2f}s")
    print(f"  样本数: {len(pcm)}")
    print(f"  耗时: {(t1-t0)*1000:.1f}ms")
    if sr != 16000:
        sys.exit(f"错误：采样率必须为 16000Hz，当前为 {sr}Hz")
    print()

    # ====== Step 2: 加载 CMVN + Tokenizer ======
    print("[2/5] 加载配置...")
    cmvn_mean, cmvn_istd = None, None
    if os.path.exists(cmvn_path):
        cmvn_mean, cmvn_istd = load_cmvn(cmvn_path)
        print(f"  CMVN: {cmvn_path} (shape={cmvn_mean.shape})")
    else:
        print(f"  CMVN: 未找到 {cmvn_path}，跳过归一化")

    tokenizer = Tokenizer()
    if os.path.exists(vocab_path):
        tokenizer.load(vocab_path)
        print(f"  Tokenizer: {vocab_path} (vocab_size={tokenizer.vocab_size})")
    else:
        sys.exit(f"错误：词表文件不存在: {vocab_path}")
    print()

    # ====== Step 3: 特征提取 ======
    print("[3/5] 特征提取...")
    t0 = time.perf_counter()
    features = extract_features(pcm, sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
    t1 = time.perf_counter()
    print(f"  输出 shape: {features.shape} (帧数×特征维度)")
    print(f"  帧数: {features.shape[0]}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  耗时: {(t1-t0)*1000:.1f}ms")
    print()

    # ====== Step 4: 加载模型 + 推理 ======
    print("[4/5] 模型推理...")
    if not os.path.exists(model_path):
        sys.exit(f"错误：模型文件不存在: {model_path}")

    # 加载主模型
    t0 = time.perf_counter()
    session, device = load_onnx_session(model_path, args.device)
    t1 = time.perf_counter()

    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    print(f"  模型: {model_path}")
    print(f"  设备: {device}")
    print(f"  输入: {input_names}")
    print(f"  输出: {output_names}")
    print(f"  加载耗时: {(t1-t0)*1000:.1f}ms")

    # 构建输入
    batch_features = features[np.newaxis, :, :]  # (1, T, 560)
    batch_lengths = np.array([features.shape[0]], dtype=np.int64)

    feed_dict = {}
    for name in input_names:
        if name == "speech":
            feed_dict[name] = batch_features.astype(np.float32)
        elif name == "speech_lengths":
            feed_dict[name] = batch_lengths
        elif "bias_embed" in name:
            # 热词 bias embedding
            if args.hotwords and os.path.exists(bias_model_path):
                bias_embed = _encode_hotwords(bias_model_path, tokenizer, args.hotwords)
                if bias_embed is not None:
                    # model_eb.onnx 输出 (H, D)，主模型要 (1, H, D)，补 batch 维
                    if bias_embed.ndim == 2:
                        bias_embed = bias_embed[np.newaxis, :, :]
                    feed_dict[name] = bias_embed.astype(np.float32)
                    print(f"  热词 bias: shape={bias_embed.shape}")
                else:
                    inp = next(i for i in session.get_inputs() if i.name == name)
                    embed_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else 512
                    feed_dict[name] = np.zeros((1, 1, embed_dim), dtype=np.float32)
            else:
                inp = next(i for i in session.get_inputs() if i.name == name)
                embed_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else 512
                feed_dict[name] = np.zeros((1, 1, embed_dim), dtype=np.float32)

    # 推理
    t0 = time.perf_counter()
    outputs = session.run(output_names, feed_dict)
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000
    print(f"  推理耗时: {infer_ms:.1f}ms")

    logits = outputs[0]  # (batch, time, vocab)
    print(f"  输出 logits shape: {logits.shape}")
    if len(outputs) > 1:
        print(f"  其他输出: {[o.shape for o in outputs[1:]]}")

    # 按 token_num 截断（CIF 输出 acoustic_embeds 长度=帧数，decoder 输出含 token_num
    # 之后的垃圾位置，必须按 token_num 截断，否则解码出大量乱码尾巴）
    token_num = None
    for j, oname in enumerate(output_names):
        if "token_num" in oname.lower():
            token_num = int(round(float(np.asarray(outputs[j]).flatten()[0])))
            break
    if token_num is not None:
        print(f"  token_num（有效 token 数）: {token_num}")
    print()

    # ====== Step 5: 解码 ======
    print("[5/5] 解码结果...")
    token_logits = logits[0]
    if token_num is not None and 0 < token_num <= token_logits.shape[0]:
        token_logits = token_logits[:token_num]
    token_ids = np.argmax(token_logits, axis=-1)  # (token_num,)
    text = tokenizer.decode(token_ids)

    print(f"  Token 序列长度: {len(token_ids)}")
    print(f"  有效 token 数: {np.sum((token_ids != 0) & (token_ids != 1) & (token_ids != 2))}")
    print()
    print("-" * 60)
    print(f"识别结果:")
    print(f"  {text}")
    print("-" * 60)
    print()

    # 性能汇总
    print("性能汇总:")
    print(f"  音频时长:  {audio_duration_s:.2f}s")
    print(f"  推理耗时:  {infer_ms:.1f}ms")
    rtf = (infer_ms / 1000) / audio_duration_s
    rtx = audio_duration_s / (infer_ms / 1000)
    print(f"  RTF:       {rtf:.4f}")
    print(f"  RTX:       {rtx:.2f}x")
    print()
    print("=" * 60)


def _encode_hotwords(bias_model_path: str, tokenizer: Tokenizer, hotwords: list[str]):
    """编码热词为 bias embeddings。"""
    import onnxruntime as ort

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        bias_session = ort.InferenceSession(
            bias_model_path, sess_options, providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        print(f"  警告：Bias encoder 加载失败: {e}")
        return None

    # 编码热词为 token IDs
    encoded = [tokenizer.encode(hw) for hw in hotwords if hw]
    encoded = [ids for ids in encoded if ids]
    if not encoded:
        return None

    # 追加 [sos]=[1] 哨兵（SeACo NO_BIAS 占位，缺失会导致热词输出乱码）
    encoded.append([1])

    max_len = max(len(ids) for ids in encoded)
    padded = np.zeros((len(encoded), max_len), dtype=np.int64)
    for i, ids in enumerate(encoded):
        padded[i, :len(ids)] = ids

    # 推理
    bias_input_names = [i.name for i in bias_session.get_inputs()]
    feed = {bias_input_names[0]: padded}
    if len(bias_input_names) >= 2:
        lengths = np.array([(row != 0).sum() for row in padded], dtype=np.int64)
        feed[bias_input_names[1]] = lengths

    try:
        outputs = bias_session.run(None, feed)
        return outputs[0]  # (num_hotwords, embed_dim) or (1, num_hotwords, embed_dim)
    except Exception as e:
        print(f"  警告：Bias encoder 推理失败: {e}")
        return None


if __name__ == "__main__":
    main()
