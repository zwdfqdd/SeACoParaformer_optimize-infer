"""
分段 ONNX 模型端到端推理测试（ORT）

使用 encoder.onnx + cif.onnx + decoder.onnx + model_eb.onnx 串联推理。
输出每一层的输入/输出维度和中间结果，用于与 TRT 推理结果对比。

用法：
    python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_10s.wav
    python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_10s.wav --device cuda
    python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_10s.wav --hotwords 张三 李四
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnxruntime as ort


class OnnxInferencer:
    """单个 ONNX 模型推理器。"""

    def __init__(self, model_path: str, device: str = "cpu"):
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False

        if device == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.name = Path(model_path).stem

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        feed = {}
        for name in self.input_names:
            if name in inputs:
                feed[name] = inputs[name]
        outputs = self.session.run(self.output_names, feed)
        return {name: out for name, out in zip(self.output_names, outputs)}


def main():
    parser = argparse.ArgumentParser(description="分段 ONNX 模型推理测试（ORT）")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--split-dir", default="./models/asr/split", help="分段 ONNX 模型目录")
    parser.add_argument("--config-dir", default="./models/asr", help="配置文件目录")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"音频不存在: {args.audio}")

    print("=" * 60)
    print(f"分段 ONNX 模型推理测试（ORT, device={args.device}）")
    print("=" * 60)

    # ====== 加载音频 ======
    print("\n[1/7] 加载音频...")
    pcm, sr = sf.read(args.audio, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    audio_duration = len(pcm) / sr
    print(f"  文件: {args.audio}, 时长: {audio_duration:.2f}s")

    # ====== 加载配置 ======
    print("\n[2/7] 加载配置...")
    from src.feature_extractor import extract_features, load_cmvn
    from src.tokenizer import Tokenizer

    cmvn_path = os.path.join(args.config_dir, "am.mvn")
    cmvn_mean, cmvn_istd = load_cmvn(cmvn_path)

    tokenizer = Tokenizer()
    vocab_path = os.path.join(args.config_dir, "tokens.json")
    tokenizer.load(vocab_path)
    print(f"  CMVN: {cmvn_path}, Tokenizer: vocab={tokenizer.vocab_size}")

    # ====== 特征提取 ======
    print("\n[3/7] 特征提取...")
    t0 = time.perf_counter()
    features = extract_features(pcm, sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
    feat_ms = (time.perf_counter() - t0) * 1000
    print(f"  输出: {features.shape}, 耗时: {feat_ms:.1f}ms")

    # ====== 加载模型 ======
    print("\n[4/7] 加载分段 ONNX 模型...")
    split_dir = Path(args.split_dir)
    encoder = OnnxInferencer(str(split_dir / "encoder.onnx"), args.device)
    cif = OnnxInferencer(str(split_dir / "cif.onnx"), args.device)
    decoder = OnnxInferencer(str(split_dir / "decoder.onnx"), args.device)
    bias_enc = OnnxInferencer(str(split_dir / "model_eb.onnx"), args.device) if (split_dir / "model_eb.onnx").exists() else None

    print(f"  encoder 输入: {encoder.input_names}, 输出: {encoder.output_names}")
    print(f"  cif 输入: {cif.input_names}, 输出: {cif.output_names}")
    print(f"  decoder 输入: {decoder.input_names}, 输出: {decoder.output_names}")
    if bias_enc:
        print(f"  model_eb 输入: {bias_enc.input_names}, 输出: {bias_enc.output_names}")

    # ====== Bias Encoder ======
    print("\n[5/7] 热词编码...")
    if args.hotwords and bias_enc:
        hw_ids = [tokenizer.encode(hw) for hw in args.hotwords if hw]
        if hw_ids:
            max_len = max(len(ids) for ids in hw_ids)
            hotword_input = np.zeros((len(hw_ids), max_len), dtype=np.int64)
            for i, ids in enumerate(hw_ids):
                hotword_input[i, :len(ids)] = ids
            print(f"  输入: hotword {hotword_input.shape}")
            t0 = time.perf_counter()
            bias_out = bias_enc.infer({"hotword": hotword_input})
            bias_ms = (time.perf_counter() - t0) * 1000
            hw_embed = bias_out[bias_enc.output_names[0]]  # (num_hw, 1, 512)
            # reshape: (num_hw, 1, 512) → squeeze → (num_hw, 512) → unsqueeze → (1, num_hw, 512)
            hw_embed = hw_embed.squeeze(1)  # (num_hw, 512)
            bias_embed = hw_embed[np.newaxis, :, :]  # (1, num_hw, 512)
            print(f"  输出: hw_embed {hw_embed.shape} → bias_embed {bias_embed.shape}, 耗时: {bias_ms:.1f}ms")
        else:
            bias_embed = np.zeros((1, 1, 512), dtype=np.float32)
            print("  无有效热词")
    else:
        bias_embed = np.zeros((1, 1, 512), dtype=np.float32)
        print("  无热词，bias_embed = zeros(1,1,512)")

    # ====== Encoder ======
    print("\n[6/7] 串联推理...")
    speech_input = features[np.newaxis, :, :].astype(np.float32)
    speech_lengths = np.array([features.shape[0]], dtype=np.int64)
    print(f"\n  --- Encoder ---")
    print(f"  输入: speech {speech_input.shape}, speech_lengths {speech_lengths}")
    t0 = time.perf_counter()
    enc_out = encoder.infer({"speech": speech_input, "speech_lengths": speech_lengths})
    enc_ms = (time.perf_counter() - t0) * 1000
    print(f"  输出: {[(k, v.shape) for k, v in enc_out.items()]}")
    print(f"  耗时: {enc_ms:.1f}ms")

    encoder_out = enc_out["encoder_out"]

    # ====== CIF ======
    print(f"\n  --- CIF ---")
    print(f"  输入: encoder_out {encoder_out.shape}")
    t0 = time.perf_counter()
    cif_out = cif.infer({"encoder_out": encoder_out})
    cif_ms = (time.perf_counter() - t0) * 1000
    print(f"  输出: {[(k, v.shape) for k, v in cif_out.items()]}")
    print(f"  耗时: {cif_ms:.1f}ms")

    acoustic_embeds = cif_out["acoustic_embeds"]
    token_num = cif_out.get("token_num", None)

    # 截取有效 token
    if token_num is not None:
        n_tokens = int(token_num.flatten()[0])
        print(f"  token_num = {n_tokens}")
        if n_tokens > 0:
            acoustic_embeds = acoustic_embeds[:, :n_tokens, :]
            print(f"  截取 acoustic_embeds → {acoustic_embeds.shape}")

    # ====== Decoder ======
    print(f"\n  --- Decoder ---")
    batch = acoustic_embeds.shape[0]
    token_len = acoustic_embeds.shape[1]
    enc_len = encoder_out.shape[1]

    acoustic_embeds_lens = np.array([token_len], dtype=np.int64)
    encoder_out_lens = np.array([enc_len], dtype=np.int64)

    # bias_embed 已经是 (1, num_hw, 512) 或 (1, 1, 512)
    bias_embed_dec = bias_embed.astype(np.float32)

    dec_inputs = {
        "acoustic_embeds": acoustic_embeds.astype(np.float32),
        "acoustic_embeds_lens": acoustic_embeds_lens,
        "encoder_out": encoder_out.astype(np.float32),
        "encoder_out_lens": encoder_out_lens,
        "bias_embed": bias_embed_dec.astype(np.float32),
    }
    print(f"  输入: {[(k, v.shape) for k, v in dec_inputs.items()]}")

    t0 = time.perf_counter()
    dec_out = decoder.infer(dec_inputs)
    dec_ms = (time.perf_counter() - t0) * 1000
    print(f"  输出: {[(k, v.shape) for k, v in dec_out.items()]}")
    print(f"  耗时: {dec_ms:.1f}ms")

    logits = dec_out[decoder.output_names[0]]

    # ====== 解码 ======
    print(f"\n[7/7] 解码...")
    token_ids = np.argmax(logits[0], axis=-1)
    text = tokenizer.decode(token_ids)

    total_ms = feat_ms + enc_ms + cif_ms + dec_ms
    print("-" * 60)
    print(f"识别结果: {text}")
    print("-" * 60)

    print(f"\n性能汇总:")
    print(f"  音频时长:    {audio_duration:.2f}s")
    print(f"  特征提取:    {feat_ms:.1f}ms")
    print(f"  Encoder:     {enc_ms:.1f}ms")
    print(f"  CIF:         {cif_ms:.1f}ms")
    print(f"  Decoder:     {dec_ms:.1f}ms")
    print(f"  总耗时:      {total_ms:.1f}ms")
    rtf = (total_ms / 1000) / audio_duration
    rtx = audio_duration / (total_ms / 1000)
    print(f"  RTF:         {rtf:.4f}")
    print(f"  RTX:         {rtx:.1f}x")


if __name__ == "__main__":
    main()
