"""
分段 ONNX 模型端到端推理测试（ORT，不含热词）

使用 encoder.onnx + cif.onnx + decoder.onnx 串联推理。
用于验证分段导出的正确性，以及与 TRT 推理结果对比。

用法：
    python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_10s.wav
    python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_10s.wav --device cuda
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
        feed = {name: inputs[name] for name in self.input_names if name in inputs}
        outputs = self.session.run(self.output_names, feed)
        return {name: out for name, out in zip(self.output_names, outputs)}


def main():
    parser = argparse.ArgumentParser(description="分段 ONNX 模型推理测试（不含热词）")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--split-dir", default="./models/asr/split", help="分段 ONNX 模型目录")
    parser.add_argument("--config-dir", default="./models/asr", help="配置文件目录")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"音频不存在: {args.audio}")

    print("=" * 60)
    print(f"分段 ONNX 模型推理测试（ORT, device={args.device}）")
    print("=" * 60)

    # 加载音频
    print("\n[1/5] 加载音频...")
    pcm, sr = sf.read(args.audio, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    audio_duration = len(pcm) / sr
    print(f"  文件: {args.audio}, 时长: {audio_duration:.2f}s")

    # 加载配置
    print("\n[2/5] 加载配置...")
    from src.feature_extractor import extract_features, load_cmvn
    from src.tokenizer import Tokenizer

    cmvn_mean, cmvn_istd = load_cmvn(os.path.join(args.config_dir, "am.mvn"))
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(args.config_dir, "tokens.json"))
    print(f"  Tokenizer: vocab={tokenizer.vocab_size}")

    # 特征提取
    print("\n[3/5] 特征提取...")
    t0 = time.perf_counter()
    features = extract_features(pcm, sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
    feat_ms = (time.perf_counter() - t0) * 1000
    print(f"  shape: {features.shape}, 耗时: {feat_ms:.1f}ms")

    # 加载模型
    print("\n[4/5] 加载分段 ONNX 模型...")
    split_dir = Path(args.split_dir)
    encoder = OnnxInferencer(str(split_dir / "encoder.onnx"), args.device)
    cif = OnnxInferencer(str(split_dir / "cif.onnx"), args.device)
    decoder = OnnxInferencer(str(split_dir / "decoder.onnx"), args.device)
    print(f"  encoder: {encoder.input_names} → {encoder.output_names}")
    print(f"  cif: {cif.input_names} → {cif.output_names}")
    print(f"  decoder: {decoder.input_names} → {decoder.output_names}")

    # 推理
    print("\n[5/5] 串联推理...")
    speech_input = features[np.newaxis, :, :].astype(np.float32)
    speech_lengths = np.array([features.shape[0]], dtype=np.int64)

    t0 = time.perf_counter()

    # Encoder
    enc_out = encoder.infer({"speech": speech_input, "speech_lengths": speech_lengths})
    encoder_out = enc_out["encoder_out"]
    enc_ms = (time.perf_counter() - t0) * 1000
    print(f"  [Encoder] {speech_input.shape} → {encoder_out.shape}, {enc_ms:.1f}ms")

    # CIF
    t1 = time.perf_counter()
    cif_inputs = {"encoder_out": encoder_out}
    if "encoder_out_lens" in cif.input_names:
        cif_inputs["encoder_out_lens"] = np.array([encoder_out.shape[1]], dtype=np.int64)
    cif_out = cif.infer(cif_inputs)
    cif_ms = (time.perf_counter() - t1) * 1000

    acoustic_embeds = cif_out["acoustic_embeds"]
    token_num = int(cif_out["token_num"].flatten()[0])
    acoustic_embeds = acoustic_embeds[:, :token_num, :]
    print(f"  [CIF] → {acoustic_embeds.shape} (token_num={token_num}), {cif_ms:.1f}ms")

    # Decoder
    t2 = time.perf_counter()
    dec_out = decoder.infer({
        "acoustic_embeds": acoustic_embeds.astype(np.float32),
        "acoustic_embeds_lens": np.array([token_num], dtype=np.int64),
        "encoder_out": encoder_out.astype(np.float32),
        "encoder_out_lens": np.array([encoder_out.shape[1]], dtype=np.int64),
    })
    dec_ms = (time.perf_counter() - t2) * 1000
    logits = dec_out["logits"]
    print(f"  [Decoder] → {logits.shape}, {dec_ms:.1f}ms")

    total_ms = feat_ms + enc_ms + cif_ms + dec_ms

    # 解码
    token_ids = np.argmax(logits[0], axis=-1)
    text = tokenizer.decode(token_ids)

    print(f"\n{'='*60}")
    print(f"识别结果: {text}")
    print(f"{'='*60}")

    print(f"\n性能汇总:")
    print(f"  音频时长:  {audio_duration:.2f}s")
    print(f"  特征提取:  {feat_ms:.1f}ms")
    print(f"  Encoder:   {enc_ms:.1f}ms")
    print(f"  CIF:       {cif_ms:.1f}ms")
    print(f"  Decoder:   {dec_ms:.1f}ms")
    print(f"  总耗时:    {total_ms:.1f}ms")
    rtf = (total_ms / 1000) / audio_duration
    rtx = audio_duration / (total_ms / 1000)
    print(f"  RTF:       {rtf:.4f}")
    print(f"  RTX:       {rtx:.1f}x")


if __name__ == "__main__":
    main()
