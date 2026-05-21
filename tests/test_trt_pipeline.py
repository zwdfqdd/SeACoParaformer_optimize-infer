"""
TRT 分段模型端到端推理测试（v2，不含热词）

串联 3 个 TRT engine 完成 ASR 推理：
1. encoder (TRT) : speech_features → encoder_out
2. cif     (TRT) : encoder_out → acoustic_embeds, token_num
3. decoder (TRT) : acoustic_embeds + encoder_out → logits
4. tokenizer     : logits → text

用法：
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav --precision fp16
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import tensorrt as trt
except ImportError:
    sys.exit("需要安装 tensorrt")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInferencer:
    """单个 TRT engine 推理器。"""

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """执行推理。"""
        d_inputs = {}
        for name in self.input_names:
            data = inputs[name]
            self.context.set_input_shape(name, data.shape)
            t = torch.from_numpy(data).cuda().contiguous()
            d_inputs[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

        # 分配输出（动态维度用 300 预分配）
        d_outputs = {}
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            for i, s in enumerate(shape):
                if s <= 0:
                    if i == 0:
                        shape[i] = list(inputs.values())[0].shape[0]
                    else:
                        shape[i] = 300
            t = torch.zeros(shape, dtype=torch.float32, device="cuda")
            d_outputs[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

        # 执行
        stream = torch.cuda.Stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        # 获取实际输出 shape
        results = {}
        for name, t in d_outputs.items():
            actual_shape = tuple(self.context.get_tensor_shape(name))
            if all(s > 0 for s in actual_shape):
                slices = tuple(slice(0, s) for s in actual_shape)
                results[name] = t[slices].cpu().numpy()
            else:
                results[name] = t.cpu().numpy()
        return results


def main():
    parser = argparse.ArgumentParser(description="TRT 分段模型推理测试（不含热词）")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--engine-dir", default="./models/asr/trt", help="TRT engine 目录")
    parser.add_argument("--config-dir", default="./models/asr", help="配置文件目录")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"], help="engine 精度")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"音频不存在: {args.audio}")

    print("=" * 60)
    print(f"TRT 分段模型推理测试（precision={args.precision}）")
    print("=" * 60)

    # ====== 加载音频 ======
    print("\n[1/5] 加载音频...")
    pcm, sr = sf.read(args.audio, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    audio_duration = len(pcm) / sr
    print(f"  文件: {args.audio}, 时长: {audio_duration:.2f}s")

    # ====== 加载配置 ======
    print("\n[2/5] 加载配置...")
    from src.feature_extractor import extract_features, load_cmvn
    from src.tokenizer import Tokenizer

    cmvn_path = os.path.join(args.config_dir, "am.mvn")
    cmvn_mean, cmvn_istd = load_cmvn(cmvn_path)

    tokenizer = Tokenizer()
    vocab_path = os.path.join(args.config_dir, "tokens.json")
    tokenizer.load(vocab_path)
    print(f"  CMVN: {cmvn_path}, Tokenizer: vocab={tokenizer.vocab_size}")

    # ====== 特征提取 ======
    print("\n[3/5] 特征提取...")
    t0 = time.perf_counter()
    features = extract_features(pcm, sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
    feat_ms = (time.perf_counter() - t0) * 1000
    print(f"  shape: {features.shape}, 耗时: {feat_ms:.1f}ms")

    # ====== 加载 TRT engines ======
    print("\n[4/5] 加载 TRT engines...")
    engine_dir = Path(args.engine_dir)

    def find_engine(keyword, precision_override=None):
        p = precision_override or args.precision
        for f in engine_dir.glob("*.engine"):
            if keyword in f.name and p in f.name:
                return str(f)
        for f in engine_dir.glob("*.engine"):
            if keyword in f.name:
                return str(f)
        return None

    # encoder 强制 fp32（fp16 精度偏差会导致 CIF cumsum 崩溃）
    # CIF 和 decoder 跟随 --precision 参数
    encoder_path = find_engine("encoder", "fp32")
    cif_path = find_engine("cif")
    decoder_path = find_engine("decoder")

    if not all([encoder_path, cif_path, decoder_path]):
        sys.exit(f"未找到 engine 文件: encoder={encoder_path}, cif={cif_path}, decoder={decoder_path}")

    encoder = TRTInferencer(encoder_path)
    cif = TRTInferencer(cif_path)
    decoder = TRTInferencer(decoder_path)

    print(f"  encoder (TRT fp32): {encoder_path}")
    print(f"  cif (TRT {args.precision}): {cif_path}")
    print(f"  decoder (TRT {args.precision}): {decoder_path}")

    # ====== 推理 ======
    print("\n[5/5] 推理...")
    speech_input = features[np.newaxis, :, :].astype(np.float32)
    speech_lengths = np.array([features.shape[0]], dtype=np.int64)

    t0 = time.perf_counter()

    # Encoder
    enc_inputs = {"speech": speech_input}
    if "speech_lengths" in encoder.input_names:
        enc_inputs["speech_lengths"] = speech_lengths
    enc_out = encoder.infer(enc_inputs)
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

    acoustic_embeds_raw = cif_out["acoustic_embeds"]
    token_num = int(cif_out["token_num"].flatten()[0]) if "token_num" in cif_out else acoustic_embeds_raw.shape[1]
    acoustic_embeds = acoustic_embeds_raw[:, :token_num, :] if token_num > 0 else acoustic_embeds_raw
    print(f"  [CIF] {encoder_out.shape} → {acoustic_embeds.shape} (token_num={token_num}), {cif_ms:.1f}ms")

    # Decoder
    t2 = time.perf_counter()
    dec_inputs = {
        "acoustic_embeds": acoustic_embeds.astype(np.float32),
        "acoustic_embeds_lens": np.array([token_num], dtype=np.int64),
        "encoder_out": encoder_out.astype(np.float32),
        "encoder_out_lens": np.array([encoder_out.shape[1]], dtype=np.int64),
    }
    dec_out = decoder.infer(dec_inputs)
    dec_ms = (time.perf_counter() - t2) * 1000
    logits = dec_out["logits"]
    print(f"  [Decoder] → {logits.shape}, {dec_ms:.1f}ms")

    total_infer_ms = (time.perf_counter() - t0) * 1000

    # ====== 解码 ======
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
    print(f"  总推理:    {total_infer_ms:.1f}ms")
    rtf = (total_infer_ms / 1000) / audio_duration
    rtx = audio_duration / (total_infer_ms / 1000)
    print(f"  RTF:       {rtf:.4f}")
    print(f"  RTX:       {rtx:.1f}x")


if __name__ == "__main__":
    main()
