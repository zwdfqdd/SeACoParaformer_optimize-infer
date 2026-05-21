"""
TRT 分段模型端到端推理测试

串联 4 个 TRT engine 完成完整 ASR 推理：
1. model_eb (TRT) : hotword_ids → bias_embed
2. encoder  (TRT) : speech_features → encoder_out
3. cif      (TRT) : encoder_out → acoustic_embeds, token_num, us_alphas, cif_peak
4. decoder  (TRT) : acoustic_embeds + encoder_out + bias_embed → logits
5. tokenizer      : logits → text

用法：
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_30s.wav
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_30s.wav --hotwords 张三 李四
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_30s.wav --engine-dir ./models/asr/trt
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
    TRT_AVAILABLE = True
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
        """执行推理，返回输出字典。"""
        # 设置 input shapes 并绑定地址
        d_inputs = {}
        for name in self.input_names:
            data = inputs[name]
            self.context.set_input_shape(name, data.shape)
            t = torch.from_numpy(data).cuda().contiguous()
            d_inputs[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

        # 推断输出 shape（设置完所有 input shape 后）
        # 对于动态输出，TRT 可以在 execute 前推断出实际 shape
        d_outputs = {}
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            # 动态维度（-1）用输入的对应维度估算
            for i, s in enumerate(shape):
                if s <= 0:
                    # 用输入 batch 的第一个维度或合理最大值
                    if i == 0:
                        shape[i] = list(inputs.values())[0].shape[0]
                    else:
                        shape[i] = 300  # 足够大的 buffer
            t = torch.zeros(shape, dtype=torch.float32, device="cuda")
            d_outputs[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

        # 执行
        stream = torch.cuda.Stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        # 执行后获取实际输出 shape
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
    parser = argparse.ArgumentParser(description="TRT 分段模型端到端推理测试")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--engine-dir", default="./models/asr/trt", help="TRT engine 目录")
    parser.add_argument("--config-dir", default="./models/asr", help="配置文件目录（am.mvn, tokens.json）")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"], help="engine 精度（默认 fp32）")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"音频不存在: {args.audio}")

    print("=" * 60)
    print("TRT 分段模型端到端推理测试")
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
    print(f"  CMVN: {cmvn_path}")

    tokenizer = Tokenizer()
    vocab_path = os.path.join(args.config_dir, "tokens.json")
    tokenizer.load(vocab_path)
    print(f"  Tokenizer: {vocab_path} (vocab={tokenizer.vocab_size})")

    # ====== 特征提取 ======
    print("\n[3/7] 特征提取...")
    t0 = time.perf_counter()
    features = extract_features(pcm, sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
    feat_ms = (time.perf_counter() - t0) * 1000
    print(f"  shape: {features.shape}, 耗时: {feat_ms:.1f}ms")

    # ====== 加载 TRT engines ======
    print("\n[4/7] 加载 TRT engines...")
    engine_dir = Path(args.engine_dir)

    # 自动查找 engine 文件（按精度匹配）
    def find_engine(keyword):
        # 优先匹配指定精度
        for f in engine_dir.glob("*.engine"):
            if keyword in f.name and args.precision in f.name:
                return str(f)
        # fallback：任意精度
        for f in engine_dir.glob("*.engine"):
            if keyword in f.name:
                return str(f)
        return None

    encoder_path = find_engine("encoder")
    cif_path = find_engine("cif")
    decoder_path = find_engine("decoder")
    bias_path = find_engine("model_eb")

    if not encoder_path:
        sys.exit(f"未找到 encoder engine: {engine_dir}")
    if not cif_path:
        sys.exit(f"未找到 cif engine: {engine_dir}")
    if not decoder_path:
        sys.exit(f"未找到 decoder engine: {engine_dir}")

    encoder = TRTInferencer(encoder_path)
    cif = TRTInferencer(cif_path)
    decoder = TRTInferencer(decoder_path)
    bias_enc = TRTInferencer(bias_path) if bias_path else None

    print(f"  encoder: {encoder_path}")
    print(f"  cif: {cif_path}")
    print(f"  decoder: {decoder_path}")
    print(f"  bias_enc: {bias_path or '未找到'}")

    # ====== Bias Encoder（热词编码）======
    print("\n[5/7] 热词编码...")
    if args.hotwords and bias_enc:
        hw_ids = [tokenizer.encode(hw) for hw in args.hotwords if hw]
        if hw_ids:
            max_len = max(len(ids) for ids in hw_ids)
            hotword_input = np.zeros((len(hw_ids), max_len), dtype=np.int64)
            for i, ids in enumerate(hw_ids):
                hotword_input[i, :len(ids)] = ids

            t0 = time.perf_counter()
            bias_out = bias_enc.infer({"hotword": hotword_input})
            bias_ms = (time.perf_counter() - t0) * 1000
            bias_embed = bias_out[bias_enc.output_names[0]]  # (num_hw, 1, 512) or (num_hw, ?, 512)
            print(f"  热词: {args.hotwords}")
            print(f"  bias_embed shape: {bias_embed.shape}, 耗时: {bias_ms:.1f}ms")
        else:
            bias_embed = np.zeros((1, 1, 512), dtype=np.float32)
            print("  无有效热词，使用零向量")
    else:
        bias_embed = np.zeros((1, 1, 512), dtype=np.float32)
        print("  无热词")

    # ====== Encoder ======
    print("\n[6/7] Encoder + CIF + Decoder 推理...")
    speech_input = features[np.newaxis, :, :].astype(np.float32)  # (1, T, 560)
    speech_lengths = np.array([features.shape[0]], dtype=np.int64)

    t0 = time.perf_counter()

    # Encoder
    enc_inputs = {"speech": speech_input}
    # 如果 encoder 有 speech_lengths 输入则传入
    if "speech_lengths" in encoder.input_names:
        enc_inputs["speech_lengths"] = speech_lengths
    enc_out = encoder.infer(enc_inputs)
    encoder_out = enc_out[encoder.output_names[0]]  # (1, T, 512)
    enc_ms = (time.perf_counter() - t0) * 1000
    print(f"  [Encoder] 输入: speech{speech_input.shape}")
    print(f"  [Encoder] 输出: {[(k, v.shape) for k, v in enc_out.items()]}")
    print(f"  [Encoder] 耗时: {enc_ms:.1f}ms")

    # CIF
    t1 = time.perf_counter()
    cif_out = cif.infer({"encoder_out": encoder_out})
    cif_ms = (time.perf_counter() - t1) * 1000
    print(f"  [CIF] 输入: encoder_out{encoder_out.shape}")
    print(f"  [CIF] 输出: {[(k, v.shape) for k, v in cif_out.items()]}")

    # CIF 输出解析
    cif_output_names = cif.output_names
    acoustic_embeds_raw = cif_out[cif_output_names[0]]  # (1, buffer_len, 512)

    # 获取实际 token 数量（从 token_num 输出）
    token_num_val = None
    for name in cif_output_names:
        val = cif_out[name]
        if val.ndim <= 1 or (val.ndim == 2 and val.shape[-1] == 1):
            # 标量或 (batch,) — 这是 token_num
            token_num_val = int(val.flatten()[0])
            break

    if token_num_val and token_num_val > 0:
        acoustic_embeds = acoustic_embeds_raw[:, :token_num_val, :]
        print(f"  [CIF] token_num={token_num_val}, 截取 acoustic_embeds → {acoustic_embeds.shape}")
    else:
        # fallback：去掉全零尾部
        norms = np.linalg.norm(acoustic_embeds_raw[0], axis=-1)
        valid_len = np.max(np.where(norms > 1e-6)[0]) + 1 if np.any(norms > 1e-6) else acoustic_embeds_raw.shape[1]
        acoustic_embeds = acoustic_embeds_raw[:, :valid_len, :]
        print(f"  [CIF] token_num 未获取，norm fallback → valid_len={valid_len}, {acoustic_embeds.shape}")
    print(f"  [CIF] 耗时: {cif_ms:.1f}ms")

    # Decoder
    t2 = time.perf_counter()
    batch = acoustic_embeds.shape[0]
    token_len = acoustic_embeds.shape[1]

    # 准备 decoder 输入
    acoustic_embeds_lens = np.array([token_len], dtype=np.int64)
    encoder_out_lens = np.array([encoder_out.shape[1]], dtype=np.int64)

    # bias_embed 需要与 batch 对齐
    if bias_embed.shape[0] != batch:
        # (num_hw, 1, 512) → (batch, num_hw, 512) — 需要 reshape
        if bias_embed.ndim == 3 and bias_embed.shape[1] == 1:
            bias_embed_dec = bias_embed.squeeze(1)[np.newaxis, :, :]  # (1, num_hw, 512)
        else:
            bias_embed_dec = bias_embed[:1, :, :]  # 取第一个
    else:
        bias_embed_dec = bias_embed

    dec_inputs = {
        "acoustic_embeds": acoustic_embeds.astype(np.float32),
        "acoustic_embeds_lens": acoustic_embeds_lens,
        "encoder_out": encoder_out.astype(np.float32),
        "encoder_out_lens": encoder_out_lens,
        "bias_embed": bias_embed_dec.astype(np.float32),
    }

    dec_out = decoder.infer(dec_inputs)
    dec_ms = (time.perf_counter() - t2) * 1000

    logits = dec_out[decoder.output_names[0]]  # (1, N, 8404)
    total_infer_ms = (time.perf_counter() - t0) * 1000

    print(f"  [Decoder] 输入: {[(k, v.shape) for k, v in dec_inputs.items()]}")
    print(f"  [Decoder] 输出: {[(k, v.shape) for k, v in dec_out.items()]}")
    print(f"  [Decoder] 耗时: {dec_ms:.1f}ms")
    print(f"  总推理耗时: {total_infer_ms:.1f}ms")

    # ====== 解码 ======
    print("\n[7/7] 解码...")
    token_ids = np.argmax(logits[0], axis=-1)
    text = tokenizer.decode(token_ids)

    print("-" * 60)
    print(f"识别结果: {text}")
    print("-" * 60)

    # 性能汇总
    print(f"\n性能汇总:")
    print(f"  音频时长:    {audio_duration:.2f}s")
    print(f"  特征提取:    {feat_ms:.1f}ms")
    print(f"  Encoder:     {enc_ms:.1f}ms")
    print(f"  CIF:         {cif_ms:.1f}ms")
    print(f"  Decoder:     {dec_ms:.1f}ms")
    print(f"  总推理:      {total_infer_ms:.1f}ms")
    rtf = (total_infer_ms / 1000) / audio_duration
    rtx = audio_duration / (total_infer_ms / 1000)
    print(f"  RTF:         {rtf:.4f}")
    print(f"  RTX:         {rtx:.1f}x")


if __name__ == "__main__":
    main()
