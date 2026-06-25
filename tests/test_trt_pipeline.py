"""
TRT 分段模型端到端推理测试（v2，含热词支持）

支持 encoder/cif/decoder/bias_encoder 各自独立指定精度，便于逐步验证 fp16 对各部分的影响。

用法：
    # 全 fp32
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav

    # 各部分独立精度
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav \\
        --encoder-precision fp16 --cif-precision fp16 \\
        --decoder-precision fp32 --bias-precision fp16

    # 含热词
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav \\
        --hotwords 埃文 账号 --encoder-precision fp32 --decoder-precision fp32
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
        d_inputs = {}
        for name in self.input_names:
            data = inputs[name]
            self.context.set_input_shape(name, data.shape)
            t = torch.from_numpy(data).cuda().contiguous()
            d_inputs[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

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

        stream = torch.cuda.Stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        results = {}
        for name, t in d_outputs.items():
            actual_shape = tuple(self.context.get_tensor_shape(name))
            if all(s > 0 for s in actual_shape):
                slices = tuple(slice(0, s) for s in actual_shape)
                results[name] = t[slices].cpu().numpy()
            else:
                results[name] = t.cpu().numpy()
        return results


def find_engine(engine_dir: Path, keyword: str, precision: str) -> str:
    """按关键字 + 精度查找 engine 文件。

    严格匹配格式 {gpu}_{keyword}_{precision}.engine，区分 'encoder' 和 'bias_encoder'。
    int8 精度优先匹配带 _qdq 后缀的 QDQ 量化产物。
    """
    candidates = list(engine_dir.glob("*.engine"))

    def matches(filename: str, kw: str, prec: str) -> bool:
        """文件名拆分后中间段必须正好是 kw（按 _ 分割）。"""
        if not filename.endswith(".engine"):
            return False
        stem = filename[:-7]  # 去掉 ".engine"
        # int8 允许 _qdq 后缀
        if prec == "int8":
            if stem.endswith("_int8_qdq"):
                stem_no_prec = stem[:-len("_int8_qdq")]
            elif stem.endswith("_int8"):
                stem_no_prec = stem[:-len("_int8")]
            else:
                return False
        else:
            suffix = f"_{prec}"
            if not stem.endswith(suffix):
                return False
            stem_no_prec = stem[:-len(suffix)]
        # stem_no_prec 应该以 _{kw} 结尾
        if not stem_no_prec.endswith(f"_{kw}"):
            return False
        # 排除 'bias_encoder' 误匹配 'encoder'
        if kw == "encoder":
            base_no_kw = stem_no_prec[:-(len(kw) + 1)]  # 去掉 _encoder
            if base_no_kw.endswith("_bias") or base_no_kw == "bias":
                return False
        return True

    # int8 优先匹配 _qdq 后缀
    if precision == "int8":
        for f in candidates:
            if f.name.endswith("_qdq.engine") and matches(f.name, keyword, precision):
                return str(f)

    # 匹配指定 precision
    for f in candidates:
        if matches(f.name, keyword, precision):
            return str(f)
    # fallback：任意精度
    for prec in ["fp32", "fp16", "int8"]:
        for f in candidates:
            if matches(f.name, keyword, prec):
                return str(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="TRT 分段模型推理测试（含热词支持，各部分独立精度）")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--engine-dir", default="./models/asr/trt", help="TRT engine 目录")
    parser.add_argument("--config-dir", default="./models/asr", help="配置文件目录")
    parser.add_argument("--encoder-precision", default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--cif-precision", default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--decoder-precision", default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--bias-precision", default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--encoder-engine", default=None, help="直接指定 encoder engine 路径")
    parser.add_argument("--cif-engine", default=None, help="直接指定 cif engine 路径")
    parser.add_argument("--decoder-engine", default=None, help="直接指定 decoder engine 路径")
    parser.add_argument("--bias-engine", default=None, help="直接指定 bias engine 路径")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"音频不存在: {args.audio}")

    print("=" * 60)
    print("TRT 分段模型推理测试")
    print(f"  encoder: {args.encoder_precision}")
    print(f"  cif:     {args.cif_precision}")
    print(f"  decoder: {args.decoder_precision}")
    print(f"  bias:    {args.bias_precision}")
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

    # 加载 engines
    print("\n[4/5] 加载 TRT engines...")
    engine_dir = Path(args.engine_dir)

    encoder_path = args.encoder_engine or find_engine(engine_dir, "encoder", args.encoder_precision)
    cif_path = args.cif_engine or find_engine(engine_dir, "cif", args.cif_precision)
    decoder_path = args.decoder_engine or find_engine(engine_dir, "decoder", args.decoder_precision)
    bias_path = args.bias_engine or find_engine(engine_dir, "bias_encoder", args.bias_precision)

    if not all([encoder_path, cif_path, decoder_path]):
        sys.exit(f"未找到 engine: encoder={encoder_path}, cif={cif_path}, decoder={decoder_path}")

    encoder = TRTInferencer(encoder_path)
    cif = TRTInferencer(cif_path)
    decoder = TRTInferencer(decoder_path)
    bias_encoder = TRTInferencer(bias_path) if bias_path else None

    print(f"  encoder: {encoder_path}")
    print(f"  cif: {cif_path}")
    print(f"  decoder: {decoder_path}")
    print(f"  bias: {bias_path or '未找到'}")

    # 编码热词
    bias_embed = None
    if args.hotwords and bias_encoder:
        print(f"\n  编码热词: {args.hotwords}")
        encoded = [tokenizer.encode(hw) for hw in args.hotwords if hw]
        encoded.append([1])
        max_len = max(len(ids) for ids in encoded)
        hotword_ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        for i, ids in enumerate(encoded):
            hotword_ids[i, :len(ids)] = ids

        bias_out = bias_encoder.infer({"hotword": hotword_ids})
        hw_embed = bias_out["hw_embed"]  # (max_len, num_hotwords, 512)

        hotword_lengths = (hotword_ids != 0).sum(axis=1) - 1
        hotword_lengths[-1] = 0
        hotword_lengths = np.clip(hotword_lengths, 0, None)
        hw_embed_t = hw_embed.transpose(1, 0, 2)
        bias_list = [hw_embed_t[i, hotword_lengths[i], :] for i in range(len(encoded))]
        bias_embed = np.stack(bias_list, axis=0)[np.newaxis, :, :].astype(np.float32)
        print(f"  bias_embed shape: {bias_embed.shape}")

    # 推理
    print("\n[5/5] 推理...")
    speech_input = features[np.newaxis, :, :].astype(np.float32)

    t0 = time.perf_counter()

    # Encoder
    enc_inputs = {"speech": speech_input}
    if "speech_lengths" in encoder.input_names:
        enc_inputs["speech_lengths"] = np.array([features.shape[0]], dtype=np.int64)
    enc_out = encoder.infer(enc_inputs)
    encoder_out = enc_out["encoder_out"]
    encoder_out_lens = enc_out.get("encoder_out_lens",
                                    np.array([encoder_out.shape[1]], dtype=np.int64))
    enc_ms = (time.perf_counter() - t0) * 1000
    has_nan_inf_enc = bool(np.isnan(encoder_out).any() or np.isinf(encoder_out).any())
    print(f"  [Encoder] {speech_input.shape} → {encoder_out.shape}, "
          f"max={encoder_out.max():.4f}, std={encoder_out.std():.4f}, "
          f"nan/inf={has_nan_inf_enc}, {enc_ms:.1f}ms")

    # CIF
    t1 = time.perf_counter()
    mask = np.ones((1, 1, encoder_out.shape[1]), dtype=np.float32)
    cif_out = cif.infer({"encoder_out": encoder_out, "mask": mask})
    cif_ms = (time.perf_counter() - t1) * 1000
    acoustic_embeds = cif_out["acoustic_embeds"]
    token_num_raw = cif_out["token_num"].flatten()[0]
    has_nan_cif = bool(np.isnan(token_num_raw))
    if has_nan_cif:
        print(f"  [CIF] token_num=NaN！精度崩溃")
        return
    token_num = int(token_num_raw)
    acoustic_embeds = acoustic_embeds[:, :token_num, :]
    print(f"  [CIF] → {acoustic_embeds.shape} (token_num={token_num}), {cif_ms:.1f}ms")

    if token_num == 0:
        print("  警告：token_num=0，精度可能崩溃")
        return

    # Decoder
    t2 = time.perf_counter()
    if bias_embed is not None:
        dec_bias = bias_embed.astype(np.float32)
    else:
        dec_bias = np.zeros((1, 1, 512), dtype=np.float32)

    # 只传 decoder ONNX 实际接收的输入
    dec_inputs = {}
    if "acoustic_embeds" in decoder.input_names:
        dec_inputs["acoustic_embeds"] = acoustic_embeds.astype(np.float32)
    if "token_num" in decoder.input_names:
        dec_inputs["token_num"] = np.array([token_num], dtype=np.int64)
    if "encoder_out" in decoder.input_names:
        dec_inputs["encoder_out"] = encoder_out.astype(np.float32)
    if "encoder_out_lens" in decoder.input_names:
        dec_inputs["encoder_out_lens"] = encoder_out_lens.astype(np.int64)
    if "bias_embed" in decoder.input_names:
        dec_inputs["bias_embed"] = dec_bias

    dec_out = decoder.infer(dec_inputs)
    dec_ms = (time.perf_counter() - t2) * 1000
    logits = dec_out["logits"]
    has_nan_dec = bool(np.isnan(logits).any() or np.isinf(logits).any())
    print(f"  [Decoder] → {logits.shape}, nan/inf={has_nan_dec}, {dec_ms:.1f}ms")

    total_infer_ms = (time.perf_counter() - t0) * 1000

    # 解码
    token_ids = np.argmax(logits[0], axis=-1)
    text = tokenizer.decode(token_ids)

    print(f"\n{'='*60}")
    print(f"识别结果: {text}")
    if args.hotwords:
        print(f"热词: {args.hotwords}")
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
    print(f"  RTF:       {rtf:.4f}, RTX: {rtx:.1f}x")


if __name__ == "__main__":
    main()
