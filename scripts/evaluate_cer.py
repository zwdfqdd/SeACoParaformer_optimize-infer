"""
数据集级 CER 批量评测（基准方案 vs 待测方案）

在整个音频数据集上跑两套 TRT engine（4 段串联），逐条对比识别文本，
统计平均 CER，判断待测方案是否满足阈值。

基准方案默认 fp16（已验证与 PT baseline 字符级一致），
待测方案默认 INT8（encoder/decoder int8_qdq + cif/bias fp16）。

engine 命名约定（models/asr/trt/ 下）：
    {gpu}_encoder_fp16.engine        {gpu}_encoder_int8_qdq.engine
    {gpu}_cif_fp16.engine
    {gpu}_decoder_fp16.engine        {gpu}_decoder_int8_qdq.engine
    {gpu}_bias_encoder_fp16.engine

用法：
    # 默认：fp16 基准 vs int8 待测，阈值 3%
    python scripts/evaluate_cer.py --audio-dir calib_data/audio_data

    # 含热词评测
    python scripts/evaluate_cer.py --audio-dir calib_data/audio_data --hotwords 埃文 账号

    # 自定义阈值 + 显式指定 engine
    python scripts/evaluate_cer.py --audio-dir calib_data/audio_data \\
        --threshold 0.03 \\
        --test-encoder ./models/asr/trt/2080_ti_encoder_int8_qdq.engine \\
        --test-decoder ./models/asr/trt/2080_ti_decoder_int8_qdq.engine

    # 导出逐条明细 CSV
    python scripts/evaluate_cer.py --audio-dir calib_data/audio_data --csv report_cer.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# CER（Levenshtein 编辑距离 / 参考长度）
# ============================================================
def cer(ref: str, hyp: str) -> float:
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


# ============================================================
# 自包含单 engine TRT 推理器
# ============================================================
class _SimpleTRT:
    def __init__(self, engine_path: str):
        import tensorrt as trt
        import torch
        self._torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"engine 反序列化失败: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, inputs: dict) -> dict:
        torch = self._torch
        for name in self.input_names:
            data = inputs[name]
            self.context.set_input_shape(name, data.shape)
            t = torch.from_numpy(data).cuda().contiguous()
            inputs.setdefault("_keep", []).append(t)
            self.context.set_tensor_address(name, t.data_ptr())
        d_outputs = {}
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            for i, s in enumerate(shape):
                if s <= 0:
                    shape[i] = list(inputs.values())[0].shape[0] if i == 0 else 300
            t = torch.zeros(shape, dtype=torch.float32, device="cuda")
            d_outputs[name] = t
            self.context.set_tensor_address(name, t.data_ptr())
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        results = {}
        for name, t in d_outputs.items():
            actual = tuple(self.context.get_tensor_shape(name))
            if all(s > 0 for s in actual):
                results[name] = t[tuple(slice(0, s) for s in actual)].cpu().numpy()
            else:
                results[name] = t.cpu().numpy()
        return results


# ============================================================
# 4 段串联推理器
# ============================================================
class Pipeline:
    def __init__(self, encoder, cif, decoder, bias_encoder, tokenizer,
                 cmvn_mean, cmvn_istd):
        self.encoder = _SimpleTRT(encoder)
        self.cif = _SimpleTRT(cif)
        self.decoder = _SimpleTRT(decoder)
        self.bias_encoder = _SimpleTRT(bias_encoder) if bias_encoder else None
        self.tokenizer = tokenizer
        self.cmvn_mean = cmvn_mean
        self.cmvn_istd = cmvn_istd

    def encode_hotwords(self, hotwords):
        if not hotwords or self.bias_encoder is None:
            return None
        encoded = [self.tokenizer.encode(hw) for hw in hotwords if hw]
        encoded.append([1])  # SeACo 哨兵
        max_len = max(len(ids) for ids in encoded)
        hotword_ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        for i, ids in enumerate(encoded):
            hotword_ids[i, :len(ids)] = ids
        out = self.bias_encoder.infer({"hotword": hotword_ids})
        hw_embed = out["hw_embed"]
        lengths = (hotword_ids != 0).sum(axis=1) - 1
        lengths[-1] = 0
        lengths = np.clip(lengths, 0, None)
        hw_t = hw_embed.transpose(1, 0, 2)
        bias = [hw_t[i, lengths[i], :] for i in range(len(encoded))]
        return np.stack(bias, axis=0)[np.newaxis, :, :].astype(np.float32)

    def infer(self, audio_path, bias_embed=None):
        from src.feature_extractor import extract_features
        pcm, sr = sf.read(audio_path, dtype="float32")
        if len(pcm.shape) > 1:
            pcm = pcm[:, 0]
        feats = extract_features(pcm, sample_rate=sr,
                                 cmvn_mean=self.cmvn_mean, cmvn_istd=self.cmvn_istd)
        speech = feats[np.newaxis, :, :].astype(np.float32)

        enc_inputs = {"speech": speech}
        if "speech_lengths" in self.encoder.input_names:
            enc_inputs["speech_lengths"] = np.array([feats.shape[0]], dtype=np.int64)
        enc_out = self.encoder.infer(enc_inputs)
        encoder_out = enc_out["encoder_out"]

        mask = np.ones((1, 1, encoder_out.shape[1]), dtype=np.float32)
        cif_out = self.cif.infer({"encoder_out": encoder_out, "mask": mask})
        token_num = int(np.round(cif_out["token_num"].flatten()[0]))
        if token_num == 0:
            return ""
        acoustic = cif_out["acoustic_embeds"][:, :token_num, :]

        dec_bias = bias_embed if bias_embed is not None else np.zeros((1, 1, 512), dtype=np.float32)
        dec_inputs = {}
        if "acoustic_embeds" in self.decoder.input_names:
            dec_inputs["acoustic_embeds"] = acoustic.astype(np.float32)
        if "encoder_out" in self.decoder.input_names:
            dec_inputs["encoder_out"] = encoder_out.astype(np.float32)
        if "bias_embed" in self.decoder.input_names:
            dec_inputs["bias_embed"] = dec_bias.astype(np.float32)
        if "token_num" in self.decoder.input_names:
            dec_inputs["token_num"] = np.array([token_num], dtype=np.int64)
        if "encoder_out_lens" in self.decoder.input_names:
            dec_inputs["encoder_out_lens"] = np.array([encoder_out.shape[1]], dtype=np.int64)
        logits = self.decoder.infer(dec_inputs)["logits"]
        token_ids = np.argmax(logits[0], axis=-1)
        return self.tokenizer.decode(token_ids)


# ============================================================
# engine 查找
# ============================================================
def find_engine(engine_dir: Path, module: str, suffix_candidates: list[str]) -> str | None:
    """按 *_{module}_{suffix}.engine 查找，suffix 按候选顺序优先。"""
    if not engine_dir.is_dir():
        return None
    files = list(engine_dir.glob("*.engine"))

    def matches(fname, mod, suf):
        stem = fname[:-7]
        if not stem.endswith(f"_{suf}"):
            return False
        stem2 = stem[:-(len(suf) + 1)]
        if not stem2.endswith(f"_{mod}"):
            return False
        if mod == "encoder":  # 排除 bias_encoder
            base = stem2[:-(len(mod) + 1)]
            if base.endswith("_bias") or base == "bias":
                return False
        return True

    for suf in suffix_candidates:
        for f in files:
            if matches(f.name, module, suf):
                return str(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="数据集级 CER 批量评测")
    parser.add_argument("--audio-dir", default="calib_data/audio_data",
                        help="测试音频目录（递归扫描 *.wav）")
    parser.add_argument("--engine-dir", default="./models/asr/trt")
    parser.add_argument("--config-dir", default="./models/asr/pt")
    parser.add_argument("--threshold", type=float, default=0.03,
                        help="CER 阈值（默认 0.03 = 3%）")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    parser.add_argument("--max-samples", type=int, default=0, help="最多评测条数（0=全部）")
    parser.add_argument("--csv", default=None, help="导出逐条明细 CSV")

    # 基准方案 engine（默认 fp16）
    parser.add_argument("--base-encoder", default=None)
    parser.add_argument("--base-cif", default=None)
    parser.add_argument("--base-decoder", default=None)
    parser.add_argument("--base-bias", default=None)
    # 待测方案 engine（默认 int8_qdq encoder/decoder + fp16 cif/bias）
    parser.add_argument("--test-encoder", default=None)
    parser.add_argument("--test-cif", default=None)
    parser.add_argument("--test-decoder", default=None)
    parser.add_argument("--test-bias", default=None)
    args = parser.parse_args()

    engine_dir = Path(args.engine_dir)

    # 解析基准 engine（fp16）
    base = {
        "encoder": args.base_encoder or find_engine(engine_dir, "encoder", ["fp16"]),
        "cif": args.base_cif or find_engine(engine_dir, "cif", ["fp16"]),
        "decoder": args.base_decoder or find_engine(engine_dir, "decoder", ["fp16"]),
        "bias_encoder": args.base_bias or find_engine(engine_dir, "bias_encoder", ["fp16"]),
    }
    # 解析待测 engine（int8_qdq 优先，回退 int8）
    test = {
        "encoder": args.test_encoder or find_engine(engine_dir, "encoder", ["int8_qdq", "int8"]),
        "cif": args.test_cif or find_engine(engine_dir, "cif", ["fp16"]),
        "decoder": args.test_decoder or find_engine(engine_dir, "decoder", ["int8_qdq", "int8"]),
        "bias_encoder": args.test_bias or find_engine(engine_dir, "bias_encoder", ["fp16"]),
    }

    print("=" * 70)
    print("数据集级 CER 评测")
    print("=" * 70)
    print(f"基准方案 engine:")
    for k, v in base.items():
        print(f"  {k:14s}: {v}")
    print(f"待测方案 engine:")
    for k, v in test.items():
        print(f"  {k:14s}: {v}")
    print(f"CER 阈值: {args.threshold*100:.1f}%")
    print(f"热词: {args.hotwords}")
    print("=" * 70)

    for label, eng in [("基准", base), ("待测", test)]:
        if not eng["encoder"] or not eng["cif"] or not eng["decoder"]:
            sys.exit(f"{label}方案 engine 不齐全: {eng}")

    # 加载配置
    from src.feature_extractor import load_cmvn
    from src.tokenizer import Tokenizer
    cmvn_mean, cmvn_istd = load_cmvn(os.path.join(args.config_dir, "am.mvn"))
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(args.config_dir, "tokens.json"))

    # 收集音频
    audios = sorted([str(p) for p in Path(args.audio_dir).rglob("*.wav")])
    if args.max_samples > 0:
        audios = audios[:args.max_samples]
    if not audios:
        sys.exit(f"未在 {args.audio_dir} 找到 .wav")
    print(f"\n测试音频: {len(audios)} 条\n")

    # 构建两套流水线
    print("加载基准流水线...")
    base_pipe = Pipeline(base["encoder"], base["cif"], base["decoder"],
                         base["bias_encoder"], tokenizer, cmvn_mean, cmvn_istd)
    print("加载待测流水线...")
    test_pipe = Pipeline(test["encoder"], test["cif"], test["decoder"],
                         test["bias_encoder"], tokenizer, cmvn_mean, cmvn_istd)

    base_bias = base_pipe.encode_hotwords(args.hotwords)
    test_bias = test_pipe.encode_hotwords(args.hotwords)

    # 逐条评测
    rows = []
    cer_sum = 0.0
    fail_count = 0
    t0 = time.perf_counter()
    for idx, audio in enumerate(audios):
        ref = base_pipe.infer(audio, base_bias)
        hyp = test_pipe.infer(audio, test_bias)
        c = cer(ref, hyp)
        cer_sum += c
        if c > args.threshold:
            fail_count += 1
        rows.append((Path(audio).name, c, ref, hyp))
        flag = "" if c <= args.threshold else "  ✗超阈值"
        print(f"[{idx+1}/{len(audios)}] CER={c*100:5.2f}%{flag}  {Path(audio).name}")
        if c > args.threshold:
            print(f"      基准: {ref}")
            print(f"      待测: {hyp}")

    elapsed = time.perf_counter() - t0
    avg_cer = cer_sum / len(audios)

    # 汇总
    print("\n" + "=" * 70)
    print("评测汇总")
    print("=" * 70)
    print(f"  样本数:       {len(audios)}")
    print(f"  平均 CER:     {avg_cer*100:.3f}%")
    print(f"  阈值:         {args.threshold*100:.1f}%")
    print(f"  超阈值条数:   {fail_count} ({fail_count/len(audios)*100:.1f}%)")
    print(f"  评测耗时:     {elapsed:.1f}s")
    print("-" * 70)
    if avg_cer <= args.threshold:
        print(f"  ✓ 通过：平均 CER {avg_cer*100:.3f}% ≤ 阈值 {args.threshold*100:.1f}%")
    else:
        print(f"  ✗ 未通过：平均 CER {avg_cer*100:.3f}% > 阈值 {args.threshold*100:.1f}%")
    print("=" * 70)

    # CSV 明细
    if args.csv:
        import csv
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["audio", "cer", "baseline_text", "test_text"])
            for name, c, ref, hyp in rows:
                w.writerow([name, f"{c*100:.2f}%", ref, hyp])
        print(f"\n逐条明细已导出: {args.csv}")


if __name__ == "__main__":
    main()
