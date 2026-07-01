"""
分段 ONNX 模型端到端推理测试（ORT，含热词支持）

使用 encoder.onnx + cif.onnx + decoder.onnx + bias_encoder.onnx 串联推理。

用法：
    python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_30s.wav
    python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_30s.wav --device cuda
    python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_30s.wav --hotwords 埃文 账号
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
    parser = argparse.ArgumentParser(description="分段 ONNX 模型推理测试（含热词支持）")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--split-dir", default="./models/asr/split", help="分段 ONNX 模型目录")
    parser.add_argument("--config-dir", default="./models/asr/pt", help="配置文件目录")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    parser.add_argument("--max-frames", type=int, default=0, help="截取特征前 N 帧（0=不截取）")
    parser.add_argument("--compare-pt", action="store_true",
                        help="诊断：用 PT 模型对同一 features 推理，逐段对比 encoder/CIF/decoder 定位分叉点")
    parser.add_argument("--model-id", default="./models/asr/pt", help="PT 模型本地目录（--compare-pt 用）")
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
    if args.max_frames and features.shape[0] > args.max_frames:
        features = features[:args.max_frames]
        print(f"  截取前 {args.max_frames} 帧")
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

    # Bias encoder（可选）
    bias_encoder_path = split_dir / "bias_encoder.onnx"
    bias_encoder = None
    if bias_encoder_path.exists():
        bias_encoder = OnnxInferencer(str(bias_encoder_path), args.device)
        print(f"  bias_encoder: {bias_encoder.input_names} → {bias_encoder.output_names}")

    # 编码热词
    bias_embed = None
    if args.hotwords and bias_encoder:
        print(f"\n  编码热词: {args.hotwords}")
        encoded = [tokenizer.encode(hw) for hw in args.hotwords if hw]
        encoded.append([1])  # <sos> as NO_BIAS marker
        max_len = max(len(ids) for ids in encoded)
        hotword_ids = np.zeros((len(encoded), max_len), dtype=np.int64)
        for i, ids in enumerate(encoded):
            hotword_ids[i, :len(ids)] = ids

        # Bias encoder 推理
        bias_out = bias_encoder.infer({"hotword": hotword_ids})
        hw_embed = bias_out["hw_embed"]  # (max_len, num_hotwords, 512)
        print(f"  hw_embed shape: {hw_embed.shape}")

        # 按热词长度取最后有效时间步
        hotword_lengths = (hotword_ids != 0).sum(axis=1) - 1
        hotword_lengths[-1] = 0  # <sos> 取位置 0
        hotword_lengths = np.clip(hotword_lengths, 0, None)

        # hw_embed: (max_len, num_hw, D) → transpose → (num_hw, max_len, D) → 取各自位置
        hw_embed_t = hw_embed.transpose(1, 0, 2)  # (num_hw, max_len, D)
        bias_list = []
        for i in range(len(encoded)):
            bias_list.append(hw_embed_t[i, hotword_lengths[i], :])
        bias_embed = np.stack(bias_list, axis=0)[np.newaxis, :, :]  # (1, num_hw, 512)
        print(f"  bias_embed shape: {bias_embed.shape}")

    # 推理
    print("\n[5/5] 串联推理...")
    speech_input = features[np.newaxis, :, :].astype(np.float32)
    speech_lengths = np.array([features.shape[0]], dtype=np.int64)

    t0 = time.perf_counter()

    # Encoder（兼容输入有/无 speech_lengths，输出有/无 encoder_out_lens）
    enc_inputs = {"speech": speech_input}
    if "speech_lengths" in encoder.input_names:
        enc_inputs["speech_lengths"] = speech_lengths
    enc_out = encoder.infer(enc_inputs)
    encoder_out = enc_out["encoder_out"]
    encoder_out_lens = enc_out.get(
        "encoder_out_lens",
        np.array([encoder_out.shape[1]], dtype=np.int64),
    )
    enc_ms = (time.perf_counter() - t0) * 1000
    print(f"  [Encoder] {speech_input.shape} → {encoder_out.shape}, {enc_ms:.1f}ms")

    # CIF
    t1 = time.perf_counter()
    # mask: (B, 1, T) — 全 True（推理时不 mask）
    mask = np.ones((1, 1, encoder_out.shape[1]), dtype=np.float32)
    cif_out = cif.infer({"encoder_out": encoder_out, "mask": mask})
    cif_ms = (time.perf_counter() - t1) * 1000

    acoustic_embeds = cif_out["acoustic_embeds"]
    # token_num 取整必须用 round（与 PT cif 的 torch.round(alphas.sum()) 一致）。
    # 用 int()（截断=floor）会在小数部分 >0.5 时少算 1 个 token，导致 decoder 对齐错位、中段重复。
    token_num = int(round(float(cif_out["token_num"].flatten()[0])))
    acoustic_embeds = acoustic_embeds[:, :token_num, :]
    print(f"  [CIF] → {acoustic_embeds.shape} (token_num={token_num}), {cif_ms:.1f}ms")

    if token_num == 0:
        print("\n  警告：token_num=0，无法继续推理")
        return

    # Decoder
    t2 = time.perf_counter()
    # 准备 bias_embed（无热词时用全零）
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
        dec_inputs["encoder_out_lens"] = np.array([encoder_out.shape[1]], dtype=np.int64)
    if "bias_embed" in decoder.input_names:
        dec_inputs["bias_embed"] = dec_bias

    dec_out = decoder.infer(dec_inputs)
    dec_ms = (time.perf_counter() - t2) * 1000
    logits = dec_out["logits"]
    print(f"  [Decoder] → {logits.shape}, {dec_ms:.1f}ms")

    total_ms = feat_ms + enc_ms + cif_ms + dec_ms

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
    print(f"  总耗时:    {total_ms:.1f}ms")
    rtf = (total_ms / 1000) / audio_duration
    rtx = audio_duration / (total_ms / 1000)
    print(f"  RTF:       {rtf:.4f}")
    print(f"  RTX:       {rtx:.1f}x")

    # ============================================================
    # --compare-pt 诊断：PT 对同一 features 推理，逐段对比定位分叉点
    # ============================================================
    if args.compare_pt:
        print(f"\n{'='*60}")
        print("【--compare-pt】PT vs ONNX 逐段对比")
        print(f"{'='*60}")
        import torch
        from seaco_paraformer.load_model import load_model

        model = load_model(model_id=args.model_id, device="cpu")
        feat_t = torch.from_numpy(features).unsqueeze(0).float()
        feat_len = torch.tensor([features.shape[0]], dtype=torch.long)

        with torch.no_grad():
            # 1. PT encoder
            pt_enc_out, pt_enc_lens = model.encode(feat_t, feat_len)
            pt_enc = pt_enc_out.cpu().numpy()
            d_enc = np.abs(pt_enc[:, :encoder_out.shape[1]] - encoder_out[:, :pt_enc.shape[1]]).max()
            print(f"\n[Encoder] PT {pt_enc.shape} vs ONNX {encoder_out.shape}")
            print(f"  encoder_out 最大绝对误差: {d_enc:.6f}  "
                  f"{'✓ 一致' if d_enc < 1e-2 else '✗ 分叉点在 encoder'}")
            # 诊断：PT/ONNX 激活峰值（看是否触及 clamp=60000）+ 误差分布
            print(f"  PT  encoder_out  abs max={np.abs(pt_enc).max():.2f}, mean={np.abs(pt_enc).mean():.4f}")
            print(f"  ONNX encoder_out abs max={np.abs(encoder_out).max():.2f}, mean={np.abs(encoder_out).mean():.4f}")
            err_per_tok = np.abs(pt_enc[0] - encoder_out[0]).max(axis=-1)  # (T,) 每 token 最大误差
            big = np.where(err_per_tok > 0.05)[0]
            print(f"  误差>0.05 的 token 数: {len(big)}/{len(err_per_tok)}"
                  f"{'（集中）' if 0 < len(big) <= 5 else '（分散）' if len(big) > 5 else ''}")
            if len(big) > 0:
                print(f"  误差最大的前5个token位置: {big[np.argsort(err_per_tok[big])[-5:]].tolist()}")

            # 2. PT predictor（CIF）—— 喂 PT 自己的 encoder_out
            pt_mask = (~_make_pad_mask(pt_enc_lens, maxlen=pt_enc_out.shape[1])[:, None, :]).to(pt_enc_out.device)
            pred_out = model.predictor(pt_enc_out, mask=pt_mask)
            pt_acoustic = pred_out[0].cpu().numpy()
            pt_token_num = pred_out[1].cpu().numpy()
            pt_tn = int(round(float(pt_token_num.flatten()[0])))
            print(f"\n[CIF] PT token_num={pt_tn} (raw={float(pt_token_num.flatten()[0]):.4f}) "
                  f"vs ONNX token_num={token_num} (raw={float(cif_out['token_num'].flatten()[0]):.4f})")
            # 对齐 token 数比较 acoustic
            onnx_acoustic_full = cif_out["acoustic_embeds"]
            ncmp = min(pt_tn, token_num, pt_acoustic.shape[1], onnx_acoustic_full.shape[1])
            d_ac = np.abs(pt_acoustic[:, :ncmp] - onnx_acoustic_full[:, :ncmp]).max()
            print(f"  前 {ncmp} token acoustic_embeds 最大绝对误差: {d_ac:.6f}  "
                  f"{'✓ 一致' if d_ac < 1e-2 else '✗ 分叉点在 CIF acoustic'}")

            # 3. PT 完整推理（含 decoder + SeACo）对比最终识别
            if args.hotwords:
                hw_list = [tokenizer.encode(hw) for hw in args.hotwords if hw]
                hw_list.append([model.sos])
                pt_logits, pt_tn2 = model.inference(feat_t, feat_len, hw_list=hw_list, nfilter=50)
            else:
                pt_logits, pt_tn2 = model(feat_t, feat_len)
            pt_text = tokenizer.decode(pt_logits[0].argmax(dim=-1).cpu().numpy())
            print(f"\n[最终识别对比]")
            print(f"  PT  : {pt_text}")
            print(f"  ONNX: {text}")
            # 逐 token argmax 对比，找第一个分叉位置
            pt_ids = pt_logits[0].argmax(dim=-1).cpu().numpy()
            onnx_ids = np.argmax(logits[0], axis=-1)
            nmin = min(len(pt_ids), len(onnx_ids))
            diff_pos = -1
            for k in range(nmin):
                if pt_ids[k] != onnx_ids[k]:
                    diff_pos = k
                    break
            print(f"  token 数: PT={len(pt_ids)}, ONNX={len(onnx_ids)}")
            print(f"  第一个 argmax 分叉位置: {'无（完全一致）' if diff_pos < 0 else diff_pos}")

            # 4. 交叉验证：PT encoder_out → ONNX cif+decoder，定位 encoder vs cif/decoder
            print(f"\n[交叉验证] 用 PT encoder_out 喂 ONNX cif+decoder：")
            pt_enc_np = pt_enc.astype(np.float32)
            mask_x = np.ones((1, 1, pt_enc_np.shape[1]), dtype=np.float32)
            cif_x = cif.infer({"encoder_out": pt_enc_np, "mask": mask_x})
            tn_x = int(round(float(cif_x["token_num"].flatten()[0])))
            ac_x = cif_x["acoustic_embeds"][:, :tn_x, :].astype(np.float32)
            print(f"  ONNX-cif(PT enc) token_num={tn_x} (raw={float(cif_x['token_num'].flatten()[0]):.4f})")
            dec_x = {}
            if "acoustic_embeds" in decoder.input_names:
                dec_x["acoustic_embeds"] = ac_x
            if "token_num" in decoder.input_names:
                dec_x["token_num"] = np.array([tn_x], dtype=np.int64)
            if "encoder_out" in decoder.input_names:
                dec_x["encoder_out"] = pt_enc_np
            if "encoder_out_lens" in decoder.input_names:
                dec_x["encoder_out_lens"] = np.array([pt_enc_np.shape[1]], dtype=np.int64)
            if "bias_embed" in decoder.input_names:
                dec_x["bias_embed"] = dec_bias
            logits_x = decoder.infer(dec_x)["logits"]
            text_x = tokenizer.decode(np.argmax(logits_x[0], axis=-1))
            print(f"  识别(PT enc → ONNX cif/dec): {text_x}")
            print(f"  → 若此结果正确：问题在 ONNX encoder；若仍乱：问题在 ONNX cif/decoder")


def _make_pad_mask(lengths, maxlen):
    import torch
    from seaco_paraformer.utils import make_pad_mask
    return make_pad_mask(lengths, maxlen=maxlen)


if __name__ == "__main__":
    main()
