"""
PyTorch 原生推理引擎（PT 后端，GPU 优先 / CPU 兜底）

用途：MODEL_PRECISION=pt 时直接用 SeacoParaformer 权重推理，无需 ONNX/TRT 转换。
      主要用于无 TRT 环境、精度基线核对、快速验证。

模型编排（复用 seaco_paraformer 已验证的子模块，与分段 ONNX 逻辑一致）：
    encode → calc_predictor(CIF, 输出 encoder_out/token_num/alphas)
    → decoder + SeACo（外部 bias_embed）→ logits
    → 可选 predictor.get_upsample_timestamp（字级时间戳）

对外接口（与 trt_engine / ort_engine 一致）：
    infer_batch_raw(padded_feats, lengths, bias_embeddings) → list[(logits, ts_data)]
    encode_hotwords(hotword_token_ids) → bias_embed (1, num_hw, 512) | None

三功能开关（时间戳/热词/Faiss）与其他后端一致，由上层 main.py + ENABLE_WORD_TIMESTAMP 控制。
"""

import numpy as np

from src.config import settings
from src.logger import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PTEngine:
    """PyTorch 原生推理引擎（GPU 优先/CPU 兜底）。"""

    def __init__(self):
        self._model = None
        self._device = "cpu"
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def has_bias_encoder(self) -> bool:
        # PT 模型自带 bias_encoder 子模块
        return self._model is not None and hasattr(self._model, "bias_encoder")

    @property
    def has_timestamp(self) -> bool:
        # ENABLE_WORD_TIMESTAMP 开启且 predictor 支持 upsample 时间戳
        return (
            settings.ENABLE_WORD_TIMESTAMP
            and self._model is not None
            and hasattr(self._model.predictor, "get_upsample_timestamp")
        )

    # ------------------------------------------------------------
    def load(self, model_dir: str | None = None, device: str | None = None):
        """加载 PT 权重。device 未指定则 GPU 优先/CPU 兜底。"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch 未安装，PT 后端不可用")
        from seaco_paraformer.load_model import load_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        model_dir = model_dir or settings.PT_MODEL_DIR

        logger.info(f"PT 模型加载中（device={device}, dir={model_dir}）...")
        self._model = load_model(model_dir, device=device)
        self._model.eval()
        self._loaded = True
        logger.info("PT 模型加载完成")

    # ------------------------------------------------------------
    def _make_mask(self, lengths: np.ndarray, seq_len: int) -> "torch.Tensor":
        """构造 (B, 1, T) mask（1=有效帧）。"""
        b = len(lengths)
        mask = torch.zeros((b, 1, seq_len), dtype=torch.float32, device=self._device)
        for i, L in enumerate(lengths):
            mask[i, 0, :int(L)] = 1.0
        return mask

    def encode_hotwords(self, hotword_token_ids: np.ndarray) -> np.ndarray | None:
        """bias_encoder 编码热词 → bias_embed (1, H, 512)。

        复用 model._hotword_representation（LSTM + 取每词最后有效步）。
        """
        if not self.has_bias_encoder:
            return None
        try:
            hw_pad = torch.from_numpy(hotword_token_ids.astype(np.int64)).to(self._device)
            hw_lengths = torch.from_numpy(
                (hotword_token_ids != 0).sum(axis=1).astype(np.int64)
            ).clamp(min=1).to(self._device)
            with torch.no_grad():
                selected = self._model._hotword_representation(hw_pad, hw_lengths)  # (H, D)
            bias_embed = selected.unsqueeze(0).float().cpu().numpy()  # (1, H, D)
            return bias_embed
        except Exception as e:
            logger.warning(f"PT bias encoder 推理失败: {e}")
            return None

    # ------------------------------------------------------------
    def infer_batch_raw(
        self,
        padded_feats: np.ndarray,
        lengths: np.ndarray,
        bias_embeddings: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, dict | None]]:
        """PT 串联推理，返回 [(logits, ts_data), ...]。"""
        if not self._loaded:
            raise RuntimeError("PT engine 未加载")

        batch_size = padded_feats.shape[0]
        real_max = max(1, int(max(int(L) for L in lengths)))
        min_seq = min(settings.BUCKET_SEQ_LENS)
        real_max = max(real_max, min_seq)
        if real_max < padded_feats.shape[1]:
            padded_feats = padded_feats[:, :real_max, :]
        seq_len = padded_feats.shape[1]

        with torch.no_grad():
            speech = torch.from_numpy(padded_feats.astype(np.float32)).to(self._device)
            speech_lengths = torch.from_numpy(lengths.astype(np.int64)).to(self._device)

            # 1. encode + CIF predictor
            encoder_out, encoder_out_lens = self._model.encode(speech, speech_lengths)
            acoustic_embeds, token_num, alphas, cif_peak = self._model.calc_predictor(
                encoder_out, encoder_out_lens
            )
            pre_token_length = token_num.round().long()

            # 2. timestamp（可选）
            us_alphas_arr = us_cif_peak_arr = None
            if self.has_timestamp:
                mask = self._make_mask(lengths, seq_len)
                us_alphas, us_cif_peak = self._model.predictor.get_upsample_timestamp(
                    encoder_out, mask=mask, token_num=token_num.round()
                )
                us_alphas_arr = us_alphas.cpu().numpy()
                us_cif_peak_arr = us_cif_peak.cpu().numpy()

            # 3. decoder + SeACo（外部 bias_embed；无热词走纯 decoder）
            if bias_embeddings is not None:
                bias_embed = torch.from_numpy(bias_embeddings.astype(np.float32)).to(self._device)
                if bias_embed.shape[0] != batch_size:
                    bias_embed = bias_embed.repeat(batch_size, 1, 1)
                logits_t = self._decode_with_bias(
                    encoder_out, encoder_out_lens, acoustic_embeds,
                    pre_token_length, bias_embed,
                )
            else:
                decoder_out, _ = self._model.decoder(
                    encoder_out, encoder_out_lens, acoustic_embeds, pre_token_length,
                )
                logits_t = torch.log_softmax(decoder_out, dim=-1)

            logits = logits_t.cpu().numpy()

        token_nums = pre_token_length.cpu().numpy().astype(np.int64)
        results = []
        has_ts = us_alphas_arr is not None and us_cif_peak_arr is not None
        up_ratio = (us_alphas_arr.shape[1] // seq_len) if has_ts and seq_len > 0 else 1
        for i in range(batch_size):
            n = int(token_nums[i])
            logits_i = logits[i, :n, :].copy()
            if has_ts:
                real_up = min(int(lengths[i]) * up_ratio, us_alphas_arr.shape[1])
                ts_data = {
                    "us_alphas": us_alphas_arr[i, :real_up].copy(),
                    "us_cif_peak": us_cif_peak_arr[i, :real_up].copy(),
                    "num_tokens": n,
                }
            else:
                ts_data = None
            results.append((logits_i, ts_data))
        return results

    def _decode_with_bias(self, encoder_out, encoder_out_lens, acoustic_embeds,
                          token_num, bias_embed):
        """Decoder + SeACo（外部 bias_embed，逻辑同 export DecoderWithSeACoWrapper）。"""
        m = self._model
        decoder_out, decoder_hidden, _ = m.decoder(
            encoder_out, encoder_out_lens, acoustic_embeds, token_num,
            return_hidden=True, return_both=True,
        )
        decoder_pred = torch.log_softmax(decoder_out, dim=-1)

        B, H, D = bias_embed.shape
        contextual_length = torch.full((B,), H, dtype=torch.long, device=bias_embed.device)
        cif_attended, _ = m.seaco_decoder(bias_embed, contextual_length, acoustic_embeds, token_num)
        dec_attended, _ = m.seaco_decoder(bias_embed, contextual_length, decoder_hidden, token_num)

        merged = cif_attended + dec_attended
        dha_output = m.hotword_output_layer(merged)
        dha_pred = torch.log_softmax(dha_output, dim=-1)

        lmbd = m.seaco_weight
        a = (1.0 - lmbd) / lmbd
        b = 1.0 / lmbd
        dha_ids = dha_pred.max(-1)[1]
        dha_mask = (dha_ids == m.NO_BIAS).int().unsqueeze(-1).float()
        dha_mask_scaled = (dha_mask + a) / b
        return decoder_pred * dha_mask_scaled + dha_pred * (1.0 - dha_mask_scaled)

    # ------------------------------------------------------------
    def warmup(self, bucket_seq_lens: list[int], batch_sizes: list[int]):
        if not self._loaded:
            return
        logger.info("PT engine 预热中...")
        feat_dim = 560
        count = 0
        for seq_len in bucket_seq_lens:
            for batch in batch_sizes:
                try:
                    feats = np.random.randn(batch, seq_len, feat_dim).astype(np.float32)
                    lens = np.full(batch, seq_len, dtype=np.int32)
                    self.infer_batch_raw(feats, lens, None)
                    count += 1
                except Exception as e:
                    logger.warning(f"  PT 预热失败 batch={batch}, seq={seq_len}: {e}")
        logger.info(f"PT engine 预热完成（{count} 个 shape）")


# 全局单例
pt_engine = PTEngine()
