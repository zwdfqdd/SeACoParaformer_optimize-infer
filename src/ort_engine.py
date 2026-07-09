"""
ONNX Runtime 分段串联推理引擎（ORT 后端字级时间戳路径）

用途：ENABLE_WORD_TIMESTAMP 开启且走 ORT 后端时，用分段 ONNX 串联推理，
      暴露 encoder_out 供 timestamp 段计算字级时间戳（整体模型 model.onnx 无法做到）。

模型组成（与 TRT 分段架构对齐，5 段独立 ONNX session）：
    encoder.onnx       — speech → encoder_out
    cif.onnx           — encoder_out + mask → acoustic_embeds, token_num, alphas, cif_peak
    decoder.onnx       — acoustic_embeds + token_num + encoder_out + encoder_out_lens + bias_embed → logits
    bias_encoder.onnx  — hotword_ids → hw_embed（外部按热词长度切片得到 bias_embed）
    timestamp.onnx     — encoder_out + mask + token_num → us_alphas, us_cif_peak（第 5 段，可选）

对外接口（与 src/trt_engine.py / asr_engine.py 一致）：
    infer_batch_raw(padded_feats, lengths, bias_embeddings) → list[(logits, ts_data)]
    encode_hotwords(hotword_token_ids) → bias_embed (1, num_hw, 512) | None

设备：GPU 可用则用 CUDAExecutionProvider，否则 CPUExecutionProvider。
"""

import os

import numpy as np
import onnxruntime as ort

from src.config import settings
from src.logger import logger


class ORTSplitEngine:
    """ORT 分段串联推理引擎（5 段独立 session）。"""

    def __init__(self):
        self._encoder: ort.InferenceSession | None = None
        self._cif: ort.InferenceSession | None = None
        self._decoder: ort.InferenceSession | None = None
        self._bias_encoder: ort.InferenceSession | None = None
        self._timestamp: ort.InferenceSession | None = None
        self._loaded = False
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def has_bias_encoder(self) -> bool:
        return self._bias_encoder is not None

    @property
    def has_timestamp(self) -> bool:
        return self._timestamp is not None

    # ------------------------------------------------------------
    def _providers(self, device: str) -> list:
        if device == "cuda":
            return [
                # kSameAsRequested：按需分配 arena，不翻倍预占，多 worker 省显存
                ("CUDAExecutionProvider", {"device_id": 0,
                                           "arena_extend_strategy": "kSameAsRequested"}),
                "CPUExecutionProvider",
            ]
        return ["CPUExecutionProvider"]

    def _new_session(self, path: str, device: str) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CIF 输出动态 token 数，关闭内存复用避免 broadcast 失败
        so.enable_mem_pattern = False
        so.enable_cpu_mem_arena = False
        if device == "cpu":
            intra = settings.ORT_INTRA_OP_THREADS or (os.cpu_count() or 4)
            so.intra_op_num_threads = intra
            so.inter_op_num_threads = settings.ORT_INTER_OP_THREADS
        return ort.InferenceSession(path, so, providers=self._providers(device))

    def load(self, encoder_path: str, cif_path: str, decoder_path: str,
             bias_encoder_path: str | None = None,
             timestamp_path: str | None = None,
             device: str = "cpu"):
        """加载分段 ONNX（encoder/cif/decoder 必需，bias/timestamp 可选）。"""
        self._device = device
        for label, path in [("encoder", encoder_path), ("cif", cif_path),
                            ("decoder", decoder_path)]:
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"ORT 分段缺失 {label}: {path}")

        logger.info(f"ORT 分段 ONNX 加载中（device={device}）:")
        logger.info(f"  encoder: {encoder_path}")
        self._encoder = self._new_session(encoder_path, device)
        logger.info(f"  cif:     {cif_path}")
        self._cif = self._new_session(cif_path, device)
        logger.info(f"  decoder: {decoder_path}")
        self._decoder = self._new_session(decoder_path, device)

        if bias_encoder_path and os.path.exists(bias_encoder_path):
            logger.info(f"  bias_encoder: {bias_encoder_path}")
            self._bias_encoder = self._new_session(bias_encoder_path, device)
        else:
            logger.info("  bias_encoder: 未加载（热词功能不可用）")

        if timestamp_path and os.path.exists(timestamp_path):
            logger.info(f"  timestamp: {timestamp_path}（字级时间戳启用）")
            self._timestamp = self._new_session(timestamp_path, device)
        else:
            logger.info("  timestamp: 未加载（字级时间戳关闭，words 为空）")

        self._loaded = True

    # ------------------------------------------------------------
    def encode_hotwords(self, hotword_token_ids: np.ndarray) -> np.ndarray | None:
        """bias_encoder 推理 + 按热词长度切片得到 bias_embed (1, H, 512)。

        逻辑与 trt_engine.encode_hotwords 一致：hw_embed (L,H,D) → 取每词最后有效步。
        """
        if self._bias_encoder is None:
            return None
        inp = self._bias_encoder.get_inputs()[0].name
        out = self._bias_encoder.run(None, {inp: hotword_token_ids.astype(np.int64)})
        hw_embed = out[0]  # (L, H, D)

        hotword_lengths = (hotword_token_ids != 0).sum(axis=1) - 1  # (H,)
        hotword_lengths[-1] = 0  # 最后一项 [sos] 哨兵固定取 0
        hotword_lengths = np.clip(hotword_lengths, 0, None)

        hw_embed_t = hw_embed.transpose(1, 0, 2)  # (H, L, D)
        bias_list = [hw_embed_t[i, hotword_lengths[i], :] for i in range(hw_embed_t.shape[0])]
        bias_embed = np.stack(bias_list, axis=0)[np.newaxis, :, :].astype(np.float32)
        return bias_embed

    # ------------------------------------------------------------
    def infer_batch_raw(
        self,
        padded_feats: np.ndarray,
        lengths: np.ndarray,
        bias_embeddings: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, dict | None]]:
        """5 段串联推理，返回 [(logits, ts_data), ...]。

        ts_data = {"us_alphas","us_cif_peak","num_tokens"} 或 None（未启用 timestamp）。
        """
        if not self._loaded:
            raise RuntimeError("ORT 分段 engine 未加载")

        batch_size = padded_feats.shape[0]
        real_max = max(1, int(max(int(L) for L in lengths)))
        min_seq = min(settings.BUCKET_SEQ_LENS)
        real_max = max(real_max, min_seq)
        if real_max < padded_feats.shape[1]:
            padded_feats = padded_feats[:, :real_max, :]
        seq_len = padded_feats.shape[1]

        # 1. Encoder
        enc_out = self._encoder.run(None, {"speech": padded_feats.astype(np.float32)})
        encoder_out = enc_out[0]  # (B, T, D)
        encoder_out_lens = lengths.astype(np.int64)

        # 2. CIF：mask 按真实长度构造
        mask = np.zeros((batch_size, 1, seq_len), dtype=np.float32)
        for i, L in enumerate(lengths):
            mask[i, 0, :int(L)] = 1.0
        cif_out = self._cif.run(None, {"encoder_out": encoder_out, "mask": mask})
        # cif 输出顺序：acoustic_embeds, token_num, alphas, cif_peak
        acoustic_embeds = cif_out[0]
        token_num_arr = cif_out[1].flatten()

        # timestamp（第 5 段，可选）
        us_alphas_arr = us_cif_peak_arr = None
        if self._timestamp is not None:
            ts_out = self._timestamp.run(None, {
                "encoder_out": encoder_out,
                "mask": mask,
                "token_num": np.round(token_num_arr).astype(np.float32),
            })
            us_alphas_arr, us_cif_peak_arr = ts_out[0], ts_out[1]

        # 3. Decoder（按 max_tok 截断 acoustic_embeds）
        token_nums = np.round(token_num_arr).astype(np.int64)
        max_tok = int(token_nums.max())
        if max_tok == 0:
            return [(np.zeros((0, 8404), dtype=np.float32), None) for _ in range(batch_size)]

        acoustic_trimmed = acoustic_embeds[:, :max_tok, :].astype(np.float32)
        if bias_embeddings is None:
            bias_embed_input = np.zeros((1, 1, 512), dtype=np.float32)
        else:
            bias_embed_input = bias_embeddings.astype(np.float32)
        if bias_embed_input.shape[0] != batch_size:
            bias_embed_input = np.tile(bias_embed_input, (batch_size, 1, 1)).astype(np.float32)

        dec_out = self._decoder.run(None, {
            "acoustic_embeds": acoustic_trimmed,
            "token_num": token_nums.astype(np.int64),
            "encoder_out": encoder_out.astype(np.float32),
            "encoder_out_lens": encoder_out_lens.astype(np.int64),
            "bias_embed": bias_embed_input,
        })
        logits = dec_out[0]  # (B, max_tok, vocab)

        # 4. 按 token_num 切片 + 组装 ts_data
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

    # ------------------------------------------------------------
    def warmup(self, bucket_seq_lens: list[int], batch_sizes: list[int]):
        """预热：各 (seq_len, batch) 组合跑一次。"""
        if not self._loaded:
            return
        logger.info("ORT 分段 engine 预热中...")
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
                    logger.warning(f"  ORT 分段预热失败 batch={batch}, seq={seq_len}: {e}")
        logger.info(f"ORT 分段 engine 预热完成（{count} 个 shape）")


# 全局单例
ort_split_engine = ORTSplitEngine()
