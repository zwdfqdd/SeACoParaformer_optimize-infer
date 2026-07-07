"""
TensorRT 推理引擎（v2 阶段 1：4 段串联架构）

模型组成：
    encoder.engine       — speech → encoder_out
    cif.engine           — encoder_out + mask → acoustic_embeds, token_num
    decoder.engine       — acoustic_embeds + encoder_out + bias_embed → logits
    bias_encoder.engine  — hotword_ids → hw_embed（外部按热词长度切片得到 bias_embed）

精度方案（推荐：纯 fp16）：
    opset 17 + clamp 60000 + trtexec --fp16
    详见 docs/README.md 的 v2 推理路径。

对外接口（与 src/asr_engine.py 一致）：
    infer_batch_raw(padded_feats, lengths, bias_embeddings) → list[logits]
    encode_hotwords(hotword_token_ids) → bias_embed (1, num_hw, 512)
"""

import os
import queue

import numpy as np

from src.config import settings
from src.logger import logger

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TRT_AVAILABLE:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================
# 单个 TRT engine 推理器（多 stream 多 context）
# ============================================================
class _TRTInferencer:
    """
    单个 TRT engine 推理器（dynamic shape + 多 stream 并发）。

    - 一份 engine（weights 共享，加载一次）
    - N 个 execution_context（各自独立 activation memory）
    - N 个 CUDA stream（真正并行）
    - Queue 连接池分配：get() 借一个空闲 (context, stream)，用完 put() 归还

    TRT 10.x 要求（★重要）：
    - 同一 execution_context 不能被两个线程同时 execute（非线程安全）
    - 之前用 round-robin counter 取模分配有致命缺陷：counter 只递增不追踪
      context 忙闲，转一圈回到同一 idx 时会把仍在执行的 context 再分给另一线程，
      两线程同时 execute_async_v3 同一 context → CUDA illegal memory access。
    - 改用 queue.Queue 阻塞式借还：context 数 < 并发线程数时自动排队串行化，
      保证任一 context 同时只被一个线程持有。
    """

    def __init__(self, engine_path: str, pool_size: int = 1):
        self.engine_path = engine_path
        self.pool_size = max(1, pool_size)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Engine 反序列化失败: {engine_path}")

        # 池：N 个 execution_context + N 个 stream
        self.contexts = [
            self.engine.create_execution_context() for _ in range(self.pool_size)
        ]
        self.streams = [torch.cuda.Stream() for _ in range(self.pool_size)]
        # 连接池：空闲槽位队列（idx），借还保证同一 context 同时只被一个线程用
        self._slot_pool: "queue.Queue[int]" = queue.Queue()
        for _idx in range(self.pool_size):
            self._slot_pool.put(_idx)

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    # 兼容旧代码里可能访问 self.context（如预热或调试）：返回第一个 context
    @property
    def context(self):
        return self.contexts[0]

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """同步推理（输入 numpy → 输出 numpy），从连接池借一个 context/stream。"""
        # 借一个空闲槽（阻塞直到有空闲，保证 context 独占）
        idx = self._slot_pool.get()
        try:
            ctx = self.contexts[idx]
            stream = self.streams[idx]

            # 让本次调用的 H2D/kernel/D2H 全部走这个 stream
            with torch.cuda.stream(stream):
                # 输入
                d_inputs = {}
                for name in self.input_names:
                    data = inputs[name]
                    ctx.set_input_shape(name, data.shape)
                    t = torch.from_numpy(data).cuda(non_blocking=True).contiguous()
                    d_inputs[name] = t
                    ctx.set_tensor_address(name, t.data_ptr())

                # 输出（按 context 推断的真实 shape 分配）
                # 动态维度兜底：用输入张量出现过的最大维度值，保证覆盖 encoder seq 维
                # （alphas/cif_peak 的动态维 = enc_len+1，长音频可达 500+，硬编码 300 会
                #  导致输出 buffer 不足 → engine 越界写 → CUDA illegal memory access）
                _batch = list(inputs.values())[0].shape[0]
                _max_dim = max((max(v.shape) for v in inputs.values()), default=512)
                _fallback = max(_max_dim + 1, 512)  # +1 覆盖 tail 帧，下限 512
                d_outputs = {}
                for name in self.output_names:
                    shape = list(ctx.get_tensor_shape(name))
                    for i, s in enumerate(shape):
                        if s <= 0:
                            shape[i] = _batch if i == 0 else _fallback
                    t = torch.zeros(shape, dtype=torch.float32, device="cuda")
                    d_outputs[name] = t
                    ctx.set_tensor_address(name, t.data_ptr())

                ctx.execute_async_v3(stream_handle=stream.cuda_stream)

            # 只等这个 stream 完成，不影响其他 stream
            stream.synchronize()

            results = {}
            for name, t in d_outputs.items():
                actual_shape = tuple(ctx.get_tensor_shape(name))
                if all(s > 0 for s in actual_shape):
                    slices = tuple(slice(0, s) for s in actual_shape)
                    results[name] = t[slices].cpu().numpy()
                else:
                    results[name] = t.cpu().numpy()
            return results
        finally:
            # 归还槽位（异常时也归还，避免池耗尽死锁）
            self._slot_pool.put(idx)


# ============================================================
# 4 段串联推理引擎
# ============================================================
class TRTEngine:
    """
    SeACo-Paraformer TRT 4 段串联推理引擎。

    内部维护 4 个 _TRTInferencer，对外暴露 ASREngine 兼容接口。
    """

    def __init__(self):
        self._encoder: _TRTInferencer | None = None
        self._cif: _TRTInferencer | None = None
        self._decoder: _TRTInferencer | None = None
        self._bias_encoder: _TRTInferencer | None = None
        self._timestamp: _TRTInferencer | None = None  # 第 5 段（可选，字级时间戳）
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def has_bias_encoder(self) -> bool:
        return self._bias_encoder is not None

    @property
    def has_timestamp(self) -> bool:
        return self._timestamp is not None

    def load(self, encoder_path: str, cif_path: str, decoder_path: str,
             bias_encoder_path: str | None = None,
             timestamp_path: str | None = None):
        """加载 4 段主 engine + 可选第 5 段 timestamp engine。"""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT 未安装")
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch 未安装（TRT 推理需要 torch 管理 GPU 内存）")

        for label, path in [
            ("encoder", encoder_path),
            ("cif", cif_path),
            ("decoder", decoder_path),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} engine 不存在: {path}")

        pool_size = max(1, settings.GPU_STREAM_POOL_SIZE)
        logger.info(f"TRT engines 加载中（stream 池大小={pool_size}）:")

        logger.info(f"  encoder: {encoder_path}")
        self._encoder = _TRTInferencer(encoder_path, pool_size=pool_size)

        logger.info(f"  cif:     {cif_path}")
        self._cif = _TRTInferencer(cif_path, pool_size=pool_size)

        logger.info(f"  decoder: {decoder_path}")
        self._decoder = _TRTInferencer(decoder_path, pool_size=pool_size)

        if bias_encoder_path and os.path.exists(bias_encoder_path):
            logger.info(f"  bias_encoder: {bias_encoder_path}")
            # bias_encoder 固定单 context（不随 GPU_STREAM_POOL_SIZE 放大）：
            #   热词编码仅在「客户端传热词」或「默认词表热更新」时调用，属低频操作，
            #   非主链路每 chunk 必经；且编码结果可缓存复用。多 context 对它无并发收益，
            #   反而白占显存，故 pool_size=1 足够。
            self._bias_encoder = _TRTInferencer(bias_encoder_path, pool_size=1)
        else:
            logger.info("  bias_encoder: 未加载（热词功能不可用）")

        # 第 5 段 timestamp engine（ENABLE_WORD_TIMESTAMP 启用时才传入路径）
        if timestamp_path and os.path.exists(timestamp_path):
            logger.info(f"  timestamp: {timestamp_path}（字级时间戳启用）")
            self._timestamp = _TRTInferencer(timestamp_path, pool_size=pool_size)
        else:
            logger.info("  timestamp: 未加载（字级时间戳关闭，words 为空）")

        self._loaded = True

    # --------------------------------------------------------
    # 热词编码：tokens (H, L) → bias_embed (1, H, 512)
    # --------------------------------------------------------
    def encode_hotwords(self, hotword_token_ids: np.ndarray) -> np.ndarray | None:
        """
        bias_encoder 推理 + 按热词长度切片得到 bias_embed。

        hotword_token_ids: (H, L) 已 pad（pad=0）。最后一项必须是 [sos]=[1] 占位。
        返回: (1, H, 512) bias_embed（每个热词最后有效时间步的 LSTM 输出）。
        """
        if self._bias_encoder is None:
            return None

        # bias_encoder 输入：hotword (H, L) → 输出 hw_embed (L, H, 512)
        out = self._bias_encoder.infer({"hotword": hotword_token_ids.astype(np.int64)})
        hw_embed = out["hw_embed"]  # (L, H, 512)

        # 取每个热词最后有效时间步
        hotword_lengths = (hotword_token_ids != 0).sum(axis=1) - 1  # (H,)
        # 最后一项是 [sos] 哨兵，固定取 0
        hotword_lengths[-1] = 0
        hotword_lengths = np.clip(hotword_lengths, 0, None)

        hw_embed_t = hw_embed.transpose(1, 0, 2)  # (H, L, 512)
        bias_list = [hw_embed_t[i, hotword_lengths[i], :] for i in range(hw_embed_t.shape[0])]
        bias_embed = np.stack(bias_list, axis=0)[np.newaxis, :, :].astype(np.float32)
        return bias_embed

    # --------------------------------------------------------
    # 主推理接口：兼容 ASREngine.infer_batch_raw
    # --------------------------------------------------------
    def infer_batch_raw(
        self,
        padded_feats: np.ndarray,
        lengths: np.ndarray,
        bias_embeddings: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """
        4 段串联推理。

        Args:
            padded_feats: (batch, bucket_seq_len, 560) float32（已 pad 到桶边界）
            lengths: (batch,) int32 桶长度
            bias_embeddings: (1, H, 512) float32，无热词时传 None（内部用全零 1×1×512）

        Returns:
            (logits, ts_data) 元组列表，每 batch 一项：
                logits: (token_num, vocab_size)，token_num 由 CIF 输出动态决定
                ts_data: {"us_alphas", "us_cif_peak", "num_tokens"} 或 None
                    用于主流程算字级时间戳（对齐 FunASR ts_prediction）；
                    engine 无 timestamp 输出时为 None（字级时间戳降级）
        """
        if not self._loaded:
            raise RuntimeError("TRT engine 未加载")

        batch_size = padded_feats.shape[0]

        # 裁剪到 batch 内最大真实帧数（engine dynamic shape，无需 pad 到桶边界），
        # 但不低于 profile 下界（最小桶），避免 setInputShape 越界。
        real_max = max(1, int(max(int(L) for L in lengths)))
        min_seq = min(settings.BUCKET_SEQ_LENS)
        real_max = max(real_max, min_seq)
        if real_max < padded_feats.shape[1]:
            padded_feats = padded_feats[:, :real_max, :]
        seq_len = padded_feats.shape[1]

        # ---- 1. Encoder: speech → encoder_out ----
        enc_inputs = {"speech": padded_feats.astype(np.float32)}
        if "speech_lengths" in self._encoder.input_names:
            enc_inputs["speech_lengths"] = lengths.astype(np.int64)
        enc_out = self._encoder.infer(enc_inputs)
        encoder_out = enc_out["encoder_out"]  # (batch, seq_len, 512)
        # encoder_out_lens = 真实有效帧数（用于 decoder cross-attention 的 memory mask，
        # 排除 encoder padding 帧污染；encoder engine 不输出 lens，用传入的真实 lengths）。
        encoder_out_lens = enc_out.get(
            "encoder_out_lens",
            lengths.astype(np.int64),
        )

        # ---- 2. CIF: encoder_out + mask → acoustic_embeds, token_num ----
        # 推理时 batch=1 走全 1 mask（与 test_trt_pipeline.py 一致）
        # 多 batch 时按各自有效长度构造 mask
        mask = np.zeros((batch_size, 1, seq_len), dtype=np.float32)
        for i, L in enumerate(lengths):
            mask[i, 0, :int(L)] = 1.0
        cif_out = self._cif.infer({"encoder_out": encoder_out, "mask": mask})
        acoustic_embeds = cif_out["acoustic_embeds"]  # (batch, max_token, 512)
        token_num_arr = cif_out["token_num"].flatten()  # (batch,)

        # 第 5 段 timestamp engine（可选，ENABLE_WORD_TIMESTAMP 启用时才加载）：
        #   输入 encoder_out + mask + token_num → us_alphas/us_cif_peak
        # 未启用（self._timestamp is None）时不跑，主链路吞吐不受影响。
        us_alphas_arr = None
        us_cif_peak_arr = None
        if self._timestamp is not None:
            ts_out = self._timestamp.infer({
                "encoder_out": encoder_out,
                "mask": mask,
                "token_num": np.round(token_num_arr).astype(np.float32),
            })
            us_alphas_arr = ts_out.get("us_alphas")
            us_cif_peak_arr = ts_out.get("us_cif_peak")

        # ---- 3. Decoder: 逐 batch 切到实际 token 数后传入 ----
        # 由于 token_num 每条不同，需要按最大 token_num 重新 pad
        token_nums = np.round(token_num_arr).astype(np.int64)
        max_tok = int(token_nums.max())
        if max_tok == 0:
            # 全零 token：返回空 logits 占位（ts_data 也给 None）
            return [(np.zeros((0, 8404), dtype=np.float32), None)
                    for _ in range(batch_size)]

        # 截断 acoustic_embeds 到 max_tok
        acoustic_trimmed = acoustic_embeds[:, :max_tok, :].astype(np.float32)

        # bias_embed: 无热词用全零 1×1×512
        if bias_embeddings is None:
            bias_embed_input = np.zeros((1, 1, 512), dtype=np.float32)
        else:
            bias_embed_input = bias_embeddings.astype(np.float32)

        # decoder 的 acoustic_embeds/encoder_out/bias_embed 共享 batch 维（engine 同名维度），
        # bias_embed 原始 batch=1，需 tile 到 batch_size 保持三者一致，否则
        # TRT 报 "Dimensions with name batch must be equal"（batch>1 时）。
        if bias_embed_input.shape[0] != batch_size:
            bias_embed_input = np.tile(bias_embed_input, (batch_size, 1, 1)).astype(np.float32)

        dec_inputs = {}
        if "acoustic_embeds" in self._decoder.input_names:
            dec_inputs["acoustic_embeds"] = acoustic_trimmed
        if "token_num" in self._decoder.input_names:
            dec_inputs["token_num"] = token_nums
        if "encoder_out" in self._decoder.input_names:
            dec_inputs["encoder_out"] = encoder_out.astype(np.float32)
        if "encoder_out_lens" in self._decoder.input_names:
            dec_inputs["encoder_out_lens"] = encoder_out_lens.astype(np.int64)
        if "bias_embed" in self._decoder.input_names:
            dec_inputs["bias_embed"] = bias_embed_input

        dec_out = self._decoder.infer(dec_inputs)
        logits = dec_out["logits"]  # (batch, max_tok, vocab) 或 list

        # ---- 4. 按各自 token_num 切片返回 (logits, ts_data) 元组 ----
        # ts_data: {"us_alphas", "us_cif_peak", "num_tokens"}，供主流程算字级时间戳；
        #          engine 无 timestamp 输出时为 None（字级时间戳降级）。
        # us_alphas/us_cif_peak 长度 = enc_len × upsample_times，按真实帧数截取去 padding。
        results = []
        has_ts = us_alphas_arr is not None and us_cif_peak_arr is not None
        up_ratio = (us_alphas_arr.shape[1] // seq_len) if has_ts and seq_len > 0 else 1
        for i in range(batch_size):
            n = int(token_nums[i])
            logits_i = logits[i, :n, :].copy()
            if has_ts:
                real_up = int(lengths[i]) * up_ratio  # 真实上采样帧数（去 padding）
                real_up = min(real_up, us_alphas_arr.shape[1])
                ts_data = {
                    "us_alphas": us_alphas_arr[i, :real_up].copy(),
                    "us_cif_peak": us_cif_peak_arr[i, :real_up].copy(),
                    "num_tokens": n,
                }
            else:
                ts_data = None
            results.append((logits_i, ts_data))
        return results

    # --------------------------------------------------------
    # 预热
    # --------------------------------------------------------
    def warmup(self, bucket_seq_lens: list[int], batch_sizes: list[int]):
        """对每个 (bucket_seq_len, batch_size) 组合执行一次推理。

        含热词维度预热：用 OPT/MAX 热词数各跑一次，避免首个带热词请求
        因 bias_embed 新 shape 触发 TRT 现场优化导致首请求延迟突增。
        """
        if not self._loaded:
            return

        logger.info("TRT engine 预热中（4 段串联）...")
        feat_dim = 560
        count = 0

        for seq_len in bucket_seq_lens:
            for batch in batch_sizes:
                try:
                    dummy_feats = np.random.randn(batch, seq_len, feat_dim).astype(np.float32)
                    dummy_lengths = np.full(batch, seq_len, dtype=np.int32)
                    self.infer_batch_raw(dummy_feats, dummy_lengths, None)
                    count += 1
                except Exception as e:
                    logger.warning(f"  TRT 预热失败 batch={batch}, seq={seq_len}: {e}")

        # 热词维度预热（bias_encoder + decoder bias_embed 路径）
        if self.has_bias_encoder:
            try:
                from src.config import settings as _settings
                seq_len = bucket_seq_lens[len(bucket_seq_lens) // 2]  # 主力桶
                # 用 OPT 与 MAX 热词数各预热一次（含 [sos] 哨兵 +1）
                hw_nums = sorted({
                    _settings.OPT_HOTWORD_NUM,
                    _settings.MAX_HOTWORD_NUM,
                })
                hw_count = 0
                for num in hw_nums:
                    # 构造 (num+1, L) 热词 token（末行 [sos] 哨兵）
                    hw_len = min(4, _settings.MAX_HOTWORD_LEN)
                    hotword_ids = np.ones((num + 1, hw_len), dtype=np.int64)
                    bias_embed = self.encode_hotwords(hotword_ids)
                    if bias_embed is None:
                        break
                    dummy_feats = np.random.randn(1, seq_len, feat_dim).astype(np.float32)
                    dummy_lengths = np.full(1, seq_len, dtype=np.int32)
                    self.infer_batch_raw(dummy_feats, dummy_lengths, bias_embed)
                    hw_count += 1
                count += hw_count
                logger.info(f"  热词维度预热完成（{hw_count} 个热词 shape: {hw_nums}）")
            except Exception as e:
                logger.warning(f"  热词维度预热失败（不影响功能）: {e}")

        logger.info(f"TRT engine 预热完成（{count} 个 shape）")


# 全局单例
trt_engine = TRTEngine()
