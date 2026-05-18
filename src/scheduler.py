"""
GPU Scheduler — 固定 shape bucket + batch_timeout 调度

- VAD 后音频经 audio_segment 合并/切分，Scheduler 将特征 pad 到桶边界
- 桶边界：2s/4s/8s（LFR 帧数 34/67/134）
- batch_timeout 窗口内收集同桶 chunk
- 按合法 batch size（1,2,4,8,12）推理
- 达到合法 batch 立即触发，超时按实际数量向上取最近合法 batch（pad dummy）
- OOM Fallback：减半 batch 重试 → 逐条推理 → 返回错误
"""

import asyncio
import concurrent.futures
from dataclasses import dataclass, field

import numpy as np

from src.asr_engine import asr_engine
from src.config import settings
from src.errors import ASRException, ErrorCode
from src.logger import logger

# 桶边界（LFR 帧数）：2s→34, 4s→67, 8s→134
BUCKET_SEQ_LENS = [34, 67, 134]

# 合法 batch sizes
VALID_BATCH_SIZES = [1, 2, 4, 8, 12]

# GPU 专用线程池（单线程串行推理）
_gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


def get_bucket_idx(seq_len: int) -> int:
    """将 seq_len 归入最近的桶。"""
    for i, b in enumerate(BUCKET_SEQ_LENS):
        if seq_len <= b:
            return i
    return len(BUCKET_SEQ_LENS) - 1


def get_trigger_batch_size(n: int) -> int:
    """取 <= n 的最大合法 batch size（用于立即触发判断）。"""
    result = 1
    for b in VALID_BATCH_SIZES:
        if b <= n:
            result = b
    return result


def get_pad_batch_size(n: int) -> int:
    """取 >= n 的最近合法 batch size（用于超时触发时 pad）。"""
    for b in VALID_BATCH_SIZES:
        if b >= n:
            return b
    return VALID_BATCH_SIZES[-1]


@dataclass
class InferRequest:
    """推理请求。"""
    features: np.ndarray  # 已 pad 到桶边界的特征 (bucket_seq_len, 560)
    length: int  # 有效帧数
    bucket_idx: int
    bias_embeddings: np.ndarray | None = None
    future: asyncio.Future = field(default=None)


class GPUScheduler:
    """
    固定 shape bucket + batch_timeout 调度器。

    - 每个桶独立队列
    - 达到合法 batch size 立即触发推理
    - batch_timeout 超时后按实际数量推理（pad 到合法 batch）
    """

    def __init__(self):
        self._buckets: list[list[InferRequest]] = [[] for _ in BUCKET_SEQ_LENS]
        self._lock = asyncio.Lock()
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info(
            f"GPU Scheduler 启动: batch_sizes={VALID_BATCH_SIZES}, "
            f"timeout={settings.BATCH_TIMEOUT}ms, "
            f"buckets={BUCKET_SEQ_LENS}"
        )

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(
        self,
        features: np.ndarray,
        bias_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """提交 chunk 推理请求。"""
        seq_len = features.shape[0]
        bucket_idx = get_bucket_idx(seq_len)
        target_len = BUCKET_SEQ_LENS[bucket_idx]

        if settings.VERBOSE:
            logger.debug(f"[Scheduler] submit: seq_len={seq_len}, bucket={bucket_idx}({target_len}帧)")

        # pad 到桶边界
        if seq_len < target_len:
            feat_dim = features.shape[1]
            padded = np.zeros((target_len, feat_dim), dtype=np.float32)
            padded[:seq_len] = features
        else:
            padded = features[:target_len]

        loop = asyncio.get_event_loop()
        request = InferRequest(
            features=padded,
            length=min(seq_len, target_len),
            bucket_idx=bucket_idx,
            bias_embeddings=bias_embeddings,
            future=loop.create_future(),
        )

        async with self._lock:
            self._buckets[bucket_idx].append(request)
            # 立即触发：达到合法 batch size
            bucket = self._buckets[bucket_idx]
            trigger_size = get_trigger_batch_size(len(bucket))
            if trigger_size >= VALID_BATCH_SIZES[1] and len(bucket) >= trigger_size:
                batch = bucket[:trigger_size]
                self._buckets[bucket_idx] = bucket[trigger_size:]
                asyncio.create_task(self._execute_batch(batch, bucket_idx))

        return await request.future

    async def _schedule_loop(self):
        """超时调度：batch_timeout 到期后按实际数量触发。"""
        while self._running:
            timeout_sec = settings.BATCH_TIMEOUT / 1000.0
            await asyncio.sleep(timeout_sec)

            if not any(self._buckets):
                continue

            async with self._lock:
                for bucket_idx in range(len(self._buckets)):
                    bucket = self._buckets[bucket_idx]
                    if not bucket:
                        continue
                    # 超时触发：取所有等待中的 chunk
                    batch = list(bucket)
                    self._buckets[bucket_idx] = []
                    asyncio.create_task(self._execute_batch(batch, bucket_idx))

    async def _execute_batch(self, batch: list[InferRequest], bucket_idx: int):
        """执行推理：pad 到合法 batch size，推理后丢弃 padding 结果。"""
        import time as _time

        actual_count = len(batch)
        target_seq_len = BUCKET_SEQ_LENS[bucket_idx]
        feat_dim = batch[0].features.shape[1]
        bias_embeddings = batch[0].bias_embeddings

        # pad 到合法 batch size
        pad_batch_size = get_pad_batch_size(actual_count)
        padded_feats = np.zeros((pad_batch_size, target_seq_len, feat_dim), dtype=np.float32)
        lengths = np.zeros(pad_batch_size, dtype=np.int32)
        actual_lengths = []

        for i, req in enumerate(batch):
            padded_feats[i] = req.features
            lengths[i] = target_seq_len  # 传桶长度（保证 attention mask 匹配）
            actual_lengths.append(req.length)

        # dummy padding（复制最后一条的数据）
        for i in range(actual_count, pad_batch_size):
            padded_feats[i] = batch[-1].features
            lengths[i] = target_seq_len

        try:
            t0 = _time.perf_counter()
            results = await asyncio.get_event_loop().run_in_executor(
                _gpu_executor,
                asr_engine.infer_batch_raw,
                padded_feats,
                lengths,
                bias_embeddings,
            )
            infer_ms = (_time.perf_counter() - t0) * 1000

            if settings.VERBOSE:
                logger.debug(
                    f"[Stage3] bucket={bucket_idx}({target_seq_len}帧), "
                    f"batch={pad_batch_size}(实际{actual_count}), "
                    f"推理={infer_ms:.1f}ms"
                )

            # 只返回实际请求的结果（按实际有效长度截断）
            for i, req in enumerate(batch):
                if not req.future.done():
                    logits = results[i]
                    req.future.set_result(logits[:actual_lengths[i]])

        except Exception as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                # OOM Fallback：减半 batch 重试 → 逐条推理 → 返回错误
                logger.warning(f"GPU OOM (batch={pad_batch_size})，尝试减半 batch 重试")
                await self._oom_fallback(batch, actual_lengths, bucket_idx, bias_embeddings)
            else:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(
                            ASRException(ErrorCode.ASR_INFER_FAILED, f"推理失败: {e}")
                        )

    async def _oom_fallback(
        self, batch: list[InferRequest], actual_lengths: list[int],
        bucket_idx: int, bias_embeddings
    ):
        """
        OOM 恢复策略：
        1. 减半 batch size 重试
        2. 仍失败则逐条推理
        3. 仍失败则返回 ASR_INFER_FAILED 错误
        """
        target_seq_len = BUCKET_SEQ_LENS[bucket_idx]
        feat_dim = batch[0].features.shape[1]
        actual_count = len(batch)

        # Step 1: 减半 batch 重试
        half_size = max(1, actual_count // 2)
        logger.info(f"OOM Fallback Step1: 减半 batch {actual_count} → {half_size}")

        for start in range(0, actual_count, half_size):
            sub_batch = batch[start:start + half_size]
            sub_lengths = actual_lengths[start:start + half_size]
            sub_count = len(sub_batch)
            pad_size = get_pad_batch_size(sub_count)

            padded_feats = np.zeros((pad_size, target_seq_len, feat_dim), dtype=np.float32)
            lengths = np.zeros(pad_size, dtype=np.int32)
            for i, req in enumerate(sub_batch):
                padded_feats[i] = req.features
                lengths[i] = target_seq_len  # 传桶长度
            for i in range(sub_count, pad_size):
                padded_feats[i] = sub_batch[-1].features
                lengths[i] = target_seq_len

            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    _gpu_executor,
                    asr_engine.infer_batch_raw,
                    padded_feats,
                    lengths,
                    bias_embeddings,
                )
                for i, req in enumerate(sub_batch):
                    if not req.future.done():
                        req.future.set_result(results[i][:sub_lengths[i]])
            except Exception as e1:
                error_msg = str(e1).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    # Step 2: 逐条推理
                    logger.warning(f"OOM Fallback Step2: 减半仍 OOM，降级为逐条推理")
                    for i, req in enumerate(sub_batch):
                        try:
                            single_feats = req.features[np.newaxis, :, :]
                            single_lengths = np.array([target_seq_len], dtype=np.int32)  # 传桶长度
                            result = await asyncio.get_event_loop().run_in_executor(
                                _gpu_executor,
                                asr_engine.infer_batch_raw,
                                single_feats,
                                single_lengths,
                                bias_embeddings,
                            )
                            if not req.future.done():
                                req.future.set_result(result[0][:sub_lengths[i]])
                        except Exception as e2:
                            # Step 3: 返回错误
                            logger.error(f"OOM Fallback Step3: 逐条推理也失败: {e2}")
                            if not req.future.done():
                                req.future.set_exception(
                                    ASRException(ErrorCode.ASR_INFER_FAILED, f"推理失败(OOM fallback): {e2}")
                                )
                else:
                    # 非 OOM 错误，直接返回失败
                    for req in sub_batch:
                        if not req.future.done():
                            req.future.set_exception(
                                ASRException(ErrorCode.ASR_INFER_FAILED, f"推理失败: {e1}")
                            )


# 全局单例
gpu_scheduler = GPUScheduler()
