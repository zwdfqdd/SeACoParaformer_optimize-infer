"""
GPU Scheduler — GPU 统一调度器

功能：
- Bucket 管理：根据 chunk 时长 5s/8s/12s/15s 进入不同 bucket
- Dynamic Batch 组装：BATCH_TIMEOUT 窗口内收集请求，达到 batch_size 或 MAX_BATCH_DURATION 立即触发
- 统一 GPU 提交：所有推理由 scheduler 统一提交，避免 stream 冲突
- OOM Fallback：CUDA OOM 时自动减小 batch 重试，最终 CPU fallback
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.asr_engine import asr_engine
from src.config import settings
from src.errors import ASRException, ErrorCode
from src.logger import logger


# Bucket 边界（毫秒）
BUCKET_BOUNDARIES = [5000, 8000, 12000, 15000]


@dataclass
class InferRequest:
    """推理请求。"""
    features: np.ndarray  # shape: (time, feat_dim)
    duration_ms: int
    bias_embeddings: np.ndarray | None = None  # 热词 bias embeddings
    future: asyncio.Future = field(default=None)

    @property
    def length(self) -> int:
        return self.features.shape[0]


def _get_bucket_index(duration_ms: int) -> int:
    """根据时长确定 bucket 索引。"""
    for i, boundary in enumerate(BUCKET_BOUNDARIES):
        if duration_ms <= boundary:
            return i
    return len(BUCKET_BOUNDARIES) - 1


class GPUScheduler:
    """
    GPU 统一调度器。

    所有 GPU inference 统一由 scheduler 提交，避免多线程直接调用 CUDA
    造成 stream 冲突、context 切换、GPU utilization 降低。
    """

    def __init__(self):
        self._buckets: list[list[InferRequest]] = [
            [] for _ in BUCKET_BOUNDARIES
        ]
        self._lock = asyncio.Lock()
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self):
        """启动调度循环。"""
        self._running = True
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info(
            f"GPU Scheduler 启动: batch_size={settings.BATCH}, "
            f"timeout={settings.BATCH_TIMEOUT}ms, "
            f"max_batch_duration={settings.MAX_BATCH_DURATION}s"
        )

    async def stop(self):
        """停止调度循环。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("GPU Scheduler 已停止")

    async def submit(self, features: np.ndarray, duration_ms: int, bias_embeddings: np.ndarray | None = None) -> np.ndarray:
        """
        提交推理请求，等待结果返回。
        """
        loop = asyncio.get_event_loop()
        request = InferRequest(
            features=features,
            duration_ms=duration_ms,
            bias_embeddings=bias_embeddings,
            future=loop.create_future(),
        )

        bucket_idx = _get_bucket_index(duration_ms)

        async with self._lock:
            self._buckets[bucket_idx].append(request)

        return await request.future

    async def _schedule_loop(self):
        """调度主循环：定期检查 bucket 并触发推理。"""
        while self._running:
            timeout_sec = settings.BATCH_TIMEOUT / 1000.0  # 每次循环重新读取，支持热更新
            await asyncio.sleep(timeout_sec)

            async with self._lock:
                for bucket_idx in range(len(self._buckets)):
                    bucket = self._buckets[bucket_idx]
                    if not bucket:
                        continue

                    # 持续取 batch 直到 bucket 清空，避免高并发时积压
                    while bucket:
                        batch = self._collect_batch(bucket)
                        if not batch:
                            break
                        self._buckets[bucket_idx] = bucket[len(batch):]
                        bucket = self._buckets[bucket_idx]
                        asyncio.create_task(self._execute_batch(batch))

    def _collect_batch(self, bucket: list[InferRequest]) -> list[InferRequest]:
        """
        从 bucket 中收集一个 batch。

        触发条件：
        - 达到 batch_size
        - 总音频时长达到 MAX_BATCH_DURATION
        - 超时（由调度循环保证）
        """
        if not bucket:
            return []

        batch: list[InferRequest] = []
        total_duration_ms = 0
        max_batch_duration_ms = settings.MAX_BATCH_DURATION * 1000

        for req in bucket:
            if len(batch) >= settings.BATCH:
                break
            if total_duration_ms + req.duration_ms > max_batch_duration_ms:
                break
            batch.append(req)
            total_duration_ms += req.duration_ms

        # 如果没有达到 batch_size 但有请求（超时触发），也返回
        if not batch and bucket:
            batch = [bucket[0]]

        return batch

    async def _execute_batch(self, batch: list[InferRequest]):
        """执行一个 batch 的推理。"""
        features_list = [req.features for req in batch]
        lengths = [req.length for req in batch]
        # 取第一个请求的 bias（同一请求的所有 chunk 共享相同 hotwords）
        bias_embeddings = batch[0].bias_embeddings

        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                asr_engine.infer_batch,
                features_list,
                lengths,
                bias_embeddings,
            )

            # 分发结果
            for req, result in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(result)

        except ASRException as e:
            if "out of memory" in str(e).lower():
                # OOM Fallback：减小 batch 重试
                await self._oom_fallback(batch)
            else:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)
        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(
                        ASRException(ErrorCode.ASR_INFER_FAILED, str(e))
                    )

    async def _oom_fallback(self, batch: list[InferRequest]):
        """
        OOM 恢复策略：
        1. 减小 batch 重试
        2. 最终 CPU fallback
        """
        logger.warning(f"GPU OOM，尝试减小 batch (原 batch_size={len(batch)})")

        # 逐条推理
        for req in batch:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    asr_engine.infer_single,
                    req.features,
                )
                if not req.future.done():
                    req.future.set_result(result)
            except Exception as e:
                logger.error(f"OOM fallback 单条推理也失败: {e}")
                if not req.future.done():
                    req.future.set_exception(
                        ASRException(ErrorCode.ASR_INFER_FAILED, f"推理失败(OOM fallback): {e}")
                    )


# 全局单例
gpu_scheduler = GPUScheduler()
