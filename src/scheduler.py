"""
GPU Scheduler — 工业标准 dynamic batching（Triton/TF-Serving 模式）

设计：
- VAD 后音频经 audio_segment 合并/切分，Scheduler 将特征 pad 到桶边界
- 桶边界：2s/4s/8s（LFR 帧数 34/67/134）
- 每个 (bucket, bias) 独立 group 队列

触发条件（OR 逻辑，先到者优先）：
    1. 满 batch：group.size >= MAX_BATCH_SIZE → 立即触发（不等）
    2. 超时：now - group[0].enqueue_time >= BATCH_TIMEOUT
       → 按当前 group 大小 pad 到最近合法 batch 触发

关键点：
- 超时按"最早入队 chunk"计算，保证严格延迟上限（不是全局定时抖动）
- 满 batch 立即触发（再等也不能更大）
- 触发后剩余 chunk（>MAX_BATCH_SIZE 部分）重新入队，enqueue_time 重置

OOM Fallback：减半 batch 重试 → 逐条推理 → 返回错误
"""

import asyncio
import concurrent.futures
import time as _time
from dataclasses import dataclass, field

import numpy as np

from src.asr_engine import asr_engine
from src.config import settings
from src.errors import ASRException, ErrorCode
from src.logger import logger

# 桶边界 / 合法 batch：统一来自 config（单一数据源，与 TRT profile 一致）
BUCKET_SEQ_LENS = settings.BUCKET_SEQ_LENS
VALID_BATCH_SIZES = settings.VALID_BATCH_SIZES
MAX_BATCH_SIZE = max(VALID_BATCH_SIZES)  # 满 batch 触发阈值（工业标准 max_batch_size）

# Uniform Chunking：audio_segment 已按 UNIFORM_CHUNK_MS 均匀切段（尾块合并到前段），
# chunk 帧数分布集中在 opt(67) 附近，最长不超过 opt + MIN_TAIL_MS 对应帧数 (~84 帧)
# scheduler 存储时 pad 到 profile max（134，形状统一便于 batch stack），
# TRT engine 会按 batch max(lengths) 二次裁剪，只算真实有效范围（无 GPU 浪费）
# 消除桶分组 → group_key 与 bucket 无关，跨请求大合并 → avg_batch 显著提升

# GPU 专用线程池：与 TRT stream 池大小对齐，允许多个 batch 并发提交到不同 stream。
# 每个线程从 _TRTInferencer 的 round-robin 池中获取独立 (context, stream)，真正并行执行。
# 主推理和热词编码共用同一池，但热词编码是低频操作（词表 reload 或客户端热词提交），
# 不会与主推理形成持续争用。
_gpu_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=settings.GPU_STREAM_POOL_SIZE,
    thread_name_prefix="gpu",
)


def encode_hotwords_on_gpu(hotword_token_ids: np.ndarray) -> np.ndarray | None:
    """
    同步版热词 bias 编码，统一收口到 GPU 单线程池。

    供热词管理器（启动加载 / reload / 轮询线程等同步上下文）调用，
    与主推理共用同一 GPU 线程，避免 CUDA stream 冲突。
    阻塞等待结果（bias_encoder 推理为毫秒级）。
    """
    return _gpu_executor.submit(asr_engine.encode_hotwords, hotword_token_ids).result()


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
    enqueue_time: float = 0.0  # 入队时刻（time.monotonic），用于超时判定


class GPUScheduler:
    """
    固定 shape bucket + batch_timeout 调度器。

    - 每个桶独立队列
    - 达到合法 batch size 立即触发推理
    - batch_timeout 超时后按实际数量推理（pad 到合法 batch）
    """

    def __init__(self):
        # 分组键 = (bucket_idx, bias_key)，bias_key 用 id(bias) 区分热词身份：
        #   - 同一请求的多 chunk 共享同一 bias 对象 → 同组合并
        #   - 默认词表 route=A 复用同一缓存 bias 对象 → 跨请求可合并
        #   - 不同客户端热词 = 不同对象 → 不会错误合并（避免热词串扰）
        #   - 无热词 bias=None → key=0，全部合并
        # value = (requests 列表, bias_embeddings 引用)
        self._groups: dict[tuple[int, int], tuple[list[InferRequest], np.ndarray | None]] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._task: asyncio.Task | None = None
        # 性能统计（batch 填充率诊断）：累计触发批次数、实际 chunk 数、pad 后 slot 数
        # 填充率 = 实际 chunk 数 / pad slot 数（越接近 1 越好，越低说明 GPU 空跑 padding 越多）
        self._stat_batches = 0
        self._stat_actual = 0
        self._stat_padded = 0

    def stats(self) -> dict:
        """返回 batch 调度统计（供 /metrics 诊断填充率）。"""
        fill = (self._stat_actual / self._stat_padded) if self._stat_padded else 0.0
        avg_batch = (self._stat_actual / self._stat_batches) if self._stat_batches else 0.0
        return {
            "batches": self._stat_batches,
            "actual_chunks": self._stat_actual,
            "padded_slots": self._stat_padded,
            "fill_rate": round(fill, 4),
            "avg_actual_batch": round(avg_batch, 2),
        }

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

    async def encode_hotwords(self, hotword_token_ids: np.ndarray) -> np.ndarray | None:
        """
        热词 bias 编码（统一收口到 GPU 单线程池）。

        bias_encoder 是 TRT/CUDA 推理，必须与主推理共用同一 GPU 线程，
        避免多线程并发提交 CUDA 工作造成 stream 冲突（见 plan.md GPU Scheduler 设计）。
        """
        return await asyncio.get_event_loop().run_in_executor(
            _gpu_executor,
            asr_engine.encode_hotwords,
            hotword_token_ids,
        )

    async def submit(
        self,
        features: np.ndarray,
        bias_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        提交 chunk 推理请求（Uniform Chunking + Batch-Internal-Pad）。

        - submit 阶段不 pad，直接存原始 features（帧数多为 67~84，尾块合并后最长 ~84）
        - _execute_batch 阶段按 batch 内 max 帧数一次性 pad + stack
          → 只 pad 到 batch 内实际最大长度，不做无谓浪费
        - 上界保护：单 chunk 超过 profile max（134）时截断（防越界）
        """
        seq_len = features.shape[0]
        # 上界保护：单 chunk 帧数超过 TRT profile max 时截断
        max_seq = max(BUCKET_SEQ_LENS)
        if seq_len > max_seq:
            features = features[:max_seq]
            seq_len = max_seq

        # bucket_idx 保留元数据用途（_execute_batch 里根据 batch max 动态确定 target_seq_len）
        bucket_idx = len(BUCKET_SEQ_LENS) - 1

        if settings.VERBOSE:
            logger.debug(
                f"[Scheduler] submit: seq_len={seq_len}帧（batch 内按实际 max 合并 pad）"
            )

        loop = asyncio.get_event_loop()
        request = InferRequest(
            features=features,  # 原始形状（不 pad），batch 组装时统一处理
            length=seq_len,
            bucket_idx=bucket_idx,
            bias_embeddings=bias_embeddings,
            future=loop.create_future(),
            enqueue_time=_time.monotonic(),
        )

        # 分组键：只按 bias 身份隔离（避免热词串扰）
        # 不再按 bucket 分组，所有相同 bias 的 chunk 全部合并 → 显著提升 avg_batch
        bias_key = id(bias_embeddings) if bias_embeddings is not None else 0
        group_key = (0, bias_key)  # 桶维度固定为 0，保留元组结构避免其他改动

        async with self._lock:
            group = self._groups.setdefault(group_key, ([], bias_embeddings))[0]
            group.append(request)
            # 满 batch 立即触发（工业标准 max_batch_size）：
            # 达到 MAX_BATCH_SIZE 后再等也不会更大，立即触发；
            # 剩余 chunk（超过 MAX_BATCH_SIZE 部分）保留在 group，enqueue_time 保持原值，
            # 由 _schedule_loop 按最早 chunk 超时兜底触发下一批。
            if len(group) >= MAX_BATCH_SIZE:
                batch = group[:MAX_BATCH_SIZE]
                remaining = group[MAX_BATCH_SIZE:]
                if remaining:
                    self._groups[group_key] = (remaining, bias_embeddings)
                else:
                    del self._groups[group_key]
                asyncio.create_task(self._execute_batch(batch, bucket_idx, bias_embeddings))

        return await request.future

    async def _schedule_loop(self):
        """
        超时触发（工业标准 max_queue_delay）：
        按最早入队 chunk 的 enqueue_time 判定超时，保证单请求延迟严格 ≤ BATCH_TIMEOUT。

        高精度 tick（1ms）扫描所有 group，一旦最早 chunk 超时，立即按当前 group 数量
        pad 到最近合法 batch 触发。若 group 大小 ≥ MAX_BATCH_SIZE（罕见，通常在 submit
        阶段已触发），也在此兜底。
        """
        max_delay_sec = settings.BATCH_TIMEOUT / 1000.0
        while self._running:
            # 1ms 高精度 tick：远小于典型 BATCH_TIMEOUT（10-30ms），几乎无抖动
            await asyncio.sleep(0.001)

            if not self._groups:
                continue

            now = _time.monotonic()
            async with self._lock:
                # 遍历所有非空 group，判定是否超时
                for group_key, (group, bias) in list(self._groups.items()):
                    if not group:
                        continue
                    # 按最早入队 chunk 计时（严格延迟上限）
                    if now - group[0].enqueue_time < max_delay_sec:
                        continue
                    # Uniform Chunking：bucket_idx 从 chunk 自身取（submit 阶段已统一）
                    bucket_idx = group[0].bucket_idx
                    # 兜底：即使超时也不超过 MAX_BATCH_SIZE（保护 engine profile 上界）
                    take = min(len(group), MAX_BATCH_SIZE)
                    batch = group[:take]
                    remaining = group[take:]
                    if remaining:
                        # 剩余 chunk 保留在 group，enqueue_time 原样保留，下次继续判定
                        self._groups[group_key] = (remaining, bias)
                    else:
                        del self._groups[group_key]
                    asyncio.create_task(self._execute_batch(batch, bucket_idx, bias))

    async def _execute_batch(
        self,
        batch: list[InferRequest],
        bucket_idx: int,
        bias_embeddings: np.ndarray | None,
    ):
        """执行推理：batch 内 pad 到实际最大帧数 + pad 到合法 batch size。

        - target_seq_len = batch 内 chunk 的实际最大帧数（不再取桶固定值），
          尾块合并后 chunk 集中在 67 帧，仅少数含尾合并的 chunk 可达 ~84 帧
        - 长度上界受 profile max 约束（submit 阶段已截断）
        - 同一 batch 内所有 chunk 共享同一 bias（由分组键保证），无热词串扰。
        """
        actual_count = len(batch)
        # batch 内实际最大帧数（≥ min_seq 兜底，避免 profile 下越界）
        target_seq_len = max(req.length for req in batch)
        target_seq_len = max(target_seq_len, min(BUCKET_SEQ_LENS))
        feat_dim = batch[0].features.shape[1]

        # pad 到合法 batch size
        pad_batch_size = get_pad_batch_size(actual_count)
        # 填充率统计（诊断 GPU 是否空跑 padding）
        self._stat_batches += 1
        self._stat_actual += actual_count
        self._stat_padded += pad_batch_size
        padded_feats = np.zeros((pad_batch_size, target_seq_len, feat_dim), dtype=np.float32)
        lengths = np.zeros(pad_batch_size, dtype=np.int32)
        actual_lengths = []

        for i, req in enumerate(batch):
            L = req.length
            padded_feats[i, :L] = req.features[:L]
            lengths[i] = L  # 真实有效帧数（CIF mask 据此排除 padding 帧，避免多 fire token）
            actual_lengths.append(L)

        # dummy padding（复制最后一条的数据，长度对齐 target_seq_len）
        last_L = batch[-1].length
        for i in range(actual_count, pad_batch_size):
            padded_feats[i, :last_L] = batch[-1].features[:last_L]
            lengths[i] = last_L

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

            # engine 内部已按 token_num 截断（TRT 4 段架构）或返回完整 logits（ORT）
            # scheduler 不再做帧级截断
            for i, req in enumerate(batch):
                if not req.future.done():
                    req.future.set_result(results[i])

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
                lengths[i] = req.length  # 真实有效帧数
            for i in range(sub_count, pad_size):
                padded_feats[i] = sub_batch[-1].features
                lengths[i] = sub_batch[-1].length

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
                        req.future.set_result(results[i])
            except Exception as e1:
                error_msg = str(e1).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    # Step 2: 逐条推理
                    logger.warning(f"OOM Fallback Step2: 减半仍 OOM，降级为逐条推理")
                    for i, req in enumerate(sub_batch):
                        try:
                            single_feats = req.features[np.newaxis, :, :]
                            single_lengths = np.array([req.length], dtype=np.int32)  # 真实有效帧数
                            result = await asyncio.get_event_loop().run_in_executor(
                                _gpu_executor,
                                asr_engine.infer_batch_raw,
                                single_feats,
                                single_lengths,
                                bias_embeddings,
                            )
                            if not req.future.done():
                                req.future.set_result(result[0])
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
