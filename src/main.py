"""
SeACo-Paraformer FastAPI 服务入口

提供：
- POST /chinese_asr — 中文语音识别接口
- GET /health — 健康检查接口
"""

import asyncio
import base64
import io
import signal
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.asr_engine import asr_engine
from src.audio_segment import ChunkMeta, extract_chunk_audio, segment_to_chunks
from src.config import settings
from src.errors import ASRException, ErrorCode, ERROR_HTTP_STATUS
from src.feature_extractor import extract_features, load_cmvn
from src.hotword_manager import (
    hotword_manager,
    ValidationError as HotwordValidationError,
    VersionConflict as HotwordVersionConflict,
)
from src.logger import (
    generate_request_id,
    log_request,
    logger,
    request_id_var,
)
from src.scheduler import gpu_scheduler
from src.schemas import (
    ASRRequest,
    ASRResponse,
    ErrorResponse,
    HealthResponse,
    HotwordReloadRequest,
    HotwordReloadResponse,
    HotwordStatusResponse,
    SegmentDetail,
)
from src.tokenizer import tokenizer
from src.vad import vad_engine

# ============================================================
# CMVN 参数（服务启动时加载）
# ============================================================
_cmvn_mean: np.ndarray | None = None
_cmvn_istd: np.ndarray | None = None

# ============================================================
# Prometheus 指标
# ============================================================
asr_request_total = Counter(
    "asr_request_total",
    "ASR 请求总数",
    ["status"],
)
asr_inference_duration = Histogram(
    "asr_inference_duration_seconds",
    "ASR 推理耗时",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# 三级流水线分级耗时（定位瓶颈在哪一级：Stage1 CPU / Stage2 CPU / Stage3 GPU）
_STAGE_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
asr_stage_duration = Histogram(
    "asr_stage_duration_seconds",
    "各流水线阶段耗时",
    ["stage"],  # stage1_decode_vad_seg / stage2_feature / stage3_gpu
    buckets=_STAGE_BUCKETS,
)
gpu_memory_usage = None  # 延迟初始化（仅 GPU 模式）
scheduler_fill_rate = None  # batch 填充率
scheduler_avg_batch = None  # 平均实际 batch 大小


def _init_gpu_metrics():
    """初始化 GPU 内存 + 调度器填充率指标。"""
    global gpu_memory_usage, scheduler_fill_rate, scheduler_avg_batch
    try:
        from prometheus_client import Gauge
        gpu_memory_usage = Gauge(
            "gpu_memory_usage_bytes",
            "GPU 显存使用量（字节）",
        )
        scheduler_fill_rate = Gauge(
            "asr_batch_fill_rate",
            "GPU batch 填充率（实际chunk数/pad slot数，越接近1越好）",
        )
        scheduler_avg_batch = Gauge(
            "asr_avg_actual_batch",
            "平均每批实际 chunk 数",
        )
    except Exception:
        pass

# ============================================================
# 并发控制
# ============================================================
_concurrent_semaphore: asyncio.Semaphore | None = None

# 独立线程池：CPU 任务和 GPU 任务分离，避免互相阻塞
import concurrent.futures
_cpu_executor: concurrent.futures.ThreadPoolExecutor | None = None


# ============================================================
# 应用生命周期
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭逻辑。"""
    global _concurrent_semaphore
    global _cpu_executor

    # 启动：加载模型
    logger.info("服务启动中...")
    _load_resources()
    vad_engine.load()
    asr_engine.load()
    await gpu_scheduler.start()

    # 默认词表加载 + 预编码缓存（路径 A）/ 热更新轮询
    # bias 编码统一收口到 GPU 单线程池（避免 CUDA stream 冲突）
    from src.scheduler import encode_hotwords_on_gpu
    hotword_manager.set_encoders(tokenizer.encode, encode_hotwords_on_gpu)
    hotword_manager.load_initial()
    hotword_manager.start_polling()

    _concurrent_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
    # CPU 线程池（Stage1 VAD + Stage2 特征提取）
    import os
    _cpu_pool_size = settings.CPU_THREAD_POOL_SIZE or (os.cpu_count() or 4)
    _cpu_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=_cpu_pool_size,
        thread_name_prefix="cpu_worker",
    )
    _init_gpu_metrics()

    # 预热特征提取（torch/torchaudio 首次调用有 lazy init）
    logger.info("特征提取预热中...")
    dummy_pcm = np.zeros(16000, dtype=np.float32)  # 1秒静音
    extract_features(dummy_pcm, cmvn_mean=_cmvn_mean, cmvn_istd=_cmvn_istd)
    logger.info("特征提取预热完成")

    # 端到端预热（完整走一遍 VAD + 特征 + 推理，触发所有 lazy init）
    logger.info("端到端预热中...")
    try:
        dummy_segments = vad_engine.detect(dummy_pcm, 16000)
    except Exception:
        pass  # 静音可能无 VAD 段，忽略
    logger.info("端到端预热完成")

    logger.info(f"服务启动完成, CPU 线程池: {_cpu_executor._max_workers} workers")

    # 注册 SIGHUP 信号处理（配置热更新）
    try:
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGHUP, _reload_config)
    except (NotImplementedError, AttributeError):
        pass  # Windows 不支持 SIGHUP

    yield

    # 关闭：优雅停止
    logger.info("服务关闭中...")
    hotword_manager.stop_polling()
    await gpu_scheduler.stop()
    logger.info("服务已关闭")


def _reload_config():
    """SIGHUP 信号处理：热更新配置。"""
    import os
    new_timeout = os.getenv("BATCH_TIMEOUT")
    new_log_level = os.getenv("LOG_LEVEL")
    if new_timeout:
        settings.BATCH_TIMEOUT = int(new_timeout)
    if new_log_level:
        settings.LOG_LEVEL = new_log_level
    logger.info(f"配置热更新: BATCH_TIMEOUT={settings.BATCH_TIMEOUT}, LOG_LEVEL={settings.LOG_LEVEL}")


def _load_resources():
    """加载 CMVN 参数和 Tokenizer 词表。

    配置文件优先在 models/asr 下查找；若不存在则回退 models/asr/pt（ModelScope
    PT 包自带 am.mvn / tokens.json 在该子目录）。
    """
    global _cmvn_mean, _cmvn_istd
    import os

    config_dir = settings.get_asr_config_dir()  # models/asr
    pt_dir = os.path.join(config_dir, "pt")      # models/asr/pt（PT 包内配置回退）

    def _resolve(name: str) -> str | None:
        """在 config_dir 与 pt_dir 依次查找文件，返回首个存在路径。"""
        for d in (config_dir, pt_dir):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return None

    # 加载 CMVN
    cmvn_path = _resolve("am.mvn")
    if cmvn_path:
        try:
            _cmvn_mean, _cmvn_istd = load_cmvn(cmvn_path)
            logger.info(f"CMVN 加载成功: {cmvn_path}, shape={_cmvn_mean.shape}")
        except Exception as e:
            logger.warning(f"CMVN 加载失败: {e}，将跳过归一化")
    else:
        logger.warning(f"CMVN 文件不存在（{config_dir} / {pt_dir}），将跳过归一化")

    # 加载 Tokenizer
    vocab_path = _resolve("tokens.json") or _resolve("tokens.txt")
    if vocab_path:
        try:
            tokenizer.load(vocab_path)
            logger.info(f"Tokenizer 加载成功: {vocab_path}, vocab_size={tokenizer.vocab_size}")
        except Exception as e:
            logger.warning(f"Tokenizer 加载失败: {e}")
    else:
        logger.warning(f"词表文件不存在（{config_dir} / {pt_dir}），Tokenizer 未加载")


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="SeACo-Paraformer ASR Service",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# 异常处理器
# ============================================================
@app.exception_handler(ASRException)
async def asr_exception_handler(request: Request, exc: ASRException):
    """统一业务异常处理。

    /chinese_asr 接口的失败响应补充 text/detail 空值字段，与成功响应结构保持一致，
    便于客户端用统一结构解析（成功 text 有值/detail 有段，失败均为空）。
    其他接口（如 /hotwords/*）保持精简的 code/error/message 结构。
    """
    asr_request_total.labels(status="error").inc()
    content = {"code": int(exc.code)}
    if request.url.path == "/chinese_asr":
        # 失败响应保持与成功响应结构一致（text/article_url/detail 空值），
        # 便于客户端用统一结构解析。article_url 在错误分支无法从已解析请求获得，
        # 填 None（与"未传"语义等价）。
        content["text"] = ""
        content["article_url"] = None
        content["detail"] = {}
    content["error"] = exc.code.name
    content["message"] = exc.message
    return JSONResponse(
        status_code=exc.http_status,
        content=content,
    )


# ============================================================
# 路由
# ============================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口。"""
    return HealthResponse(
        status="ok",
        device=settings.get_device(),
        models_loaded=vad_engine.is_loaded and asr_engine.is_loaded,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus 指标接口。"""
    # scrape 时刷新 GPU 显存实际值（Gauge 标准做法）
    _update_gpu_memory_metric()
    _update_scheduler_metrics()
    return JSONResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


def _update_gpu_memory_metric():
    """刷新 GPU 显存使用量指标（当前进程占用的显存）。"""
    if gpu_memory_usage is None:
        return
    try:
        import torch
        if torch.cuda.is_available():
            # 已预留显存（reserved）最能反映进程实际占用
            gpu_memory_usage.set(float(torch.cuda.memory_reserved(0)))
    except Exception:
        pass


def _update_scheduler_metrics():
    """刷新 GPU 调度器 batch 填充率指标（诊断吞吐瓶颈）。"""
    if scheduler_fill_rate is None:
        return
    try:
        s = gpu_scheduler.stats()
        scheduler_fill_rate.set(s["fill_rate"])
        scheduler_avg_batch.set(s["avg_actual_batch"])
    except Exception:
        pass


# ============================================================
# 词表热更新接口（运行时不中断，多 worker 文件轮询收敛）
# ============================================================
@app.get("/hotwords/status", response_model=HotwordStatusResponse)
async def hotwords_status():
    """查看当前默认词表版本状态（巡检各 worker 收敛）。"""
    return HotwordStatusResponse(**hotword_manager.status())


@app.post("/hotwords/reload", response_model=HotwordReloadResponse)
async def hotwords_reload(req: HotwordReloadRequest):
    """
    重载默认词表（本 worker 立即生效，其他 worker 轮询收敛）。

    - words 与 reload_from_file 二选一
    - expected_version 乐观并发，与磁盘当前版本不符返回 409
    - 校验失败返回 400，保留旧词表
    """
    if not settings.HOTWORD_RELOAD_ENABLED:
        raise ASRException(ErrorCode.INPUT_PARAM_FAILED, "词表热更新未开启")

    loop = asyncio.get_event_loop()
    try:
        # reload 含文件 IO + bias 预编码，放线程池避免阻塞事件循环
        cache = await loop.run_in_executor(
            _cpu_executor,
            lambda: hotword_manager.reload(
                words=req.words,
                reload_from_file=req.reload_from_file,
                expected_version=req.expected_version,
            ),
        )
    except HotwordVersionConflict as e:
        raise ASRException(ErrorCode.HOTWORD_VERSION_CONFLICT, str(e))
    except HotwordValidationError as e:
        raise ASRException(ErrorCode.INPUT_PARAM_FAILED, f"词表校验失败：{e}")
    except Exception as e:
        raise ASRException(ErrorCode.INPUT_PARAM_FAILED, f"词表更新失败：{e}")

    return HotwordReloadResponse(
        code=0,
        version=cache.version,
        md5=cache.md5,
        count=cache.count,
        route=cache.route,
        message=f"词表更新成功，已切换至 version {cache.version}",
    )


@app.post("/hotwords/rollback", response_model=HotwordReloadResponse)
async def hotwords_rollback():
    """回滚到上一版默认词表。"""
    if not settings.HOTWORD_RELOAD_ENABLED:
        raise ASRException(ErrorCode.INPUT_PARAM_FAILED, "词表热更新未开启")

    loop = asyncio.get_event_loop()
    try:
        cache = await loop.run_in_executor(_cpu_executor, hotword_manager.rollback)
    except HotwordValidationError as e:
        raise ASRException(ErrorCode.INPUT_PARAM_FAILED, f"回滚失败：{e}")
    except Exception as e:
        raise ASRException(ErrorCode.INPUT_PARAM_FAILED, f"回滚失败：{e}")

    return HotwordReloadResponse(
        code=0,
        version=cache.version,
        md5=cache.md5,
        count=cache.count,
        route=cache.route,
        message=f"已回滚至上一版内容（发布为 version {cache.version}）",
    )


@app.post("/chinese_asr", response_model=ASRResponse)
async def asr_recognize(req: ASRRequest):
    """
    语音识别接口 — 三级流水线架构。

    Stage 1 (CPU): 音频解码 + VAD 切段 + 长音频切分
    Stage 2 (CPU): 特征提取（线程池并行）
    Stage 3 (GPU): ASR 推理（Scheduler batch 调度）

    多请求间各 Stage 独立并行，CPU/GPU 同时满载。
    """
    # 并发控制：等待至多 ACQUIRE_TIMEOUT 秒，超时则拒绝（SERVICE_BUSY）。
    # ACQUIRE_TIMEOUT=0 表示无限等待（不拒绝，退回纯排队语义）。
    if settings.ACQUIRE_TIMEOUT > 0:
        try:
            await asyncio.wait_for(
                _concurrent_semaphore.acquire(),
                timeout=settings.ACQUIRE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            asr_request_total.labels(status="error").inc()
            raise ASRException(
                ErrorCode.SERVICE_BUSY,
                f"服务过载，并发已达上限 {settings.MAX_CONCURRENT_REQUESTS}，请稍后重试",
            )
    else:
        await _concurrent_semaphore.acquire()

    try:
        rid = generate_request_id()
        request_id_var.set(rid)
        start_time = time.time()

        loop = asyncio.get_event_loop()

        # ====== Stage 1: 音频解码 + VAD + 切段（CPU 线程池） ======
        def _stage1_cpu():
            t0 = time.time()
            pcm, sample_rate, audio_duration_ms = _decode_audio(req.b64)
            t1 = time.time()
            vad_segments = vad_engine.detect(pcm, sample_rate)
            t2 = time.time()
            chunks = segment_to_chunks(vad_segments, audio_duration_ms)
            t3 = time.time()
            sub_ms = {"decode": t1 - t0, "vad": t2 - t1, "segment": t3 - t2}
            if settings.VERBOSE:
                logger.debug(
                    f"[Stage1] 解码={int(sub_ms['decode']*1000)}ms, "
                    f"VAD={int(sub_ms['vad']*1000)}ms({len(vad_segments)}段), "
                    f"切段={int(sub_ms['segment']*1000)}ms({len(chunks)}chunks)"
                )
            return pcm, sample_rate, audio_duration_ms, vad_segments, chunks, sub_ms

        _t_s1 = time.time()
        (
            pcm, sample_rate, audio_duration_ms, vad_segments, chunks, _s1_sub,
        ) = await loop.run_in_executor(_cpu_executor, _stage1_cpu)
        asr_stage_duration.labels(stage="stage1_decode_vad_seg").observe(time.time() - _t_s1)
        # Stage1 内部拆分打点（定位 decode/vad/segment 哪个是主导）
        for _k, _v in _s1_sub.items():
            asr_stage_duration.labels(stage=f"stage1_{_k}").observe(_v)

        # ====== Stage 2: 特征提取（CPU 线程池） ======
        # 热词路由（按生效词表大小三路分流）：
        #   1) 客户端传 hotwords → 路径 A：实时编码（含 Top-N 截断）
        #   2) 未传 + 默认词表 route=A（≤MAX_HOTWORD_NUM）→ 复用启动预编码缓存
        #   3) 未传 + 默认词表 route=B（>MAX_HOTWORD_NUM）→ bias=None（普通 ASR，后续 Faiss 纠错）
        bias_embeddings = None
        use_faiss_correction = False
        if req.hotwords and asr_engine.has_bias_model:
            bias_embeddings = await _encode_hotwords(req.hotwords)
        elif not req.hotwords:
            default_cache = hotword_manager.cache
            if default_cache is not None:
                if default_cache.route == "A":
                    bias_embeddings = default_cache.bias_embed
                else:
                    # route=B：普通 ASR + Faiss 后处理纠错（阶段 4 实现）
                    use_faiss_correction = True

        def _extract_one(chunk: ChunkMeta) -> np.ndarray:
            chunk_audio = extract_chunk_audio(pcm, chunk, sample_rate)
            return extract_features(
                chunk_audio,
                sample_rate=sample_rate,
                cmvn_mean=_cmvn_mean,
                cmvn_istd=_cmvn_istd,
            )

        # Stage 2：多 chunk 特征提取并行（各 chunk 提交到 CPU 线程池，asyncio.gather 并发）
        # 长音频多 chunk 时，相比串行循环显著降低 Stage2 墙钟耗时。
        t_feat0 = time.time()
        feat_tasks = [loop.run_in_executor(_cpu_executor, _extract_one, c) for c in chunks]
        features_list = await asyncio.gather(*feat_tasks)
        asr_stage_duration.labels(stage="stage2_feature").observe(time.time() - t_feat0)
        if settings.VERBOSE:
            shapes = [f.shape for f in features_list]
            logger.debug(
                f"[Stage2] 特征提取={int((time.time()-t_feat0)*1000)}ms（并行）, "
                f"chunks={len(features_list)}, shapes={shapes}"
            )

        # ====== Stage 3: GPU 推理（Scheduler: 固定 shape bucket + batch_timeout） ======
        _t_s3 = time.time()
        with asr_inference_duration.time():
            tasks = []
            for features in features_list:
                task = gpu_scheduler.submit(features, bias_embeddings)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
        asr_stage_duration.labels(stage="stage3_gpu").observe(time.time() - _t_s3)

        # 检查异常
        chunk_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise ASRException(
                    ErrorCode.ASR_INFER_FAILED,
                    f"Chunk {i} 推理失败: {result}",
                )
            chunk_results.append(result)

        # ====== 结果合并 ======
        response = _build_response(
            chunks, chunk_results, hotwords=req.hotwords,
            use_faiss_correction=use_faiss_correction,
            article_url=req.article_url,
        )

        # 记录日志
        elapsed_ms = (time.time() - start_time) * 1000
        log_request(
            logger,
            audio_duration_ms=audio_duration_ms,
            vad_segments=len(vad_segments),
            asr_latency_ms=elapsed_ms,
            result_length=len(response.text),
        )
        asr_request_total.labels(status="success").inc()

        return response
    finally:
        _concurrent_semaphore.release()


# ============================================================
# 内部函数
# ============================================================
def _decode_audio(b64_str: str) -> tuple[np.ndarray, int, int]:
    """
    解码 Base64 音频为 PCM。

    返回: (pcm_float32, sample_rate, duration_ms)
    """
    try:
        audio_bytes = base64.b64decode(b64_str)
    except Exception:
        raise ASRException(ErrorCode.DECODE_FAILED, "Base64 解码失败")

    if not audio_bytes:
        raise ASRException(ErrorCode.INPUT_PARAM_FAILED, "音频数据为空")

    try:
        pcm, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        raise ASRException(ErrorCode.DECODE_FAILED, "音频格式不合法，无法解码")

    # 采样率检查
    if sample_rate != 16000:
        raise ASRException(
            ErrorCode.DECODE_FAILED,
            f"采样率不匹配，需要16kHz，实际为{sample_rate}Hz",
        )

    # 确保单声道
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]

    # 空音频检查
    if len(pcm) == 0:
        raise ASRException(ErrorCode.INPUT_PARAM_FAILED, "音频数据为空")

    duration_ms = int(len(pcm) / sample_rate * 1000)

    # 时长上限检查（MAX_AUDIO_DURATION_MS=0 表示不限制）
    if settings.MAX_AUDIO_DURATION_MS > 0 and duration_ms > settings.MAX_AUDIO_DURATION_MS:
        raise ASRException(
            ErrorCode.AUDIO_TOO_LONG,
            f"音频时长 {duration_ms/1000:.1f}s 超出上限 "
            f"{settings.MAX_AUDIO_DURATION_MS/1000:.0f}s",
        )

    return pcm, sample_rate, duration_ms


async def _encode_hotwords(hotwords: list[str]) -> np.ndarray | None:
    """
    将热词列表编码为 bias embeddings（路径 A：SeACo 在线热词）。

    流程：hotwords → Top-N 截断 → tokenizer.encode → 追加 [sos]=[1] 哨兵 → padding → bias encoder → embeddings

    超过 MAX_HOTWORD_NUM（默认 256）时截断保留前 N 个并告警，
    保证 bias 维度恒定、TRT engine profile 无需重建（显存上界固定）。

    SeACo 架构要求 hotword 矩阵末尾必须有一行 [sos] 占位（NO_BIAS 标记），
    由模型 SeACo decoder 内部用于"无热词修正"的回退路径。

    bias 编码统一走 GPU 单线程池（gpu_scheduler.encode_hotwords），避免 CUDA stream 冲突。
    """
    # 过滤空热词
    valid = [hw for hw in hotwords if hw]
    if not valid:
        return None

    # Top-N 截断（超限保留前 N 个 + 告警）
    max_num = settings.MAX_HOTWORD_NUM
    if len(valid) > max_num:
        logger.warning(
            f"热词数量 {len(valid)} 超过上限 {max_num}，截断保留前 {max_num} 个"
        )
        valid = valid[:max_num]

    # 将每个热词编码为 token ID 序列
    # 注：此处在 encode 后过滤 OOV（编码为空的词），与预编码缓存路径
    # （hotword_manager._validate_and_clean 在编码前剔 OOV）行为等价——
    # 最终行数只会 ≤ MAX_HOTWORD_NUM，加哨兵后 ≤ MAX_HOTWORD_NUM+1，不超 engine profile。
    encoded = [tokenizer.encode(hw) for hw in valid]
    encoded = [ids for ids in encoded if ids]
    if not encoded:
        return None

    # 追加 [sos]=[1] 哨兵（SeACo NO_BIAS 占位）
    encoded.append([1])

    # Padding 到相同长度
    max_len = max(len(ids) for ids in encoded)
    padded = np.zeros((len(encoded), max_len), dtype=np.int64)
    for i, ids in enumerate(encoded):
        padded[i, :len(ids)] = ids

    # 通过 bias encoder 获取 embeddings（GPU 单线程池）
    return await gpu_scheduler.encode_hotwords(padded)


def _build_response(
    chunks: list[ChunkMeta],
    logits_list: list[np.ndarray],
    hotwords: list[str] | None = None,
    use_faiss_correction: bool = False,
    article_url: str | None = None,
) -> ASRResponse:
    """
    构建最终响应。

    将各 chunk 的 logits 解码为文本，并恢复原始时间戳。
    use_faiss_correction=True（路径 B）时，对每段文本做拼音检索纠错。
    article_url：原样透传请求中的 URL 到响应，未传时为 None。
    """
    corrector = None
    if use_faiss_correction:
        from src.hotword_faiss import faiss_corrector
        if faiss_corrector.is_ready:
            corrector = faiss_corrector

    detail: dict[str, SegmentDetail] = {}
    full_text_parts: list[str] = []

    for i, (chunk, logits) in enumerate(zip(chunks, logits_list)):
        # 解码 logits → 文本（argmax 贪心解码 + tokenizer）
        token_ids = np.argmax(logits, axis=-1).flatten()
        text = tokenizer.decode(token_ids)

        # 路径 B：Faiss 后处理纠错
        if corrector is not None and text:
            text = corrector.correct(text)

        # 使用原始 VAD 时间戳
        detail[str(i)] = SegmentDetail(
            text=text,
            start_ms=chunk.effective_start_ms,
            end_ms=chunk.effective_end_ms,
        )
        full_text_parts.append(text)

    return ASRResponse(
        code=0,
        text="".join(full_text_parts),
        article_url=article_url,
        detail=detail,
    )
