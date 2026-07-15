"""
SeACo-Paraformer FastAPI 服务入口

三级流水线（Stage1 解码/VAD/切段 → Stage2 特征提取 → Stage3 GPU 推理）+ 结果后处理
（贪心解码 → 字级时间戳 → Faiss 纠错 → 句子级标点分句）+ 热词双路路由
（A: SeACo 在线 / B: Faiss 后处理）+ 词表运行时热更新。

时间戳三级粒度（按开关递进）：
    - 段级（默认）：asr[] 每项为 VAD 切段，timestamp 源自 VAD 时间轴
    - 字级（ENABLE_WORD_TIMESTAMP）：asr[].words 每字时间戳（CIF timestamp head）
    - 子句级（ENABLE_SENTENCE_TIMESTAMP，强依赖字级）：asr[] 每项为一子句，
      CT-Transformer 标点模型（纯 onnxruntime）逐 token 恢复标点，任何标点（，。？、）
      都切成独立子句 + 字级时间戳定位子句边界（见 src/sentence_segmenter.py）

接口：
- POST /chinese_asr      — 中文语音识别（base64 音频 + 可选 hotwords/article_url）
- GET  /health           — 健康检查
- GET  /metrics          — Prometheus 指标
- POST /hotwords/reload   — 重载默认词表（运行时热更新）
- GET  /hotwords/status   — 查看当前词表版本状态
- POST /hotwords/rollback — 回滚到上一版词表
"""

import asyncio
import base64
import io
import logging
import os
import signal
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

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
    ASRSegment,
    ASRWord,
    ErrorResponse,
    HealthResponse,
    HotwordReloadRequest,
    HotwordReloadResponse,
    HotwordStatusResponse,
)
from src.sentence_segmenter import sentence_segmenter
from src.timestamp import compute_word_timestamps
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
    "fastapi_requests_total",
    "FastAPI 客户端请求总数 (按 endpoint/method/status/http_status 维度)",
    labelnames=["method", "endpoint", "status", "http_status"],
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

    # 句子级时间戳（asr[] 粒度变为句）：强依赖字级时间戳定位句子时间边界。
    # 开启句子级但未开字级时间戳 → 无法构造句子时间，自动降级回段级 + 告警。
    if settings.ENABLE_SENTENCE_TIMESTAMP:
        if not settings.ENABLE_WORD_TIMESTAMP:
            logger.warning(
                "ENABLE_SENTENCE_TIMESTAMP=true 但 ENABLE_WORD_TIMESTAMP=false："
                "句子级时间戳依赖字级时间戳定位句子边界，已自动降级回段级输出。"
                "如需句子级请同时开启 ENABLE_WORD_TIMESTAMP=true。"
            )
        else:
            sentence_segmenter.load()

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

    # 打印所有实际生效的运行配置（复现 / 排错用；结构化 JSON 字段，按启用模块动态组装）
    _cfg_record = logger.makeRecord(
        name=logger.name, level=logging.INFO, fn="", lno=0,
        msg="实际生效运行配置", args=(), exc_info=None,
    )
    _cfg_record.extra_fields = {"effective_config": settings.dump_effective_config()}
    logger.handle(_cfg_record)

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
    # 关闭 CPU 线程池（Stage1/2）+ GPU 线程池（scheduler），释放线程资源
    if _cpu_executor is not None:
        _cpu_executor.shutdown(wait=False, cancel_futures=True)
    try:
        from src.scheduler import _gpu_executor
        _gpu_executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    # 多进程 Prometheus：本 worker 退出时标记进程死亡，清理其 gauge 分片，
    # 避免死进程的指标残留污染聚合结果（Counter/Histogram 保留累计值供聚合）。
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        try:
            multiprocess.mark_process_dead(os.getpid())
        except Exception:
            pass
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
# 指标中间件（统一记录所有 HTTP 请求）
# ============================================================
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """按 method/endpoint/status/http_status 维度统一记录请求计数。

    统一在中间件出口打点，覆盖成功、业务异常（ASRException）、兜底异常三种路径
    （异常经异常处理器转成 JSONResponse 后仍会回到这里），避免在各处手动 inc
    造成漏记或重复计数。endpoint 用路由模板（如 /chinese_asr）而非原始 path，
    避免高基数 label 撑爆 Prometheus。
    """
    response = await call_next(request)

    # 路由模板优先（无匹配路由时退回原始 path，如 404）
    route = request.scope.get("route")
    endpoint = getattr(route, "path", None) or request.url.path
    status = "success" if response.status_code < 400 else "error"

    asr_request_total.labels(
        method=request.method,
        endpoint=endpoint,
        status=status,
        http_status=str(response.status_code),
    ).inc()

    return response


# ============================================================
# 异常处理器
# ============================================================
@app.exception_handler(ASRException)
async def asr_exception_handler(request: Request, exc: ASRException):
    """统一业务异常处理。

    /chinese_asr 接口的失败响应补充 text/detail 空值字段，与成功响应结构保持一致，
    便于客户端用统一结构解析（成功 text 有值/detail 有段，失败均为空）。
    其他接口（如 /hotwords/*）保持精简的 code/error/message 结构。

    请求计数统一由 metrics 中间件按 method/endpoint/status/http_status 记录，
    此处不再手动 inc（避免与中间件重复计数）。
    """
    content = {"code": int(exc.code)}
    if request.url.path == "/chinese_asr":
        # 失败响应保持与成功响应结构一致（istar_asr/article_url/asr 空值），
        # 便于客户端用统一结构解析。article_url 在错误分支无法从已解析请求获得，
        # 填 None（与"未传"语义等价）。
        content["article_url"] = None
        content["istar_asr"] = ""
        content["asr"] = []
    content["error"] = exc.code.name
    content["message"] = exc.message
    return JSONResponse(
        status_code=exc.http_status,
        content=content,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜底异常处理：未被 ASRException 包装的异常（切段/特征/解码等）统一转为
    结构化 500（ASR_INFER_FAILED），避免 FastAPI 默认处理器返回无 code/asr 字段的
    原生 500 破坏 API 契约。

    请求计数由 metrics 中间件统一记录（见上）。"""
    logger.error(f"未处理异常: {type(exc).__name__}: {exc}")
    content = {"code": int(ErrorCode.ASR_INFER_FAILED)}
    if request.url.path == "/chinese_asr":
        content["article_url"] = None
        content["istar_asr"] = ""
        content["asr"] = []
    content["error"] = ErrorCode.ASR_INFER_FAILED.name
    content["message"] = f"服务内部错误: {type(exc).__name__}"
    return JSONResponse(status_code=500, content=content)


# ============================================================
# 路由
# ============================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口（加载态 + 运行时健康，R12/R14）。

    status 判定：
        - 模型未加载 → degraded（供探针识别加载失败/半死状态）
        - 运行时连续推理失败超 HEALTH_MAX_CONSECUTIVE_FAILURES（GPU 卡死典型症状）→ degraded
        - 加载阶段发生静默降级（后端回退，如 TRT→ORT）→ 仍 ok（服务可用），
          但 runtime.degraded_reason 暴露原因供运维察觉低性能/降功能模式
        - 否则 ok

    HEALTH_ACTIVE_PROBE=true 时额外主动跑一次极小 dummy 推理（带 INFER_TIMEOUT 超时），
    直接验证 GPU 链路存活（探测被动统计未覆盖的“无流量期间”卡死）。
    """
    loaded = vad_engine.is_loaded and asr_engine.is_loaded
    runtime = asr_engine.runtime_health()

    # 主动探针（可选）：真跑一次极小推理验证 GPU 链路（超时视为不健康）
    if loaded and settings.HEALTH_ACTIVE_PROBE:
        loop = asyncio.get_event_loop()
        try:
            probe_ok = await asyncio.wait_for(
                loop.run_in_executor(_cpu_executor, asr_engine.active_probe),
                timeout=settings.INFER_TIMEOUT,
            )
            runtime = asr_engine.runtime_health()
            runtime["active_probe_ok"] = probe_ok
        except asyncio.TimeoutError:
            asr_engine.record_infer_failure()
            runtime = asr_engine.runtime_health()
            runtime["active_probe_ok"] = False

    healthy = loaded and runtime.get("runtime_ok", True)
    body = HealthResponse(
        status="ok" if healthy else "degraded",
        device=settings.get_device(),
        models_loaded=loaded,
        runtime=runtime,
    )
    # 不健康时返回 HTTP 503，使 K8s readiness/liveness 探针（按状态码判定）真正摘除
    # 卡死/未加载实例；健康返回 200。静默降级（degraded_reason 有值但 runtime_ok=True）
    # 仍算健康（服务可用），只在 body 暴露原因，不触发探针摘除。
    status_code = 200 if healthy else 503
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.get("/metrics")
async def metrics():
    """Prometheus 指标接口。

    多进程聚合：uvicorn 多 worker（WORKERS>1）时每个 worker 是独立进程，各自的
    Counter/Histogram 互不相通。若设置了环境变量 PROMETHEUS_MULTIPROC_DIR，则用
    MultiProcessCollector 从该目录聚合所有 worker 的指标（QPS 等按全进程汇总，
    否则 /metrics 只反映被抓到的那个 worker，QPS 严重偏低失真）。
    未设置该环境变量时退回单进程默认 registry（兼容 WORKERS=1）。

    QPS 由 Prometheus 服务端对 fastapi_requests_total 做 rate() 计算（Counter + rate
    是标准做法），本端点只需保证多进程下计数完整可聚合。
    """
    # scrape 时刷新 GPU 显存实际值（Gauge 标准做法）
    _update_gpu_memory_metric()
    _update_scheduler_metrics()

    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        # 多进程模式：聚合所有 worker 的指标
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        # 单进程模式：默认全局 registry
        data = generate_latest()

    return JSONResponse(
        content=data.decode("utf-8"),
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
    语音识别接口 — 三级流水线 + 结果后处理。

    准入：并发信号量（MAX_CONCURRENT_REQUESTS / ACQUIRE_TIMEOUT，超时 SERVICE_BUSY）。

    Stage 1 (CPU 线程池): 音频解码 + VAD 切段（VAD_SESSION_POOL_SIZE）+ 均匀切段
    Stage 2 (CPU 线程池 CPU_THREAD_POOL_SIZE): 热词路由 + 特征提取（多 chunk 并行）
        热词路由（防误触发）：
          - 客户端传 hotwords 且 ENABLE_HOTWORD → 路径 A：SeACo 在线编码 bias（Top-N 截断）
          - 客户端不传 且 ENABLE_FAISS_CORRECTION → 路径 B：默认词表 Faiss 后处理纠错
    Stage 3 (GPU 池 GPU_STREAM_POOL_SIZE): ASR 推理（dynamic batching：BATCH/BATCH_TIMEOUT）
        - 超长音频分片限流 MAX_INFLIGHT_CHUNKS_PER_REQUEST（在途 chunk 上限）
        - 单 chunk future 超时 INFER_TIMEOUT（防 GPU 卡死永久挂起）
        - 字级时间戳 ENABLE_WORD_TIMESTAMP（第 5 段 timestamp engine 输出 us_alphas/us_cif_peak）
    结果合并 (CPU 线程池): _build_response —— 贪心解码 → 字级时间戳 → Faiss 纠错（I3 回写 words）
        → 子句级分句（ENABLE_SENTENCE_TIMESTAMP，CT-Transformer 标点分类，asr[] 粒度变为子句）
        （CPU 密集，放线程池避免阻塞事件循环）

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
            # 计数由 metrics 中间件统一记录（SERVICE_BUSY → 503 → status=error）
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
            pcm, sample_rate, audio_duration_ms = _decode_audio(req.base64)
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

        # VAD 后无有效语音（静音/极短音频）：不报错，返回成功空结果 + 提示
        if not chunks:
            logger.info("VAD 后无有效语音段，返回空结果（音频内容为空）")
            return ASRResponse(
                code=0,
                article_url=req.article_url,
                istar_asr="",
                asr=[],
                message="音频内容为空",
            )

        # ====== Stage 2: 特征提取（CPU 线程池） ======
        # 热词路由（防通用识别误触发）：
        #   1) 客户端传 hotwords → 路径 A：SeACo 实时编码
        #      （客户端主动传 = 明确知道音频含这些词，激进增强合理；
        #        超 MAX_HOTWORD_NUM 时 _encode_hotwords 内部 Top-N 截断）
        #   2) 客户端不传 → 默认词表恒走路径 B（Faiss 保守纠错）
        #      （通用识别多数音频不含默认热词，SeACo 会误纠相似音，
        #        Faiss 三重判定仅在拼音+编辑距离高度吻合时替换，大幅降低误触发）
        #   两条路径分别受 ENABLE_HOTWORD / ENABLE_FAISS_CORRECTION 开关控制，
        #   关闭对应开关即跳过该路径（纯通用识别可全关省开销）。
        bias_embeddings = None
        use_faiss_correction = False

        if req.hotwords and asr_engine.has_bias_model and settings.ENABLE_HOTWORD:
            # 路径 A：SeACo 在线增强（客户端主动传热词，且未关闭热词开关）
            bias_embeddings = await _encode_hotwords(req.hotwords)
        elif not req.hotwords and settings.ENABLE_FAISS_CORRECTION:
            # 客户端不传：默认词表恒走 Faiss（_determine_route 已固定 route=B）
            default_cache = hotword_manager.cache
            if default_cache is not None:
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
        # M3 超长音频分片限流：用 per-request 信号量约束同时在途（已 submit 未完成）的
        # chunk 数，避免超长音频（上千 chunk）一次性灌满调度器、饿死其他请求、内存峰值飙高。
        # 完成一个 chunk 立即放行下一个，保持流水线连续（非分批 barrier，无停顿）。
        # 限流值 0 或 >= chunk 数时退化为旧的一次性 gather 行为。
        _t_s3 = time.time()
        with asr_inference_duration.time():
            inflight_limit = settings.MAX_INFLIGHT_CHUNKS_PER_REQUEST
            if inflight_limit and inflight_limit > 0 and len(features_list) > inflight_limit:
                chunk_sem = asyncio.Semaphore(inflight_limit)

                async def _submit_throttled(feats):
                    async with chunk_sem:
                        return await gpu_scheduler.submit(feats, bias_embeddings)

                tasks = [asyncio.ensure_future(_submit_throttled(f)) for f in features_list]
            else:
                tasks = [gpu_scheduler.submit(f, bias_embeddings) for f in features_list]

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

        # ====== 结果合并（CPU 线程池） ======
        # _build_response 含解码 + Faiss 纠错 + 子句级标点分句（CT-Transformer onnx 推理，
        # CPU 密集）。放线程池执行，避免在事件循环线程阻塞、拖累本 worker 其他请求的
        # async 调度（GPU 结果回收 / 新请求接入）。onnxruntime/Faiss 均释放 GIL，可并行。
        response = await loop.run_in_executor(
            _cpu_executor,
            lambda: _build_response(
                chunks, chunk_results, hotwords=req.hotwords,
                use_faiss_correction=use_faiss_correction,
                article_url=req.article_url,
            ),
        )

        # 记录日志
        elapsed_ms = (time.time() - start_time) * 1000
        log_request(
            logger,
            audio_duration_ms=audio_duration_ms,
            vad_segments=len(vad_segments),
            asr_latency_ms=elapsed_ms,
            result_length=len(response.istar_asr),
        )

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


def _build_word_timestamps(
    ts_data: dict | None,
    char_list: list[str],
    chunk_start_ms: int,
) -> list[ASRWord]:
    """
    用 timestamp head 输出（us_alphas/us_cif_peak）计算字级时间戳。

    对齐 FunASR ts_prediction_lfr6_standard（见 src/timestamp.py）：
        相邻 fire 间隔作为前一 token 时长（不重叠），单 token 超时截断，
        首尾静音扣除。时间戳加 chunk_start_ms 偏移对齐全局音频时间轴。

    ts_data 为 None（旧 engine/ORT 无 timestamp 输出）时返回空列表（降级）。
    """
    if ts_data is None or not char_list:
        return []
    ts_list = compute_word_timestamps(
        us_alphas=ts_data["us_alphas"],
        us_cif_peak=ts_data["us_cif_peak"],
        num_tokens=len(char_list),
        upsample_times=settings.TIMESTAMP_UPSAMPLE_TIMES,
        offset_ms=chunk_start_ms,
    )
    words = []
    n = min(len(ts_list), len(char_list))
    for i in range(n):
        s, e = ts_list[i]
        words.append(ASRWord(
            text=char_list[i],
            timestamp=[round(s / 1000.0, 3), round(e / 1000.0, 3)],
        ))
    return words


def _clamp_words_monotonic(words: list[ASRWord], prev_end_s: float) -> float:
    """跨 chunk 字级时间戳单调钳制，消除段边界重叠。

    每个 chunk 的字级时间戳在各自 VAD 时间轴独立计算（段内已保证不重叠），
    但加偏移拼接到全局时间轴后，上一段尾字 end 可能晚于下一段首字 start，
    产生重叠（如尾字 [72.682,72.742] 与下段首字 [72.682,72.982]）。

    这里按全局顺序逐字钳制：任一字 start 不早于前一字 end；若钳制后 start>end，
    则把 end 也顶到 start（零时长，避免逆序）。返回本段最后一个字的 end，
    供下一段继续钳制。
    """
    for w in words:
        s, e = w.timestamp
        if s < prev_end_s:
            s = prev_end_s
        if e < s:
            e = s
        w.timestamp = [round(s, 3), round(e, 3)]
        prev_end_s = e
    return prev_end_s


def _decode_char_list(token_ids: np.ndarray) -> list[str]:
    """token_ids → 有效字符列表（过滤 <blank>/<sos>/<eos>，用 _token_list 逐字取）。

    字级时间戳按 CIF fire 对齐（每个声学 token 一次 fire），故须保持“一 token 一项”，
    数量与 fire 数一致，不能在此合并 BPE subword（否则时间戳与字错位）。

    但显示口径须与 tokenizer.decode（段 text）一致：清理 BPE 连接标记 `@@` 与
    sentencepiece 前缀 `▁`（I5 中英混合错位修复）。这样英文 subword 如 `and@@`/`roid`
    显示为 `and`/`roid`（各自带时间戳），拼接后与段 text 的 `android` 字面一致，
    仅粒度更细（按 subword 切分）；中文逐字 1 token=1 字，行为不变。
    """
    chars = []
    tl = getattr(tokenizer, "_token_list", None)
    for tid in token_ids.tolist():
        tid = int(tid)
        if tid <= 2:  # blank/sos/eos
            continue
        if tl is not None and 0 <= tid < len(tl) and tl[tid]:
            tok = tl[tid]
            # 跳过 <...> 特殊标记（与 tokenizer.decode 一致）
            if tok.startswith("<") and tok.endswith(">"):
                continue
            # 清理 BPE 连接标记，保持与段 text 显示口径一致
            tok = tok[:-2] if tok.endswith("@@") else tok
            tok = tok.replace("▁", "")
            if tok:
                chars.append(tok)
    return chars


def _apply_faiss_to_words(
    words: list[ASRWord],
    orig_text: str,
    spans: list[tuple[int, int, str]],
) -> list[ASRWord]:
    """将 Faiss 替换区间同步映射到字级 words，保证 words 与纠错后 text 一致（I3）。

    orig_text 为纠错前段文本（Faiss 检索所用），spans 为按 start 升序、互不重叠的
    替换区间 (start, end, cand)：orig_text[start:end] → cand。

    对齐策略：字级 words 的 text 逐个拼接理论上等于 orig_text（纯中文 1 字 1 word 严格
    成立）。据此建立“word 序号 → orig_text 字符偏移”映射：
      - 命中区间被 cand 逐字替换：被覆盖 words 的时间区间合并为 [t0,t1]，按 cand 字数
        等分重新切分，生成新的字级 words（时间单调、与 cand 字面一致）。
      - 未命中区间的 words 原样保留。
    若拼接与 orig_text 不一致（中英混合含空格等边界情形），无法可靠对齐，则放弃 words
    改写、保持原样（text 仍已纠错），避免破坏时间戳（保守降级）。
    """
    # 建立 word → 字符区间映射（按 word.text 长度累加）
    concat = "".join(w.text for w in words)
    if concat != orig_text:
        # 口径不一致（空格/BPE 边界），无法可靠对齐 → 放弃 words 改写（保守降级）
        return words

    # 每个 word 覆盖的字符区间 [cstart, cend)
    word_char_ranges: list[tuple[int, int]] = []
    pos = 0
    for w in words:
        word_char_ranges.append((pos, pos + len(w.text)))
        pos += len(w.text)

    span_map = {s[0]: (s[1], s[2]) for s in spans}
    out: list[ASRWord] = []
    wi = 0
    n = len(words)
    while wi < n:
        cstart = word_char_ranges[wi][0]
        if cstart in span_map:
            end, cand = span_map[cstart]
            # 收集覆盖 [cstart, end) 的所有 word
            covered = []
            j = wi
            while j < n and word_char_ranges[j][1] <= end:
                covered.append(words[j])
                j += 1
            if covered:
                t0 = covered[0].timestamp[0]
                t1 = covered[-1].timestamp[1]
                # 按 cand 字数等分时间区间
                m = len(cand)
                if m <= 0:
                    wi = j
                    continue
                step = (t1 - t0) / m
                for k, ch in enumerate(cand):
                    s = round(t0 + step * k, 3)
                    e = round(t0 + step * (k + 1), 3) if k < m - 1 else round(t1, 3)
                    out.append(ASRWord(text=ch, timestamp=[s, e]))
                wi = j
            else:
                out.append(words[wi])
                wi += 1
        else:
            out.append(words[wi])
            wi += 1
    return out


def _build_response(
    chunks: list[ChunkMeta],
    chunk_results: list[tuple[np.ndarray, dict | None]],
    hotwords: list[str] | None = None,
    use_faiss_correction: bool = False,
    article_url: str | None = None,
) -> ASRResponse:
    """
    构建最终响应。

    将各 chunk 的 logits 解码为文本，并恢复原始时间戳；
    利用 CIF alphas 反推字级时间戳，填充 ASRSegment.words。
    use_faiss_correction=True（路径 B）时，对每段文本做拼音检索纠错。
    article_url：原样透传请求中的 URL 到响应，未传时为 None。
    """
    corrector = None
    if use_faiss_correction:
        from src.hotword_faiss import faiss_corrector
        if faiss_corrector.is_ready:
            corrector = faiss_corrector

    asr_segments: list[ASRSegment] = []
    full_text_parts: list[str] = []
    # 全局字级时间戳游标：跨 chunk 单调钳制，消除段边界重叠
    prev_word_end_s = 0.0

    for i, (chunk, (logits, ts_data)) in enumerate(zip(chunks, chunk_results)):
        # 解码 logits → token_ids → 文本（argmax 贪心解码 + tokenizer）
        token_ids = np.argmax(logits, axis=-1).flatten()
        text = tokenizer.decode(token_ids)

        # 字级时间戳（timestamp head：us_alphas/us_cif_peak → 官方 ts_prediction）
        # ★仅在 ts_data 存在（ENABLE_WORD_TIMESTAMP=true）时才解码字符列表 + 计算时间戳，
        #   关闭时间戳时 ts_data 恒为 None，跳过 _decode_char_list 的逐 token 循环，
        #   避免热路径无谓开销（极限压测下影响吞吐）。
        words_pyd: list[ASRWord] = []
        if ts_data is not None:
            char_list = _decode_char_list(token_ids)
            words_pyd = _build_word_timestamps(
                ts_data=ts_data,
                char_list=char_list,
                chunk_start_ms=chunk.effective_start_ms,
            )
            # 跨 chunk 边界单调钳制（段内官方算法已不重叠，仅需处理拼接处）
            if words_pyd:
                prev_word_end_s = _clamp_words_monotonic(words_pyd, prev_word_end_s)

        # 路径 B：Faiss 后处理纠错。返回替换区间，同步回写字级 words，
        # 保证 words 拼接 == 纠错后 text（I3：消除 words 反映纠错前、text 纠错后的不一致）。
        if corrector is not None and text:
            corrected_text, corr_spans = corrector.correct_with_spans(text)
            if corr_spans and words_pyd:
                words_pyd = _apply_faiss_to_words(words_pyd, text, corr_spans)
            text = corrected_text

        # 段级时间戳用原始 VAD 时间戳（ms → s，保留 3 位小数与外部标准对齐）
        start_s = round(chunk.effective_start_ms / 1000.0, 3)
        end_s = round(chunk.effective_end_ms / 1000.0, 3)

        asr_segments.append(ASRSegment(
            idx=i,
            slid="",           # 语种识别未实现
            text=text,
            speaker="",        # 说话人识别未实现
            timestamp=[start_s, end_s],
            words=words_pyd,
        ))
        full_text_parts.append(text)

    # ── 句子级时间戳（asr[] 粒度变为句）──
    # 启用且分句器就绪且字级时间戳可用时，把段级结果重组为句子级：
    #   对全文跑标点模型断句，按句子字符区间映射字级 words 时间 → 句子 [start,end]。
    # 任一前置不满足则保持段级输出（降级）。
    if (
        settings.ENABLE_SENTENCE_TIMESTAMP
        and settings.ENABLE_WORD_TIMESTAMP
        and sentence_segmenter.is_ready
    ):
        sent_segments = _build_sentence_segments(asr_segments)
        if sent_segments is not None:
            istar_asr = "".join(s.text for s in sent_segments)
            return ASRResponse(
                code=0,
                article_url=article_url,
                istar_asr=istar_asr,
                asr=sent_segments,
            )

    # istar_asr 段间用逗号分隔，便于阅读（各段为 VAD/切段单位，非完整句，用逗号最稳）
    istar_asr = "，".join(p for p in full_text_parts if p)

    return ASRResponse(
        code=0,
        article_url=article_url,
        istar_asr=istar_asr,
        asr=asr_segments,
    )


def _build_sentence_segments(seg_list: list[ASRSegment]) -> list[ASRSegment] | None:
    """将段级结果重组为子句级 asr[]（句子级时间戳）。

    CT-Transformer 标点模型逐 token 分类，对长文本无失效问题（内部按 PUNC_MAX_LEN
    无重叠分块推理），故直接汇总全局字级 words → 全文一次分句，无需按 VAD 段分窗。

    流程：
        1. 汇总所有段的字级 words 为全局有序序列（words 拼接 = 全文无标点文本）；
        2. split_sentences 恢复标点并按标点切子句，返回每子句 [c_start, c_end) 字符区间；
        3. 每个 word 按其「起始字符」唯一归属一个子句 → 子句 timestamp=[首字 start, 末字 end]，
           子句 words = 归属本子句的字级 words（text 用带标点子句）。

    对齐口径：字级 words 逐字（中文）或英文 subword，split_sentences 内部按去空白逐字对齐。
    正常情况 word.text 无空白（_decode_char_list 已清理 ▁/@@），字符偏移与拼接口径一致；
    若出现空白则口径不一致，保守降级回段级（见下方 whitespace 保护）。
    word 按「起始字符」唯一归属子句（而非区间重叠），避免横跨子句边界的多字符英文 subword
    被相邻两子句重复收入（造成 word 重复、timestamp 重叠）。
    返回 None 表示无法构造（无字级 words / 含空白口径不一致），上层保持段级输出。
    """
    all_words: list[ASRWord] = []
    for seg in seg_list:
        all_words.extend(seg.words)
    if not all_words:
        return None

    full_text = "".join(w.text for w in all_words)
    if not full_text:
        return None

    # Bug B 保护：split_sentences 内部按「去空白逐字」对齐，其返回的字符区间以无空白字符
    # 序列为基准。此处 full_text 由字级 words 拼接，正常无空白（_decode_char_list 已清理
    # ▁/@@）。若出现空白（异常 word.text），字符偏移口径不一致会导致映射错位，保守降级回段级。
    if any(ch.isspace() for ch in full_text):
        logger.warning("字级 words 含空白字符，句子级映射口径不一致，降级回段级输出")
        return None

    # 每个 word 覆盖的全局字符结束偏移（word.text 可能是多字符英文 subword）
    word_char_end: list[int] = []
    pos = 0
    for w in all_words:
        pos += len(w.text)
        word_char_end.append(pos)
    total_chars = pos

    sentences = sentence_segmenter.split_sentences(full_text)
    if not sentences:
        return None

    # 每个 word 唯一归属其「起始字符」所在的子句（Bug A 修复）：
    # 用起始字符 wstart 归属，而非「区间重叠」判定——否则横跨子句边界的多字符英文
    # subword 会被相邻两子句同时收入，造成 word 重复、子句 timestamp 重叠。
    # 子句字符区间 [c_start,c_end) 连续无缝覆盖 [0,total_chars)，故每个 wstart 唯一落在一个子句。
    word_start = [word_char_end[i] - len(all_words[i].text) for i in range(len(all_words))]

    out: list[ASRSegment] = []
    idx = 0
    for sent_text, (c_start, c_end) in sentences:
        c_start = max(0, min(c_start, total_chars))
        c_end = max(c_start, min(c_end, total_chars))
        # 收集起始字符落在本子句区间内的 word（唯一归属，无重复无重叠）
        sent_words = [
            all_words[wi] for wi in range(len(all_words))
            if c_start <= word_start[wi] < c_end
        ]
        if not sent_words:
            # 子句区间过短（夹在一个 subword 内部），无 word 起点落入 → 并入相邻，跳过
            continue
        out.append(ASRSegment(
            idx=idx,
            slid="",
            text=sent_text,
            speaker="",
            timestamp=[round(sent_words[0].timestamp[0], 3),
                       round(sent_words[-1].timestamp[1], 3)],
            words=sent_words,
        ))
        idx += 1

    return out if out else None
