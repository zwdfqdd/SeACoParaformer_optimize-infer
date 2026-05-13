"""
SeACo-Paraformer FastAPI 服务入口

提供：
- POST /asr — 语音识别接口
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
gpu_memory_usage = None  # 延迟初始化（仅 GPU 模式）


def _init_gpu_metrics():
    """初始化 GPU 内存指标（仅在 GPU 可用时）。"""
    global gpu_memory_usage
    try:
        from prometheus_client import Gauge
        gpu_memory_usage = Gauge(
            "gpu_memory_usage_bytes",
            "GPU 显存使用量（字节）",
        )
    except Exception:
        pass

# ============================================================
# 并发控制
# ============================================================
_concurrent_semaphore: asyncio.Semaphore | None = None


# ============================================================
# 应用生命周期
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭逻辑。"""
    global _concurrent_semaphore

    # 启动：加载模型
    logger.info("服务启动中...")
    _load_resources()
    vad_engine.load()
    asr_engine.load()
    await gpu_scheduler.start()
    _concurrent_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
    _init_gpu_metrics()
    logger.info("服务启动完成")

    # 注册 SIGHUP 信号处理（配置热更新）
    try:
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGHUP, _reload_config)
    except (NotImplementedError, AttributeError):
        pass  # Windows 不支持 SIGHUP

    yield

    # 关闭：优雅停止
    logger.info("服务关闭中...")
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
    """加载 CMVN 参数和 Tokenizer 词表。"""
    global _cmvn_mean, _cmvn_istd
    import os

    # 加载 CMVN
    cmvn_path = os.path.join(settings.MODEL_DIR, "asr", "fp16", "am.mvn")
    if os.path.exists(cmvn_path):
        try:
            _cmvn_mean, _cmvn_istd = load_cmvn(cmvn_path)
            logger.info(f"CMVN 加载成功: {cmvn_path}, shape={_cmvn_mean.shape}")
        except Exception as e:
            logger.warning(f"CMVN 加载失败: {e}，将跳过归一化")
    else:
        logger.warning(f"CMVN 文件不存在: {cmvn_path}，将跳过归一化")

    # 加载 Tokenizer
    vocab_path = os.path.join(settings.MODEL_DIR, "asr", "fp16", "tokens.json")
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(settings.MODEL_DIR, "asr", "fp16", "tokens.txt")

    if os.path.exists(vocab_path):
        try:
            tokenizer.load(vocab_path)
            logger.info(f"Tokenizer 加载成功: {vocab_path}, vocab_size={tokenizer.vocab_size}")
        except Exception as e:
            logger.warning(f"Tokenizer 加载失败: {e}")
    else:
        logger.warning(f"词表文件不存在，Tokenizer 未加载")


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="SeACo-Paraformer ASR Service",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# 异常处理器
# ============================================================
@app.exception_handler(ASRException)
async def asr_exception_handler(request: Request, exc: ASRException):
    """统一业务异常处理。"""
    asr_request_total.labels(status="error").inc()
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "code": int(exc.code),
            "error": exc.code.name,
            "message": exc.message,
        },
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
    return JSONResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/asr", response_model=ASRResponse)
async def asr_recognize(req: ASRRequest):
    """
    语音识别接口。

    输入: {"b64": "wav_16k_1channel_base64", "hotwords": ["张三", "李四"]}
    输出: {"code": 0, "text": "全文", "detail": {"0": {"text": "...", "start_ms": 0, "end_ms": 5200}}}
    """
    # 并发控制
    try:
        await asyncio.wait_for(
            _concurrent_semaphore.acquire(), timeout=0
        )
    except asyncio.TimeoutError:
        raise ASRException(ErrorCode.SERVICE_BUSY)

    try:
        # 设置请求 ID
        rid = generate_request_id()
        request_id_var.set(rid)
        start_time = time.time()

        # Step 1: 音频解码
        pcm, sample_rate, audio_duration_ms = _decode_audio(req.b64)

        # Step 2: VAD 切段
        vad_segments = vad_engine.detect(pcm, sample_rate)

        # Step 3: 长音频切段
        chunks = segment_to_chunks(vad_segments, audio_duration_ms)

        # Step 4: ASR 推理（通过 GPU Scheduler）
        with asr_inference_duration.time():
            chunk_results = await _infer_chunks(
                pcm, chunks, sample_rate, hotwords=req.hotwords
            )

        # Step 5: 结果拼接
        response = _build_response(chunks, chunk_results, hotwords=req.hotwords)

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

    # 时长限制检查（可选，暂不设上限）
    return pcm, sample_rate, duration_ms


async def _infer_chunks(
    pcm: np.ndarray,
    chunks: list[ChunkMeta],
    sample_rate: int,
    hotwords: list[str] | None = None,
) -> list[np.ndarray]:
    """
    对所有 chunk 进行 ASR 推理。

    流程：
    1. 热词编码（如有 hotwords 且 bias encoder 可用）
    2. 并行特征提取（CPU 密集，使用线程池）
    3. 通过 GPU Scheduler 统一调度 batch 推理
    """
    loop = asyncio.get_event_loop()

    # Step 1: 热词编码（通过 bias encoder）
    bias_embeddings = None
    if hotwords and asr_engine.has_bias_model:
        bias_embeddings = _encode_hotwords(hotwords)

    # Step 2: 并行特征提取
    def _extract_all_features():
        features_list = []
        for chunk in chunks:
            chunk_audio = extract_chunk_audio(pcm, chunk, sample_rate)
            features = extract_features(
                chunk_audio,
                sample_rate=sample_rate,
                cmvn_mean=_cmvn_mean,
                cmvn_istd=_cmvn_istd,
            )
            features_list.append(features)
        return features_list

    features_list = await loop.run_in_executor(None, _extract_all_features)

    # Step 3: 提交 GPU Scheduler 推理
    tasks = []
    for chunk, features in zip(chunks, features_list):
        task = gpu_scheduler.submit(features, chunk.duration_ms, bias_embeddings)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 检查异常
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise ASRException(
                ErrorCode.ASR_INFER_FAILED,
                f"Chunk {i} 推理失败: {result}",
            )
        final_results.append(result)

    return final_results


def _encode_hotwords(hotwords: list[str]) -> np.ndarray | None:
    """
    将热词列表编码为 bias embeddings。

    流程：hotwords → tokenizer.encode → padding → bias encoder → embeddings
    """
    # 将每个热词编码为 token ID 序列
    encoded = [tokenizer.encode(hw) for hw in hotwords if hw]
    if not encoded:
        return None

    # Padding 到相同长度
    max_len = max(len(ids) for ids in encoded)
    padded = np.zeros((len(encoded), max_len), dtype=np.int32)
    for i, ids in enumerate(encoded):
        padded[i, :len(ids)] = ids

    # 通过 bias encoder 获取 embeddings
    return asr_engine.encode_hotwords(padded)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 检查异常
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise ASRException(
                ErrorCode.ASR_INFER_FAILED,
                f"Chunk {i} 推理失败: {result}",
            )
        final_results.append(result)

    return final_results


def _build_response(
    chunks: list[ChunkMeta],
    logits_list: list[np.ndarray],
    hotwords: list[str] | None = None,
) -> ASRResponse:
    """
    构建最终响应。

    将各 chunk 的 logits 解码为文本，并恢复原始时间戳。
    """
    detail: dict[str, SegmentDetail] = {}
    full_text_parts: list[str] = []

    for i, (chunk, logits) in enumerate(zip(chunks, logits_list)):
        # 解码 logits → 文本（argmax 贪心解码 + tokenizer）
        token_ids = np.argmax(logits, axis=-1).flatten()
        text = tokenizer.decode(token_ids)

        # 使用有效时间戳（去除 overlap）
        detail[str(i)] = SegmentDetail(
            text=text,
            start_ms=chunk.effective_start_ms,
            end_ms=chunk.effective_end_ms,
        )
        full_text_parts.append(text)

    return ASRResponse(
        code=0,
        text="".join(full_text_parts),
        detail=detail,
    )
