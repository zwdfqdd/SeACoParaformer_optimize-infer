"""
结构化日志模块

双输出策略：
- stdout：JSON 格式输出，供 Docker/ELK 采集
- 本地文件：按天轮转，保留 7 天，路径 logs/asr_{pid}.log（多 worker 各进程独立文件）

日志字段：
request_id, audio_duration_ms, vad_segments, asr_latency_ms, result_length
"""

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from src.config import settings

# 请求级上下文变量
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# 日志目录
LOG_DIR = "logs"
LOG_FILE_PREFIX = "asr"
LOG_BACKUP_COUNT = 7  # 保留 7 天


class JSONFormatter(logging.Formatter):
    """JSON 格式日志输出。"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(""),
        }

        # 附加业务字段
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logger() -> logging.Logger:
    """
    初始化结构化日志。

    双输出：
    1. stdout — 供 Docker 日志驱动 / ELK 采集
    2. 本地文件 — 按天轮转，保留 7 天
    """
    logger = logging.getLogger("seaco_asr")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # 避免重复添加 handler（多 worker 场景）
    if logger.handlers:
        return logger

    formatter = JSONFormatter()

    # Handler 1: stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Handler 2: 本地文件（按天轮转，保留 7 天）
    # 多 worker（WORKERS>1）场景：每个 worker 进程写独立文件 asr_{pid}.log，
    # 避免多进程并发轮转同一文件竞争导致日志丢失/损坏。
    # 单 worker 时文件名仍是 asr_{pid}.log（pid 唯一），stdout 始终汇总。
    try:
        log_dir = Path(LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{LOG_FILE_PREFIX}_{os.getpid()}.log"
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            interval=1,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        # 轮转后的文件名格式：asr_{pid}.log.2026-05-12
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # 文件日志初始化失败不影响服务启动
        logger.warning(f"本地文件日志初始化失败: {e}，仅使用 stdout")

    return logger


def generate_request_id() -> str:
    """生成请求 ID。"""
    return str(uuid.uuid4())[:8]


def log_request(
    logger: logging.Logger,
    audio_duration_ms: int,
    vad_segments: int,
    asr_latency_ms: float,
    result_length: int,
):
    """记录请求处理日志。"""
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn="",
        lno=0,
        msg="请求处理完成",
        args=(),
        exc_info=None,
    )
    record.extra_fields = {
        "audio_duration_ms": audio_duration_ms,
        "vad_segments": vad_segments,
        "asr_latency_ms": round(asr_latency_ms, 2),
        "result_length": result_length,
    }
    logger.handle(record)


logger = setup_logger()
