"""
业务错误码定义

错误码与 HTTP Status 对应：
    1000/1001/1005 → 400（客户端可修复）
    1002/1003/1004/1006 → 500（服务端内部异常）
    1007 → 503（服务不可用）
"""

from enum import IntEnum


class ErrorCode(IntEnum):
    """业务错误码。"""
    SUCCESS = 0
    INPUT_PARAM_FAILED = 1000
    DECODE_FAILED = 1001
    VAD_SEGMENT_ERROR = 1002
    AUDIO_SEGMENT_ERROR = 1003
    ASR_INFER_FAILED = 1004
    AUDIO_TOO_LONG = 1005
    MODEL_LOAD_FAILED = 1006
    SERVICE_BUSY = 1007


# 错误码 → HTTP Status 映射
ERROR_HTTP_STATUS: dict[int, int] = {
    ErrorCode.INPUT_PARAM_FAILED: 400,
    ErrorCode.DECODE_FAILED: 400,
    ErrorCode.AUDIO_TOO_LONG: 400,
    ErrorCode.VAD_SEGMENT_ERROR: 500,
    ErrorCode.AUDIO_SEGMENT_ERROR: 500,
    ErrorCode.ASR_INFER_FAILED: 500,
    ErrorCode.MODEL_LOAD_FAILED: 500,
    ErrorCode.SERVICE_BUSY: 503,
}

# 错误码 → 默认错误消息
ERROR_MESSAGES: dict[int, str] = {
    ErrorCode.INPUT_PARAM_FAILED: "输入参数错误",
    ErrorCode.DECODE_FAILED: "音频解码失败，请确认为16kHz单声道WAV格式",
    ErrorCode.VAD_SEGMENT_ERROR: "VAD 模型推理异常",
    ErrorCode.AUDIO_SEGMENT_ERROR: "音频切段处理异常",
    ErrorCode.ASR_INFER_FAILED: "ASR 模型推理失败",
    ErrorCode.AUDIO_TOO_LONG: "音频超出最大时长限制",
    ErrorCode.MODEL_LOAD_FAILED: "模型加载失败",
    ErrorCode.SERVICE_BUSY: "服务繁忙，请稍后重试",
}


class ASRException(Exception):
    """ASR 服务业务异常。"""

    def __init__(self, code: ErrorCode, message: str | None = None):
        self.code = code
        self.message = message or ERROR_MESSAGES.get(code, "未知错误")
        self.http_status = ERROR_HTTP_STATUS.get(code, 500)
        super().__init__(self.message)
