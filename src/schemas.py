"""
请求/响应数据模型定义
"""

from pydantic import BaseModel, Field


class ASRRequest(BaseModel):
    """ASR 识别请求。"""
    b64: str = Field(..., description="WAV 16kHz 单声道音频的 Base64 编码")
    hotwords: list[str] | None = Field(
        default=None,
        description="热词列表，可选参数",
        examples=[["张三", "李四"]],
    )


class SegmentDetail(BaseModel):
    """单段识别结果。"""
    text: str = Field(..., description="该段识别文本")
    start_ms: int = Field(..., description="原始音频中的起始时间（毫秒）")
    end_ms: int = Field(..., description="原始音频中的结束时间（毫秒）")


class ASRResponse(BaseModel):
    """ASR 识别成功响应。"""
    code: int = Field(default=0, description="业务状态码，0 表示成功")
    text: str = Field(..., description="全文拼接结果")
    detail: dict[str, SegmentDetail] = Field(
        ...,
        description="分段识别结果，key 为段序号",
    )


class ErrorResponse(BaseModel):
    """ASR 识别失败响应。"""
    code: int = Field(..., description="业务错误码")
    error: str = Field(..., description="错误码名称")
    message: str = Field(..., description="错误描述")


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str = Field(default="ok")
    device: str = Field(..., description="当前推理设备")
    models_loaded: bool = Field(..., description="模型是否已加载")
