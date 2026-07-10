"""
请求/响应数据模型定义
"""

from pydantic import BaseModel, Field


class ASRRequest(BaseModel):
    """ASR 识别请求。"""
    base64: str = Field(..., description="WAV 16kHz 单声道音频的 Base64 编码（必填）")
    article_url: str | None = Field(
        default=None,
        description="原始音频文件的 URL（可选，服务端原样透传到响应中，用于业务侧追踪）",
        examples=["https://cdn.example.com/audio/xxx.wav"],
    )
    hotwords: list[str] | None = Field(
        default=None,
        description="热词列表，可选参数",
        examples=[["张三", "李四"]],
    )


class ASRWord(BaseModel):
    """字级识别结果（含时间戳）。"""
    text: str = Field(..., description="字符（中文单字或英文 BPE subword）")
    timestamp: list[float] = Field(
        ...,
        description="[起始秒, 结束秒]，源自 CIF fire 位置对应的原始音频时间轴",
        examples=[[0.12, 0.24]],
    )


class ASRSegment(BaseModel):
    """单段识别结果（与外部标准 asr 数组格式对齐）。

    默认粒度为 VAD 切段；开启句子级时间戳（ENABLE_SENTENCE_TIMESTAMP=true，且
    ENABLE_WORD_TIMESTAMP=true）后，每项粒度变为「一句话」：text 为带标点句子，
    timestamp 为句子起止（由句内字级时间戳定位），words 为该句字级时间戳。
    """
    idx: int = Field(..., description="段/句序号（从 0 起）")
    slid: str = Field(
        default="",
        description="语种识别结果（当前未实现，固定空字符串）",
    )
    text: str = Field(..., description="该段识别文本")
    speaker: str = Field(
        default="",
        description="说话人识别结果（当前未实现，固定空字符串）",
    )
    timestamp: list[float] = Field(
        ...,
        description="[起始秒, 结束秒]，源自原始音频时间轴",
        examples=[[0.0, 12.0]],
    )
    words: list[ASRWord] = Field(
        default_factory=list,
        description="字级时间戳数组（可选，由 CIF fire 位置反推得到）",
    )


class ASRResponse(BaseModel):
    """ASR 识别成功响应（与外部标准结构对齐）。"""
    code: int = Field(default=0, description="业务状态码，0 表示成功")
    article_url: str | None = Field(
        default=None,
        description="原样透传请求中的 article_url，未传时为 null",
    )
    istar_asr: str = Field(
        ...,
        description="全文拼接结果（各段 text 顺序拼接）",
    )
    asr: list[ASRSegment] = Field(
        ...,
        description="分段识别结果数组，按时间顺序排列",
    )
    message: str = Field(
        default="",
        description="提示信息（正常识别为空；如 VAD 后无有效语音则提示“音频内容为空”）",
    )


class ErrorResponse(BaseModel):
    """ASR 识别失败响应。"""
    code: int = Field(..., description="业务错误码")
    error: str = Field(..., description="错误码名称")
    message: str = Field(..., description="错误描述")


class HealthResponse(BaseModel):
    """健康检查响应。

    status 语义：
        ok       — 模型已加载且运行时健康
        degraded — 模型未加载 / 运行时连续推理失败超阈值（GPU 卡死）/ 加载阶段静默降级
    runtime 字段暴露运行时健康明细（连续失败数、累计成功/失败、静默降级原因），
    供探针与运维判定实例是否需摘除。
    """
    status: str = Field(default="ok")
    device: str = Field(..., description="当前推理设备")
    models_loaded: bool = Field(..., description="模型是否已加载")
    runtime: dict | None = Field(
        default=None,
        description="运行时健康明细（backend/连续失败数/累计成功失败/静默降级原因等）",
    )


class HotwordReloadRequest(BaseModel):
    """词表热更新请求。words 与 reload_from_file 二选一。"""
    words: list[str] | None = Field(
        default=None, description="新词表内容（与 reload_from_file 二选一）"
    )
    reload_from_file: bool = Field(
        default=False, description="true 表示重读磁盘 hotwords.txt"
    )
    expected_version: int | None = Field(
        default=None, description="乐观并发版本号，与当前不符则拒绝"
    )


class HotwordReloadResponse(BaseModel):
    """词表热更新响应。"""
    code: int = Field(default=0)
    version: int = Field(..., description="更新后的词表版本号")
    md5: str = Field(..., description="词表内容哈希")
    count: int = Field(..., description="有效词条数")
    route: str = Field(..., description="生效路径 A（SeACo）或 B（Faiss）")
    message: str = Field(default="")


class HotwordStatusResponse(BaseModel):
    """词表状态响应。"""
    version: int = Field(default=0)
    md5: str = Field(default="")
    count: int = Field(default=0)
    route: str | None = Field(default=None)
    loaded_at: str | None = Field(default=None)
    cache_ready: bool = Field(default=False)
