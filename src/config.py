"""
服务配置管理

从环境变量加载配置，提供默认值。
"""

import os


class Settings:
    """服务配置，从环境变量读取。"""

    WORKS: int = int(os.getenv("WORKS", "1"))
    BATCH: int = int(os.getenv("BATCH", "1"))
    PORT: int = int(os.getenv("PORT", "30960"))
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")
    DEVICE: str = os.getenv("DEVICE", "auto")
    BATCH_TIMEOUT: int = int(os.getenv("BATCH_TIMEOUT", "10"))  # 毫秒
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_BATCH_DURATION: int = int(os.getenv("MAX_BATCH_DURATION", "30"))  # 秒
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2000"))

    @classmethod
    def get_device(cls) -> str:
        """根据 DEVICE 配置和硬件环境确定实际推理设备。"""
        if cls.DEVICE != "auto":
            return cls.DEVICE
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @classmethod
    def get_asr_model_path(cls) -> str:
        """ASR 主模型路径（fp16）。"""
        return os.path.join(cls.MODEL_DIR, "asr", "fp16", "model.onnx")

    @classmethod
    def get_asr_bias_model_path(cls) -> str:
        """ASR 热词 bias encoder 模型路径（fp16）。"""
        return os.path.join(cls.MODEL_DIR, "asr", "fp16", "model_eb.onnx")

    @classmethod
    def get_vad_model_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "vad", "silero_vad.onnx")


settings = Settings()
