"""
服务配置管理

从环境变量加载配置，提供默认值。
"""

import os


class Settings:
    """服务配置。运行时可调参数从环境变量读取，固定参数硬编码。"""

    # 运行时可调（通过环境变量 / docker-compose）
    WORKS: int = int(os.getenv("WORKS", "1"))
    BATCH: int = int(os.getenv("BATCH", "12"))
    PORT: int = int(os.getenv("PORT", "8080"))
    BATCH_TIMEOUT: int = int(os.getenv("BATCH_TIMEOUT", "10"))  # 毫秒
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2000"))
    VERBOSE: bool = os.getenv("VERBOSE", "0") in ("1", "true", "True", "yes")

    # 固定参数（模型已打包进镜像，路径固定）
    MODEL_DIR: str = "./models"

    @classmethod
    def get_device(cls) -> str:
        """自动检测推理设备：有 GPU 用 GPU，没有用 CPU。"""
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
        """ASR 主模型路径（fp32）。"""
        return os.path.join(cls.MODEL_DIR, "asr", "fp32", "model.onnx")

    @classmethod
    def get_asr_bias_model_path(cls) -> str:
        """ASR 热词 bias encoder 模型路径（fp32）。"""
        return os.path.join(cls.MODEL_DIR, "asr", "fp32", "model_eb.onnx")

    @classmethod
    def get_vad_model_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "vad", "silero_vad.onnx")

    @classmethod
    def get_asr_config_dir(cls) -> str:
        """ASR 配置文件目录（am.mvn, tokens.json 等）。"""
        return os.path.join(cls.MODEL_DIR, "asr")


settings = Settings()
