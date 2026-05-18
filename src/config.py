"""
服务配置管理

从环境变量加载配置，提供默认值。

模型精度选择策略：
- GPU 环境：自动使用 fp32 模型（CUDA 推理）
- CPU 环境：优先使用 int8 模型（若存在），否则回退 fp32
- 可通过 MODEL_PRECISION 环境变量强制指定：fp32 / int8
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

    # 模型精度：auto / fp32 / int8
    # auto: GPU→fp32, CPU→int8(若存在)否则fp32
    MODEL_PRECISION: str = os.getenv("MODEL_PRECISION", "auto")

    # 固定参数（模型已打包进镜像，路径固定）
    MODEL_DIR: str = "./models"

    @classmethod
    def get_device(cls) -> str:
        """
        自动检测推理设备：有 GPU 用 GPU，没有用 CPU。
        
        检测逻辑：通过 torch.cuda.is_available() 验证 CUDA 驱动是否真正可用，
        避免 onnxruntime-gpu 包含 CUDAExecutionProvider 但无驱动时 segfault。
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @classmethod
    def get_model_precision(cls) -> str:
        """
        确定实际使用的模型精度。

        策略：
        - MODEL_PRECISION=fp32 → 强制 fp32
        - MODEL_PRECISION=int8 → 强制 int8（仅 CPU）
        - MODEL_PRECISION=auto → GPU 用 fp32，CPU 优先 int8
        """
        precision = cls.MODEL_PRECISION.lower()

        if precision == "fp32":
            return "fp32"
        if precision == "int8":
            return "int8"

        # auto 模式
        device = cls.get_device()
        if device == "cuda":
            return "fp32"

        # CPU 模式：优先 int8
        int8_path = os.path.join(cls.MODEL_DIR, "asr", "int8", "model.onnx")
        if os.path.exists(int8_path):
            return "int8"
        return "fp32"

    @classmethod
    def get_asr_model_path(cls) -> str:
        """ASR 主模型路径（根据精度策略自动选择）。"""
        precision = cls.get_model_precision()
        return os.path.join(cls.MODEL_DIR, "asr", precision, "model.onnx")

    @classmethod
    def get_asr_bias_model_path(cls) -> str:
        """ASR 热词 bias encoder 模型路径。"""
        precision = cls.get_model_precision()
        return os.path.join(cls.MODEL_DIR, "asr", precision, "model_eb.onnx")

    @classmethod
    def get_vad_model_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "vad", "silero_vad.onnx")

    @classmethod
    def get_asr_config_dir(cls) -> str:
        """ASR 配置文件目录（am.mvn, tokens.json 等）。"""
        return os.path.join(cls.MODEL_DIR, "asr")


settings = Settings()
