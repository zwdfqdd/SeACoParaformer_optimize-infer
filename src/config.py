"""
服务配置管理

从环境变量加载配置，提供默认值。

模型精度选择策略（MODEL_PRECISION）：
- auto：GPU 优先 TRT engine → 回退 ORT fp32；CPU 优先 int8 → 回退 fp32
- fp32：强制 ORT fp32
- int8：强制 ORT int8（仅 CPU）
- trt_fp16：强制 TensorRT fp16 engine
- trt_int8：强制 TensorRT int8 engine
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

    # 模型精度：auto / fp32 / int8 / trt_fp16 / trt_int8
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
        - MODEL_PRECISION=fp32 → ORT fp32
        - MODEL_PRECISION=int8 → ORT int8（仅 CPU）
        - MODEL_PRECISION=trt_fp16 → TensorRT fp16
        - MODEL_PRECISION=trt_int8 → TensorRT int8
        - MODEL_PRECISION=auto →
            GPU: TRT engine 存在 → trt_fp16，否则 fp32
            CPU: int8 存在 → int8，否则 fp32
        """
        precision = cls.MODEL_PRECISION.lower()

        if precision in ("fp32", "int8", "trt_fp16", "trt_int8"):
            return precision

        # auto 模式
        device = cls.get_device()
        if device == "cuda":
            # GPU：优先 TRT fp16 engine
            trt_path = cls._find_trt_engine("fp16")
            if trt_path:
                return "trt_fp16"
            return "fp32"

        # CPU：优先 int8
        int8_path = os.path.join(cls.MODEL_DIR, "asr", "int8", "model.onnx")
        if os.path.exists(int8_path):
            return "int8"
        return "fp32"

    @classmethod
    def get_inference_backend(cls) -> str:
        """返回推理后端类型：'ort' 或 'trt'。"""
        precision = cls.get_model_precision()
        if precision.startswith("trt_"):
            return "trt"
        return "ort"

    @classmethod
    def get_asr_model_path(cls) -> str:
        """ASR 主模型路径（ORT 模式）。"""
        precision = cls.get_model_precision()
        if precision.startswith("trt_"):
            # TRT 模式下仍需要 ORT 路径作为回退
            return os.path.join(cls.MODEL_DIR, "asr", "fp32", "model.onnx")
        return os.path.join(cls.MODEL_DIR, "asr", precision, "model.onnx")

    @classmethod
    def get_asr_trt_engine_path(cls) -> str | None:
        """ASR TRT engine 路径（自动匹配当前 GPU）。"""
        precision = cls.get_model_precision()
        if not precision.startswith("trt_"):
            return None
        trt_precision = precision.replace("trt_", "")  # fp16 或 int8
        return cls._find_trt_engine(trt_precision)

    @classmethod
    def get_bias_trt_engine_path(cls) -> str | None:
        """Bias encoder TRT engine 路径。"""
        precision = cls.get_model_precision()
        if not precision.startswith("trt_"):
            return None
        trt_precision = precision.replace("trt_", "")
        return cls._find_trt_engine(trt_precision, model_name="model_eb")

    @classmethod
    def get_asr_bias_model_path(cls) -> str:
        """ASR 热词 bias encoder 模型路径（ORT 模式）。"""
        precision = cls.get_model_precision()
        if precision.startswith("trt_"):
            return os.path.join(cls.MODEL_DIR, "asr", "fp32", "model_eb.onnx")
        return os.path.join(cls.MODEL_DIR, "asr", precision, "model_eb.onnx")

    @classmethod
    def get_vad_model_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "vad", "silero_vad.onnx")

    @classmethod
    def get_asr_config_dir(cls) -> str:
        """ASR 配置文件目录（am.mvn, tokens.json 等）。"""
        return os.path.join(cls.MODEL_DIR, "asr")

    @classmethod
    def _find_trt_engine(cls, precision: str, model_name: str = "model") -> str | None:
        """
        查找匹配当前 GPU 的 TRT engine 文件。

        搜索路径：models/asr/trt/
        命名规则：{gpu_name}_{model_name}_{precision}.engine
        回退：{model_name}_{precision}.engine（不含 GPU 名称）
        """
        trt_dir = os.path.join(cls.MODEL_DIR, "asr", "trt")
        if not os.path.isdir(trt_dir):
            return None

        # 获取 GPU 名称
        gpu_name = cls._get_gpu_name()

        # 精确匹配：{gpu}_{model}_{precision}.engine
        exact_path = os.path.join(trt_dir, f"{gpu_name}_{model_name}_{precision}.engine")
        if os.path.exists(exact_path):
            return exact_path

        # 通用匹配：{model}_{precision}.engine
        generic_path = os.path.join(trt_dir, f"{model_name}_{precision}.engine")
        if os.path.exists(generic_path):
            return generic_path

        # 模糊匹配：任何包含 model_name 和 precision 的 engine
        for f in os.listdir(trt_dir):
            if f.endswith(".engine") and model_name in f and precision in f:
                return os.path.join(trt_dir, f)

        return None

    @classmethod
    def _get_gpu_name(cls) -> str:
        """获取简化的 GPU 名称。"""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0).lower()
                for prefix in ["nvidia ", "geforce ", "rtx ", "tesla "]:
                    name = name.replace(prefix, "")
                return name.strip().replace(" ", "_")
        except ImportError:
            pass
        return "unknown"


settings = Settings()
