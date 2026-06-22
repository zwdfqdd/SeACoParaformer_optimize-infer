"""
服务配置管理

从环境变量加载配置，提供默认值。

模型精度选择策略（MODEL_PRECISION）：
- auto：GPU 优先 TRT engine → 回退 ORT fp32；CPU 优先 int8 → 回退 fp32
- fp32：强制 ORT fp32
- int8：强制 ORT int8（仅 CPU）
- trt_fp32：TensorRT 4 段 fp32 engine
- trt_fp16：TensorRT 4 段 fp16 engine（推荐，v2 阶段 1 最终方案）
- trt_int8：TensorRT 4 段 INT8 engine（v2 阶段 2 规划）
"""

import os


class Settings:
    """服务配置。运行时可调参数从环境变量读取，固定参数硬编码。"""

    # 运行时可调
    WORKS: int = int(os.getenv("WORKS", "1"))
    BATCH: int = int(os.getenv("BATCH", "12"))
    PORT: int = int(os.getenv("PORT", "8080"))
    BATCH_TIMEOUT: int = int(os.getenv("BATCH_TIMEOUT", "10"))  # 毫秒
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2000"))
    VERBOSE: bool = os.getenv("VERBOSE", "0") in ("1", "true", "True", "yes")

    # 模型精度：auto / fp32 / int8 / trt_fp32 / trt_fp16 / trt_int8
    MODEL_PRECISION: str = os.getenv("MODEL_PRECISION", "auto")

    # 固定参数（模型已打包进镜像）
    MODEL_DIR: str = "./models"

    # ============================================================
    # 设备 / 精度选择
    # ============================================================
    @classmethod
    def get_device(cls) -> str:
        """检测推理设备，有 GPU 用 GPU，否则用 CPU。"""
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

        - 显式指定（fp32/int8/trt_fp32/trt_fp16/trt_int8）：直接返回
        - auto 模式：
            GPU + 存在 TRT 4 段 fp16 engine → trt_fp16
            GPU                              → fp32（ORT）
            CPU + 存在 int8 模型             → int8
            CPU                              → fp32（ORT）
        """
        precision = cls.MODEL_PRECISION.lower()

        if precision in ("fp32", "int8", "trt_fp32", "trt_fp16", "trt_int8"):
            return precision

        device = cls.get_device()
        if device == "cuda":
            # 优先 TRT fp16 4 段架构
            if cls._has_trt_split_engines("fp16"):
                return "trt_fp16"
            return "fp32"

        # CPU
        int8_path = os.path.join(cls.MODEL_DIR, "asr", "int8", "model.onnx")
        if os.path.exists(int8_path):
            return "int8"
        return "fp32"

    @classmethod
    def get_inference_backend(cls) -> str:
        """返回推理后端：'ort' 或 'trt'。"""
        precision = cls.get_model_precision()
        if precision.startswith("trt_"):
            return "trt"
        return "ort"

    # ============================================================
    # 模型路径
    # ============================================================
    @classmethod
    def get_asr_config_dir(cls) -> str:
        """ASR 配置文件目录（am.mvn, tokens.json）。"""
        return os.path.join(cls.MODEL_DIR, "asr")

    @classmethod
    def get_vad_model_path(cls) -> str:
        return os.path.join(cls.MODEL_DIR, "vad", "silero_vad.onnx")

    # --- ORT 路径（v1 整体模型，向后兼容） ---
    @classmethod
    def get_asr_model_path(cls) -> str:
        """ORT 主模型路径（v1 整体模型 model.onnx）。"""
        precision = cls.get_model_precision()
        if precision.startswith("trt_"):
            # TRT 模式下 ORT 路径作为回退
            return os.path.join(cls.MODEL_DIR, "asr", "fp32", "model.onnx")
        return os.path.join(cls.MODEL_DIR, "asr", precision, "model.onnx")

    @classmethod
    def get_asr_bias_model_path(cls) -> str:
        """ORT 热词 bias encoder 路径（v1 整体模型 model_eb.onnx）。"""
        precision = cls.get_model_precision()
        if precision.startswith("trt_"):
            return os.path.join(cls.MODEL_DIR, "asr", "fp32", "model_eb.onnx")
        return os.path.join(cls.MODEL_DIR, "asr", precision, "model_eb.onnx")

    # --- TRT 4 段 engine 路径（v2 阶段 1 主路径） ---
    @classmethod
    def get_trt_engine_paths(cls) -> dict[str, str | None]:
        """获取 TRT 4 段 engine 路径字典。

        返回：{
            "encoder": .../{gpu}_encoder_{prec}.engine,
            "cif":     .../{gpu}_cif_{prec}.engine,
            "decoder": .../{gpu}_decoder_{prec}.engine,
            "bias_encoder": .../{gpu}_bias_encoder_{prec}.engine,
        }
        缺失的段返回 None。
        """
        precision = cls.get_model_precision()
        if not precision.startswith("trt_"):
            return {"encoder": None, "cif": None, "decoder": None, "bias_encoder": None}

        trt_prec = precision.replace("trt_", "")  # fp32 / fp16 / int8
        return {
            "encoder": cls._find_trt_engine(trt_prec, "encoder"),
            "cif": cls._find_trt_engine(trt_prec, "cif"),
            "decoder": cls._find_trt_engine(trt_prec, "decoder"),
            "bias_encoder": cls._find_trt_engine(trt_prec, "bias_encoder"),
        }

    @classmethod
    def _has_trt_split_engines(cls, precision: str) -> bool:
        """检查 4 段 TRT engine（除 bias_encoder 外）是否齐全。"""
        for module in ("encoder", "cif", "decoder"):
            if cls._find_trt_engine(precision, module) is None:
                return False
        return True

    @classmethod
    def _find_trt_engine(cls, precision: str, module: str) -> str | None:
        """
        按 {gpu}_{module}_{precision}.engine 命名规则查找 engine 文件。

        严格匹配文件名结构，避免 'encoder' 误匹配 'bias_encoder'。
        """
        trt_dir = os.path.join(cls.MODEL_DIR, "asr", "trt")
        if not os.path.isdir(trt_dir):
            return None

        gpu_name = cls._get_gpu_name()

        def _match(filename: str, mod: str, prec: str) -> bool:
            if not filename.endswith(".engine"):
                return False
            stem = filename[:-7]  # 去 .engine
            suffix = f"_{prec}"
            if not stem.endswith(suffix):
                return False
            stem_no_prec = stem[:-len(suffix)]
            if not stem_no_prec.endswith(f"_{mod}"):
                return False
            # encoder vs bias_encoder 区分
            if mod == "encoder":
                base = stem_no_prec[:-(len(mod) + 1)]
                if base.endswith("_bias") or base == "bias":
                    return False
            return True

        # 优先匹配当前 GPU 名称
        for f in os.listdir(trt_dir):
            if _match(f, module, precision) and gpu_name in f.lower():
                return os.path.join(trt_dir, f)
        # 不区分 GPU 名称的兜底匹配
        for f in os.listdir(trt_dir):
            if _match(f, module, precision):
                return os.path.join(trt_dir, f)
        return None

    @classmethod
    def _get_gpu_name(cls) -> str:
        """简化的 GPU 名称（用于 engine 文件名匹配）。"""
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
