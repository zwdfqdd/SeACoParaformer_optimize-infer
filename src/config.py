"""
服务配置管理

从环境变量加载配置，提供默认值。

MODEL_PRECISION 支持的完整精度矩阵：

  后端     取值                  各段精度(encoder/cif/decoder/bias)   说明
  ------  --------------------  -----------------------------------  ----------------------------
  自动     auto                  —                                    自动探测（见 get_model_precision）
  PT      pt                    —                                    原始 PyTorch 模型（转换环境，服务不支持，回退 onnx_fp32）
  ORT     onnx_fp32             —                                    ONNX Runtime fp32（v1 整体模型）
  ORT     onnx_int8             —                                    ONNX Runtime int8 动态量化（CPU）
  TRT     trt_fp32              fp32/fp32/fp32/fp32                   4 段全 fp32
  TRT     trt_fp16              fp16/fp16/fp16/fp16                   4 段全 fp16
  TRT     trt_int8              int8/int8/int8/int8                   4 段全 int8 QDQ（实测可跑，精度损失较大，不推荐线上）
  TRT     trt_int8_enc          int8/fp16/fp16/fp16                   仅 encoder int8（★线上推荐）

各 TRT 段精度也可用环境变量单独覆盖（优先级最高）：
    ENCODER_PRECISION / CIF_PRECISION / DECODER_PRECISION / BIAS_PRECISION
    取值：fp32 / fp16 / int8
"""

import os


# ============================================================
# TRT 各段精度组合（per-module）
# ============================================================
# int8 段实际加载时优先匹配 int8_qdq engine（QDQ Explicit 量化）
TRT_PRECISION_PROFILES = {
    "trt_fp32": {"encoder": "fp32", "cif": "fp32", "decoder": "fp32", "bias_encoder": "fp32"},
    "trt_fp16": {"encoder": "fp16", "cif": "fp16", "decoder": "fp16", "bias_encoder": "fp16"},
    # 全 int8（4 段都 QDQ 量化：encoder/decoder + cif/bias）
    # 实测 4 段 engine 可正常运行，但精度损失较大（cif cumsum + bias LSTM 量化），不推荐线上
    "trt_int8": {"encoder": "int8", "cif": "int8", "decoder": "int8", "bias_encoder": "int8"},
    # ★线上推荐：仅 encoder int8，其余 fp16（encoder 显存减半，热词精度保留，CER≈0）
    "trt_int8_enc": {"encoder": "int8", "cif": "fp16", "decoder": "fp16", "bias_encoder": "fp16"},
}

# 所有合法的 TRT profile 名称
TRT_PROFILE_NAMES = set(TRT_PRECISION_PROFILES.keys())

# ORT / PT 后端取值
ORT_PRECISIONS = {"onnx_fp32", "onnx_int8"}
PT_PRECISIONS = {"pt"}


class Settings:
    """服务配置。运行时可调参数从环境变量读取，固定参数硬编码。"""

    # 运行时可调
    WORKS: int = int(os.getenv("WORKS", "1"))
    BATCH: int = int(os.getenv("BATCH", "12"))
    PORT: int = int(os.getenv("PORT", "8080"))  # 容器内部固定端口（entrypoint 硬编码 8080，对外由 HOST_PORT 映射）
    BATCH_TIMEOUT: int = int(os.getenv("BATCH_TIMEOUT", "10"))  # 毫秒
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2000"))
    VERBOSE: bool = os.getenv("VERBOSE", "0") in ("1", "true", "True", "yes")

    # 音频最大时长限制（毫秒），超出返回 AUDIO_TOO_LONG(1005)。0 = 不限制。
    # 默认 2 小时（7200000ms）
    MAX_AUDIO_DURATION_MS: int = int(os.getenv("MAX_AUDIO_DURATION_MS", "7200000"))
    # 过载拒绝：并发信号量等待超时（秒），超时返回 SERVICE_BUSY(1007)。0 = 无限等待（不拒绝）。
    ACQUIRE_TIMEOUT: float = float(os.getenv("ACQUIRE_TIMEOUT", "5"))

    # 模型精度（见文件头精度矩阵）
    MODEL_PRECISION: str = os.getenv("MODEL_PRECISION", "auto")

    # 固定参数（模型已打包进镜像）
    MODEL_DIR: str = "./models"

    # ============================================================
    # Bucket / Batch 参数（单一数据源）
    #   - scheduler 运行时分桶 + batch 调度
    #   - asr_engine 预热
    #   - convert_trt 生成 TRT dynamic shape profile
    # 三处共用，保证转换 shape 与运行 shape 一致。
    # 启动前可通过环境变量调整，prepare_model 转 engine 时读取生效。
    # ============================================================
    # 桶边界（LFR 帧数）：2s→34, 4s→67, 8s→134
    BUCKET_SEQ_LENS: list[int] = [
        int(x) for x in os.getenv("BUCKET_SEQ_LENS", "34,67,134").split(",") if x.strip()
    ]
    # 合法 batch size
    VALID_BATCH_SIZES: list[int] = [
        int(x) for x in os.getenv("VALID_BATCH_SIZES", "1,2,4,8,12").split(",") if x.strip()
    ]
    # TRT profile 优化主力桶（opt point 的 seq_len）；默认取中间桶
    TRT_OPT_SEQ: int = int(os.getenv("TRT_OPT_SEQ", "67"))
    # 特征维度（LFR 后）
    FEAT_DIM: int = 560
    # encoder 输出维度
    HIDDEN_DIM: int = 512
    # 热词最大长度（token 数，用于 bias profile）
    MAX_HOTWORD_LEN: int = int(os.getenv("MAX_HOTWORD_LEN", "8"))

    # ============================================================
    # 热词管理参数（路径 A：SeACo 在线热词，单一数据源）
    #   - main.py 编码热词时按 MAX_HOTWORD_NUM 截断 + 告警
    #   - TRT bias/decoder profile 的热词数维度 max=MAX_HOTWORD_NUM, opt=OPT_HOTWORD_NUM
    #   - model.py ASF 过滤保留 top-NFILTER 注入 decoder
    # 显存上界由 MAX_HOTWORD_NUM 固定，engine 无需随词表规模重建。
    # ============================================================
    # SeACo 在线热词硬上限 / 路径切换点（客户端热词超限截断 Top-N + 告警；
    # 默认词表 ≤ 此值走路径 A SeACo，> 此值走路径 B Faiss）
    MAX_HOTWORD_NUM: int = int(os.getenv("MAX_HOTWORD_NUM", "256"))
    # TRT profile opt point 的热词数（主力工作点）
    OPT_HOTWORD_NUM: int = int(os.getenv("OPT_HOTWORD_NUM", "64"))
    # ASF 过滤后保留的 top-K 热词数（注入 decoder）
    NFILTER: int = int(os.getenv("NFILTER", "50"))

    # ============================================================
    # 默认词表 + 热更新参数（路径 A 预编码缓存 / 路径 B Faiss / 运行时热更新）
    # ============================================================
    # 服务端默认词表路径（容器本地文件，多 worker 共享）
    DEFAULT_HOTWORD_PATH: str = os.getenv(
        "DEFAULT_HOTWORD_PATH", os.path.join("models", "asr", "hotwords.txt")
    )
    # 是否开启词表热更新接口（/hotwords/reload 等）
    HOTWORD_RELOAD_ENABLED: bool = os.getenv("HOTWORD_RELOAD_ENABLED", "true") in (
        "1", "true", "True", "yes"
    )
    # 各 worker 轮询 version 文件间隔（秒），实现多进程最终一致收敛
    HOTWORD_POLL_INTERVAL: float = float(os.getenv("HOTWORD_POLL_INTERVAL", "5"))

    # ============================================================
    # 路径 B：Faiss 大词库后处理纠错参数
    # ============================================================
    # 滑窗大小（字数），逗号分隔
    FAISS_WINDOW_SIZES: list[int] = [
        int(x) for x in os.getenv("FAISS_WINDOW_SIZES", "2,3,4").split(",") if x.strip()
    ]
    # 召回 TopK
    FAISS_TOPK: int = int(os.getenv("FAISS_TOPK", "30"))
    # 重排权重：拼音 / 编辑距离
    FAISS_PINYIN_WEIGHT: float = float(os.getenv("FAISS_PINYIN_WEIGHT", "0.75"))
    FAISS_EDIT_WEIGHT: float = float(os.getenv("FAISS_EDIT_WEIGHT", "0.25"))
    # 三重联合判定阈值
    FAISS_SCORE_THRESHOLD: float = float(os.getenv("FAISS_SCORE_THRESHOLD", "0.85"))
    GAP_THRESHOLD: float = float(os.getenv("GAP_THRESHOLD", "0.05"))
    FINAL_SCORE_THRESHOLD: float = float(os.getenv("FINAL_SCORE_THRESHOLD", "0.88"))

    @classmethod
    def min_seq(cls) -> int:
        return min(cls.BUCKET_SEQ_LENS)

    @classmethod
    def max_seq(cls) -> int:
        return max(cls.BUCKET_SEQ_LENS)

    @classmethod
    def max_batch(cls) -> int:
        return max(cls.VALID_BATCH_SIZES)

    @classmethod
    def opt_batch(cls) -> int:
        """TRT opt point 的 batch（取合法 batch 中位附近，默认 4 或最大值较小者）。"""
        b = int(os.getenv("TRT_OPT_BATCH", "4"))
        return b if b in cls.VALID_BATCH_SIZES else cls.VALID_BATCH_SIZES[0]

    @classmethod
    def get_trt_profiles(cls, profile_type: str) -> dict:
        """根据 bucket/batch 参数动态生成 TRT dynamic shape profile。

        profile_type: encoder / cif / decoder / bias
        返回 {input_name: {"min":..., "opt":..., "max":...}}
        """
        mn, opt, mx = cls.min_seq(), cls.TRT_OPT_SEQ, cls.max_seq()
        ob, mb = cls.opt_batch(), cls.max_batch()
        fd, hd = cls.FEAT_DIM, cls.HIDDEN_DIM

        if profile_type == "encoder":
            return {
                "speech": {"min": (1, mn, fd), "opt": (ob, opt, fd), "max": (mb, mx, fd)},
            }
        if profile_type == "cif":
            return {
                "encoder_out": {"min": (1, mn, hd), "opt": (ob, opt, hd), "max": (mb, mx, hd)},
                "mask": {"min": (1, 1, mn), "opt": (ob, 1, opt), "max": (mb, 1, mx)},
            }
        if profile_type == "decoder":
            # token 数 ≤ seq_len，用同一范围；bias_embed 的热词数维度独立
            # 热词维度 = 实际热词数 + 1（[sos] 哨兵行），故 max 用 MAX_HOTWORD_NUM + 1
            max_hw = cls.MAX_HOTWORD_NUM + 1
            opt_hw = cls.OPT_HOTWORD_NUM
            return {
                "acoustic_embeds": {"min": (1, 2, hd), "opt": (ob, opt, hd), "max": (mb, mx, hd)},
                "encoder_out": {"min": (1, mn, hd), "opt": (ob, opt, hd), "max": (mb, mx, hd)},
                "bias_embed": {"min": (1, 1, hd), "opt": (1, opt_hw, hd), "max": (mb, max_hw, hd)},
            }
        if profile_type == "bias":
            # 热词矩阵行数 = 实际热词数 + 1（[sos] 哨兵），故 max 用 MAX_HOTWORD_NUM + 1
            max_hw = cls.MAX_HOTWORD_NUM + 1
            opt_hw = cls.OPT_HOTWORD_NUM
            return {
                "hotword": {"min": (1, 1), "opt": (opt_hw, 4), "max": (max_hw, cls.MAX_HOTWORD_LEN)},
            }
        raise ValueError(f"未知 profile_type: {profile_type}")

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

        - 显式指定：直接返回
        - auto 模式：
            GPU：trt_int8_enc → trt_fp16 → trt_fp32 → onnx_fp32
            CPU：onnx_int8 → onnx_fp32
        """
        precision = cls.MODEL_PRECISION.lower()

        if precision in TRT_PROFILE_NAMES or precision in ORT_PRECISIONS or precision in PT_PRECISIONS:
            return precision

        # auto 或未知取值 → 自动探测
        device = cls.get_device()
        if device == "cuda":
            # 优先线上推荐方案：encoder int8 + 其余 fp16
            if cls._has_trt_profile("trt_int8_enc"):
                return "trt_int8_enc"
            if cls._has_trt_profile("trt_fp16"):
                return "trt_fp16"
            if cls._has_trt_profile("trt_fp32"):
                return "trt_fp32"
            return "onnx_fp32"

        # CPU
        int8_path = os.path.join(cls.MODEL_DIR, "asr", "int8", "model.onnx")
        if os.path.exists(int8_path):
            return "onnx_int8"
        return "onnx_fp32"

    @classmethod
    def get_inference_backend(cls) -> str:
        """返回推理后端：'trt' / 'ort' / 'pt'。"""
        precision = cls.get_model_precision()
        if precision.startswith("trt_"):
            return "trt"
        if precision in PT_PRECISIONS:
            return "pt"
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
    def _ort_subdir(cls) -> str:
        """ORT 模型子目录（onnx_fp32→fp32, onnx_int8→int8）。"""
        precision = cls.get_model_precision()
        if precision == "onnx_int8":
            return "int8"
        # onnx_fp32 / pt / trt_*（TRT 回退用 fp32）/ 其他 → fp32
        return "fp32"

    @classmethod
    def get_asr_model_path(cls) -> str:
        """ORT 主模型路径（v1 整体模型 model.onnx）。"""
        return os.path.join(cls.MODEL_DIR, "asr", cls._ort_subdir(), "model.onnx")

    @classmethod
    def get_asr_bias_model_path(cls) -> str:
        """ORT 热词 bias encoder 路径（v1 整体模型 model_eb.onnx）。"""
        return os.path.join(cls.MODEL_DIR, "asr", cls._ort_subdir(), "model_eb.onnx")

    # --- TRT 4 段 engine 路径（v2 主路径，支持混合精度） ---
    @classmethod
    def get_trt_precision_map(cls) -> dict[str, str]:
        """获取 4 段各自的精度（fp32/fp16/int8）。

        优先级：环境变量 {MODULE}_PRECISION > MODEL_PRECISION 对应的 profile。
        """
        precision = cls.get_model_precision()
        base = TRT_PRECISION_PROFILES.get(precision, TRT_PRECISION_PROFILES["trt_fp16"]).copy()

        # 环境变量单段覆盖
        env_map = {
            "encoder": os.getenv("ENCODER_PRECISION"),
            "cif": os.getenv("CIF_PRECISION"),
            "decoder": os.getenv("DECODER_PRECISION"),
            "bias_encoder": os.getenv("BIAS_PRECISION"),
        }
        for module, val in env_map.items():
            if val:
                base[module] = val.lower()
        return base

    @classmethod
    def get_trt_engine_paths(cls) -> dict[str, str | None]:
        """获取 TRT 4 段 engine 路径字典（按 per-module 精度）。

        返回缺失的段为 None。
        """
        precision = cls.get_model_precision()
        if not precision.startswith("trt_"):
            return {"encoder": None, "cif": None, "decoder": None, "bias_encoder": None}

        prec_map = cls.get_trt_precision_map()
        return {
            module: cls._find_trt_engine(prec_map[module], module)
            for module in ("encoder", "cif", "decoder", "bias_encoder")
        }

    @classmethod
    def _has_trt_profile(cls, profile_name: str) -> bool:
        """检查指定 profile 的 encoder/cif/decoder 三段 engine 是否齐全。"""
        prec_map = TRT_PRECISION_PROFILES.get(profile_name)
        if not prec_map:
            return False
        for module in ("encoder", "cif", "decoder"):
            if cls._find_trt_engine(prec_map[module], module) is None:
                return False
        return True

    @classmethod
    def _find_trt_engine(cls, precision: str, module: str) -> str | None:
        """
        查找 engine 文件。

        命名规则：{gpu}_{module}_{precision}[_qdq].engine
        - int8 段优先匹配带 _qdq 后缀的（QDQ Explicit 量化产物）
        - 严格匹配文件名结构，避免 'encoder' 误匹配 'bias_encoder'
        """
        trt_dir = os.path.join(cls.MODEL_DIR, "asr", "trt")
        if not os.path.isdir(trt_dir):
            return None

        gpu_name = cls._get_gpu_name()

        def _match(filename: str, mod: str, prec: str) -> bool:
            if not filename.endswith(".engine"):
                return False
            stem = filename[:-7]  # 去 .engine
            # int8 允许 _qdq 后缀
            if prec == "int8":
                if stem.endswith("_int8_qdq"):
                    stem_no_prec = stem[:-len("_int8_qdq")]
                elif stem.endswith("_int8"):
                    stem_no_prec = stem[:-len("_int8")]
                else:
                    return False
            else:
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

        # int8 优先匹配 _qdq；其余精度正常匹配
        # 1) 当前 GPU 名称 + _qdq（仅 int8）
        if precision == "int8":
            for f in os.listdir(trt_dir):
                if f.endswith("_qdq.engine") and _match(f, module, precision) and gpu_name in f.lower():
                    return os.path.join(trt_dir, f)
            for f in os.listdir(trt_dir):
                if f.endswith("_qdq.engine") and _match(f, module, precision):
                    return os.path.join(trt_dir, f)

        # 2) 当前 GPU 名称匹配
        for f in os.listdir(trt_dir):
            if _match(f, module, precision) and gpu_name in f.lower():
                return os.path.join(trt_dir, f)
        # 3) 不区分 GPU 名称兜底
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
