"""
服务配置管理

从环境变量加载配置，提供默认值。

MODEL_PRECISION 支持的完整精度矩阵：

  后端     取值                  各段精度(encoder/cif/decoder/bias/timestamp)  说明
  ------  --------------------  --------------------------------------------  ----------------------------
  自动     auto                  —                                            自动探测（见 get_model_precision）
  PT      pt                    —                                            原始 PyTorch 模型（GPU 优先/CPU 兜底）
  ORT     onnx_fp32             —                                            ONNX Runtime fp32（分段串联）
  ORT     onnx_int8             —                                            ONNX Runtime int8 动态量化（CPU）
  TRT     trt_fp32              fp32/fp32/fp32/fp32/fp32                      全 fp32
  TRT     trt_fp16              fp16/fp16/fp16/fp16/fp16                      全 fp16
  TRT     trt_int8              int8/int8/int8/int8/fp16                      encoder..bias int8 QDQ + timestamp fp16（精度损失较大，不推荐线上）
  TRT     trt_int8_enc          int8/fp16/fp16/fp16/fp16                      仅 encoder int8（★线上推荐）

各段精度可用环境变量单独覆盖（优先级最高）：
    ENCODER_PRECISION / CIF_PRECISION / DECODER_PRECISION / BIAS_PRECISION / TIMESTAMP_PRECISION
    取值：fp32 / fp16 / int8（timestamp 仅 fp32/fp16，含 BLSTM 不量化，int8 会回退 fp16）
"""

import os


# ============================================================
# TRT 各段精度组合（per-module）
# ============================================================
# int8 段实际加载时优先匹配 int8_qdq engine（QDQ Explicit 量化）
# timestamp（第 5 段，字级时间戳）精度说明：
#   含双向 BLSTM，int8 量化精度损失大且 TRT 对 LSTM int8 支持差，故不支持 int8。
#   只支持 fp32 / fp16：trt_fp32 profile 下用 fp32，其余（含 int8 系列）一律 fp16。
#   可用 TIMESTAMP_PRECISION 环境变量强制覆盖（仅 fp32/fp16 生效，int8 会被拒绝回退 fp16）。
TRT_PRECISION_PROFILES = {
    "trt_fp32": {"encoder": "fp32", "cif": "fp32", "decoder": "fp32", "bias_encoder": "fp32", "timestamp": "fp32"},
    "trt_fp16": {"encoder": "fp16", "cif": "fp16", "decoder": "fp16", "bias_encoder": "fp16", "timestamp": "fp16"},
    # 全 int8（4 段都 QDQ 量化：encoder/decoder + cif/bias；timestamp 不量化，保持 fp16）
    # 实测 4 段 engine 可正常运行，但精度损失较大（cif cumsum + bias LSTM 量化），不推荐线上
    "trt_int8": {"encoder": "int8", "cif": "int8", "decoder": "int8", "bias_encoder": "int8", "timestamp": "fp16"},
    # ★线上推荐：仅 encoder int8，其余 fp16（encoder 显存减半，热词精度保留，CER≈0）
    "trt_int8_enc": {"encoder": "int8", "cif": "fp16", "decoder": "fp16", "bias_encoder": "fp16", "timestamp": "fp16"},
}

# timestamp 段允许的精度（BLSTM 不量化）
TIMESTAMP_ALLOWED_PRECISIONS = {"fp16", "fp32"}

# 所有合法的 TRT profile 名称
TRT_PROFILE_NAMES = set(TRT_PRECISION_PROFILES.keys())

# ORT / PT 后端取值
ORT_PRECISIONS = {"onnx_fp32", "onnx_int8"}
PT_PRECISIONS = {"pt"}


class Settings:
    """服务配置。运行时可调参数从环境变量读取，固定参数硬编码。"""

    # 运行时可调（默认取 A10 24GB 实测最优；小显存需显式调小 WORKERS 防 OOM）
    WORKERS: int = int(os.getenv("WORKERS", "11"))
    BATCH: int = int(os.getenv("BATCH", "12"))
    PORT: int = int(os.getenv("PORT", "8080"))  # 容器内部固定端口（entrypoint 硬编码 8080，对外由 HOST_PORT 映射）
    # 工业标准 dynamic batching 参数（Triton/TF-Serving 模式）：
    #   - max_batch_size：VALID_BATCH_SIZES 最大值（满 batch 立即触发）
    #   - max_queue_delay_ms：BATCH_TIMEOUT（超时按最早入队 chunk 计时，严格延迟上限）
    BATCH_TIMEOUT: int = int(os.getenv("BATCH_TIMEOUT", "10"))  # 毫秒（max_queue_delay）
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2000"))
    VERBOSE: bool = os.getenv("VERBOSE", "0") in ("1", "true", "True", "yes")

    # 音频最大时长限制（毫秒），超出返回 AUDIO_TOO_LONG(1005)。0 = 不限制。
    # 默认 2 小时（7200000ms）
    MAX_AUDIO_DURATION_MS: int = int(os.getenv("MAX_AUDIO_DURATION_MS", "7200000"))
    # 过载拒绝：并发信号量等待超时（秒），超时返回 SERVICE_BUSY(1007)。0 = 无限等待（不拒绝）。
    ACQUIRE_TIMEOUT: float = float(os.getenv("ACQUIRE_TIMEOUT", "5"))

    # 模型精度（见文件头精度矩阵）。
    # 默认 auto：代码层兜底，按硬件自动探测（GPU→trt_int8_enc/trt_fp16，CPU→onnx_int8）。
    # 部署脚本会显式覆盖：run.sh 与 docker-compose.yml 默认均为 trt_fp16（GPU 生产稳定基线）。
    MODEL_PRECISION: str = os.getenv("MODEL_PRECISION", "auto")

    # CPU 推理线程数。
    #   ORT_INTRA_OP_THREADS：单个推理 session 内算子并行线程数（0=自动取 cpu_count）
    #   ORT_INTER_OP_THREADS：session 间算子并行线程数（默认 1）
    #
    # ★应用范围（重要）：仅作用于「主 ASR 模型走 CPU（onnx_fp32 / onnx_int8）」时，
    #   见 asr_engine.py 的 device=="cpu" 分支。以下场景这两个参数完全不生效：
    #     - TRT 后端（trt_*）：主 ASR 在 GPU 上推理，不经过 ORT CPU session；
    #     - Silero VAD：session 线程数在 vad.py 中硬编码 intra=inter=1
    #       （VAD 是串行 LSTM，单次 run 仅处理 (1,576) 极小张量，多线程无收益且徒增
    #        调度开销；并行度由 VAD_SESSION_POOL_SIZE 多 session round-robin 提供）。
    #   因此在 GPU/TRT 部署下调这两个参数无效，网格压测请勿将其列为变量。
    # 高并发（CPU 部署）务必显式设小（如 总核数 / 预期并发），避免线程超额订阅导致越并发越慢。
    ORT_INTRA_OP_THREADS: int = int(os.getenv("ORT_INTRA_OP_THREADS", "0"))
    ORT_INTER_OP_THREADS: int = int(os.getenv("ORT_INTER_OP_THREADS", "1"))

    # CPU 流水线线程池大小（Stage1 VAD + Stage2 特征提取）。0=自动取 cpu_count。
    # 默认 32（256 核 + WORKERS=11 实测最优，per-worker）；小核数机器需调小。
    # 多 worker（WORKERS>1）时务必显式设小，避免每 worker 各开满核导致线程超额订阅。
    CPU_THREAD_POOL_SIZE: int = int(os.getenv("CPU_THREAD_POOL_SIZE", "32"))

    # ============================================================
    # 热词模块开关（按需裁剪推理路径，纯通用识别可全关省开销）
    # ============================================================
    # 路径 A（SeACo 在线热词）开关。默认 true。
    #   true：客户端传入 hotwords 时走 SeACo 实时编码 bias_embed 增强；
    #   false：忽略客户端 hotwords，不做 SeACo 增强（bias_encoder 仍可加载但不调用）。
    ENABLE_HOTWORD: bool = os.getenv("ENABLE_HOTWORD", "true").lower() in ("1", "true", "yes")
    # 路径 B（默认词表 Faiss 后处理纠错）开关。默认 true。
    #   true：客户端不传热词时，用默认词表 Faiss 三重判定保守纠错；
    #   false：不构建/不运行 Faiss 索引，通用识别零后处理开销。
    ENABLE_FAISS_CORRECTION: bool = os.getenv("ENABLE_FAISS_CORRECTION", "true").lower() in (
        "1", "true", "yes"
    )

    # 字级时间戳开关。默认 false（关闭，最大吞吐）。
    #   true：加载独立 timestamp engine（第 5 段），响应 asr[].words 返回字级时间戳。
    #         upsample_cnn + blstm 计算量较大，实测吞吐下降约 30%（2800→2000 req/s）。
    #   false：不加载/不运行 timestamp engine，words 为空数组，吞吐不受影响。
    ENABLE_WORD_TIMESTAMP: bool = os.getenv("ENABLE_WORD_TIMESTAMP", "false").lower() in (
        "1", "true", "yes"
    )
    # 字级时间戳 upsample 倍数（对齐模型 CifPredictorV3.upsample_times，本模型为 3）。
    # TIME_RATE = 10 * LFR_N(6) / 1000 / upsample_times，决定字级时间戳粒度（3 → 20ms）。
    # 必须与导出 timestamp engine 时的 upsample_times 一致，否则时间戳整体缩放错误。
    TIMESTAMP_UPSAMPLE_TIMES: int = int(os.getenv("TIMESTAMP_UPSAMPLE_TIMES", "3"))

    # GPU 多 stream 多 execution_context 池大小（单卡榨干利用率）。
    # 每个 TRT engine 共享 weights，创建 N 个 execution_context + N 个 CUDA stream，
    # 推理时用 queue.Queue 连接池借还 (context, stream)，不同 stream 上的 batch
    # 可在 GPU SM 上真正并行执行（GPU 分时调度）。
    # 压测发现 GPU sm-util 只有 13-15%，开 4 stream 预期提升到 40-60%，QPS 翻倍。
    #
    # ★应用范围（各段池化策略，见 trt_engine.py TRTEngine.load）：
    #   - encoder / cif / decoder：主链路每 chunk 必经，全部按本值池化（pool_size=N）；
    #   - timestamp（第 5 段，ENABLE_WORD_TIMESTAMP 启用时）：同为主链路串联一环
    #     （CIF 后 Decoder 前每 chunk 都调），也按本值池化，否则单 context 会成为
    #     串行瓶颈拖垮整条流水线；
    #   - bias_encoder：固定 pool_size=1，不随本值放大。原因：热词编码仅在「客户端传
    #     热词」或「默认词表热更新」时调用（低频，非每 chunk），且结果可缓存复用；
    #     多 context 对它无并发收益，反而白占显存，故单 context 足够。
    #
    # 显存开销：每 stream × 每段 activation ≈ 200-300MB。
    #   关闭时间戳：encoder+cif+decoder 3 段 × N；启用时间戳：再加 timestamp 1 段 × N。
    #   （bias_encoder 恒 1 份，不计入 N 倍放大）
    # 建议值：3-6，超过后收益递减且显存吃紧。
    GPU_STREAM_POOL_SIZE: int = int(os.getenv("GPU_STREAM_POOL_SIZE", "4"))

    # VAD ORT session 池大小（round-robin 分配，多请求真正并行）。
    # 单一全局 session 在并发场景下会被 ORT 内部串行化，需要多 session 才能真正并行。
    # 20 并发 × 30s 音频压测扫描结果（OMP_NUM_THREADS=1）：
    #   pool=1  QPS=12.76   pool=2  QPS=12.88（最优）  pool=4  QPS=12.67
    #   pool=8  QPS=12.48   pool=16 QPS=12.32          pool=32 QPS=12.08
    # 反直觉发现：pool 越大 QPS 反而略降，因为 OMP=1 后单 session 已高效，
    # 多 session 增加内存分配/缓存 miss/上下文切换开销。
    # 默认 2：性能网格实测 WORKERS=11 下 VAD_POOL=2 最优（详见性能网格测试报告）。
    VAD_SESSION_POOL_SIZE: int = int(os.getenv("VAD_SESSION_POOL_SIZE", "2"))

    # 固定参数（模型已打包进镜像）
    MODEL_DIR: str = "./models"
    # PT 后端权重目录（MODEL_PRECISION=pt 时用；GPU 优先/CPU 兜底）
    PT_MODEL_DIR: str = os.getenv("PT_MODEL_DIR", os.path.join("models", "asr", "pt"))

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
    # TRT profile seq_len 上限（max point）= 最大桶（134 帧 = 8s）。
    # audio_segment 切段上限 SPLIT_MAX_MS=133帧 < 134，永不越界，故 max 贴合最大桶即可，
    # 不留额外余量（余量会无谓增大 activation 显存峰值）。
    TRT_MAX_SEQ: int = int(os.getenv("TRT_MAX_SEQ", "134"))
    # 特征维度（LFR 后）
    FEAT_DIM: int = 560
    # encoder 输出维度
    HIDDEN_DIM: int = 512
    # 热词最大长度（token 数，用于 bias profile）
    # 16：覆盖英文热词 BPE 切分后的较长 subword 序列（seg_dict 集成后英文词可能切 8-16 subword）；
    #     中文热词逐字 1 字=1 token，16 足够长（如长机构名）。
    MAX_HOTWORD_LEN: int = int(os.getenv("MAX_HOTWORD_LEN", "16"))

    # ============================================================
    # 热词管理参数（路径 A：SeACo 在线热词，单一数据源）
    #   - main.py 编码热词时按 MAX_HOTWORD_NUM 截断 + 告警
    #   - TRT bias/decoder profile 的热词数维度 max=MAX_HOTWORD_NUM, opt=OPT_HOTWORD_NUM
    #   - model.py ASF 过滤保留 top-NFILTER 注入 decoder
    # 显存上界由 MAX_HOTWORD_NUM 固定，engine 无需随词表规模重建。
    # ============================================================
    # 客户端热词（路径 A SeACo）硬上限 / 截断点：客户端传入 hotwords 超过此值时
    # 截断保留 Top-N + 告警。engine bias profile 的热词维度 max = 此值 + 1（含哨兵）。
    # 注：默认词表恒走路径 B（Faiss，见 _determine_route），此值不再是默认词表的
    # 路由切换点，仅约束客户端在线热词的数量上限。
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
        mn, opt, mx = cls.min_seq(), cls.TRT_OPT_SEQ, cls.TRT_MAX_SEQ
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
                "bias_embed": {"min": (1, 1, hd), "opt": (ob, opt_hw, hd), "max": (mb, max_hw, hd)},
            }
        if profile_type == "bias":
            # 热词矩阵行数 = 实际热词数 + 1（[sos] 哨兵），故 max 用 MAX_HOTWORD_NUM + 1
            max_hw = cls.MAX_HOTWORD_NUM + 1
            opt_hw = cls.OPT_HOTWORD_NUM
            return {
                "hotword": {"min": (1, 1), "opt": (opt_hw, 4), "max": (max_hw, cls.MAX_HOTWORD_LEN)},
            }
        if profile_type == "timestamp":
            # 第 5 段字级时间戳：输入 encoder_out + mask + token_num
            # 输出 us_alphas/us_cif_peak（enc_len × upsample_times）由 engine 推断，无需 profile
            return {
                "encoder_out": {"min": (1, mn, hd), "opt": (ob, opt, hd), "max": (mb, mx, hd)},
                "mask": {"min": (1, 1, mn), "opt": (ob, 1, opt), "max": (mb, 1, mx)},
                "token_num": {"min": (1,), "opt": (ob,), "max": (mb,)},
            }
        raise ValueError(f"未知 profile_type: {profile_type}")

    # ============================================================
    # 设备 / 精度选择
    # ============================================================
    @classmethod
    def _detect_hardware_device(cls) -> str:
        """仅探测硬件：有可用 GPU 返回 cuda，否则 cpu（不考虑精度）。"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @classmethod
    def get_device(cls) -> str:
        """推理设备。

        在硬件探测基础上叠加精度约束：
        onnx_int8 是 CPU 专用（动态量化算子在 CUDA EP 上不支持，会 fallback
        CPU 并反复 Memcpy，既慢又无意义），故强制返回 cpu。
        """
        device = cls._detect_hardware_device()
        if device == "cuda" and cls.get_model_precision() == "onnx_int8":
            return "cpu"
        return device

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
        device = cls._detect_hardware_device()
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

    # --- ORT 分段 ONNX 路径（启用字级时间戳时用，替代整体模型） ---
    @classmethod
    def get_split_onnx_dir(cls) -> str:
        """分段 ONNX 目录（encoder/cif/decoder/bias_encoder/timestamp.onnx）。"""
        return os.path.join(cls.MODEL_DIR, "asr", "split")

    @classmethod
    def get_split_onnx_paths(cls) -> dict[str, str | None]:
        """获取 ORT 分段 ONNX 路径字典。缺失的段为 None。

        用于 ENABLE_WORD_TIMESTAMP 开启且走 ORT 后端时的分段串联推理：
        encoder → cif → decoder + bias_encoder + timestamp 五段独立 session。
        timestamp 仅 ENABLE_WORD_TIMESTAMP 开启时返回路径。
        """
        d = cls.get_split_onnx_dir()

        def _p(name: str) -> str | None:
            path = os.path.join(d, f"{name}.onnx")
            return path if os.path.exists(path) else None

        paths = {
            "encoder": _p("encoder"),
            "cif": _p("cif"),
            "decoder": _p("decoder"),
            "bias_encoder": _p("bias_encoder"),
        }
        paths["timestamp"] = _p("timestamp") if cls.ENABLE_WORD_TIMESTAMP else None
        return paths

    @classmethod
    def use_ort_split(cls) -> bool:
        """是否走 ORT 分段串联路径。

        条件：ORT 后端 + ENABLE_WORD_TIMESTAMP 开启 + 分段 ONNX（含 timestamp）齐全。
        否则回退整体模型（无字级时间戳）。
        """
        if cls.get_inference_backend() != "ort":
            return False
        if not cls.ENABLE_WORD_TIMESTAMP:
            return False
        paths = cls.get_split_onnx_paths()
        return all(paths.get(m) for m in ("encoder", "cif", "decoder", "timestamp"))

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
            "timestamp": os.getenv("TIMESTAMP_PRECISION"),
        }
        for module, val in env_map.items():
            if val:
                base[module] = val.lower()

        # timestamp 段兜底：BLSTM 不支持 int8，非法精度回退 fp16
        if base.get("timestamp") not in TIMESTAMP_ALLOWED_PRECISIONS:
            base["timestamp"] = "fp16"
        return base

    @classmethod
    def get_trt_engine_paths(cls) -> dict[str, str | None]:
        """获取 TRT 4 段 engine 路径字典（按 per-module 精度）。

        返回缺失的段为 None。
        """
        precision = cls.get_model_precision()
        if not precision.startswith("trt_"):
            return {"encoder": None, "cif": None, "decoder": None,
                    "bias_encoder": None, "timestamp": None}

        prec_map = cls.get_trt_precision_map()
        paths = {
            module: cls._find_trt_engine(prec_map[module], module)
            for module in ("encoder", "cif", "decoder", "bias_encoder")
        }
        # 第 5 段 timestamp engine：仅 ENABLE_WORD_TIMESTAMP 开启时查找。
        # timestamp 含 blstm，只支持 fp32/fp16（见 TIMESTAMP_ALLOWED_PRECISIONS），
        # 精度由 precision_map 决定（trt_fp32→fp32，其余→fp16，或 TIMESTAMP_PRECISION 覆盖）。
        if cls.ENABLE_WORD_TIMESTAMP:
            paths["timestamp"] = cls._find_trt_engine(prec_map["timestamp"], "timestamp")
        else:
            paths["timestamp"] = None
        return paths

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

    @classmethod
    def dump_effective_config(cls) -> dict:
        """收集实际生效的运行配置，返回结构化 dict（供启动时作为 JSON 字段输出）。

        ★按实际启用的模块动态组装：仅包含当前生效的分组。例如未开启热词模块
        （ENABLE_HOTWORD=false）则不含热词参数；未开 Faiss 则不含 Faiss 参数；
        非 CPU 后端不含 ORT 线程数；非 PT 后端不含 PT 权重目录。
        便于按实际运行状态复现与核对配置正确性。
        """
        precision = cls.get_model_precision()
        backend = cls.get_inference_backend()
        device = cls.get_device()
        import os as _os
        cpu_pool = cls.CPU_THREAD_POOL_SIZE or (_os.cpu_count() or 4)

        cfg: dict = {}

        # ── 模型精度（始终）──
        model = {
            "model_precision_requested": cls.MODEL_PRECISION,
            "precision": precision,
            "backend": backend,
            "device": device,
        }
        if backend == "trt":
            pm = cls.get_trt_precision_map()
            model["trt_seg_precision"] = {
                "encoder": pm["encoder"], "cif": pm["cif"], "decoder": pm["decoder"],
                "bias_encoder": pm["bias_encoder"], "timestamp": pm["timestamp"],
            }
        cfg["模型精度"] = model

        # ── 服务运行参数（始终）──
        svc = {
            "WORKERS": cls.WORKERS,
            "BATCH": cls.BATCH,
            "BATCH_TIMEOUT_ms": cls.BATCH_TIMEOUT,
            "MAX_CONCURRENT_REQUESTS": cls.MAX_CONCURRENT_REQUESTS,
            "ACQUIRE_TIMEOUT_s": cls.ACQUIRE_TIMEOUT,
            "MAX_AUDIO_DURATION_MS": cls.MAX_AUDIO_DURATION_MS,
            "CPU_THREAD_POOL_SIZE": cls.CPU_THREAD_POOL_SIZE,
            "CPU_THREAD_POOL_effective": cpu_pool,
            "CPU_threads_total_est": cls.WORKERS * cpu_pool,
            "VAD_SESSION_POOL_SIZE": cls.VAD_SESSION_POOL_SIZE,
        }
        if backend == "trt":
            svc["GPU_STREAM_POOL_SIZE"] = cls.GPU_STREAM_POOL_SIZE
        cfg["服务运行参数"] = svc

        # ── CPU 推理线程数（仅 CPU 后端 onnx_* 生效）──
        if backend == "ort" and device == "cpu":
            cfg["CPU推理线程数"] = {
                "ORT_INTRA_OP_THREADS": cls.ORT_INTRA_OP_THREADS,
                "ORT_INTER_OP_THREADS": cls.ORT_INTER_OP_THREADS,
            }

        # ── OMP / BLAS 线程数（始终，稳定性红线）──
        cfg["OMP_BLAS线程数"] = {
            "OMP_NUM_THREADS": _os.getenv("OMP_NUM_THREADS", "?"),
            "MKL_NUM_THREADS": _os.getenv("MKL_NUM_THREADS", "?"),
            "OPENBLAS_NUM_THREADS": _os.getenv("OPENBLAS_NUM_THREADS", "?"),
        }

        # ── 字级时间戳（仅启用时）──
        if cls.ENABLE_WORD_TIMESTAMP:
            ts = {"TIMESTAMP_UPSAMPLE_TIMES": cls.TIMESTAMP_UPSAMPLE_TIMES}
            if backend == "trt":
                ts["timestamp_precision"] = cls.get_trt_precision_map()["timestamp"]
            cfg["字级时间戳"] = ts

        # ── 热词模块（路径 A，仅启用时）──
        if cls.ENABLE_HOTWORD:
            cfg["热词模块_路径A_SeACo"] = {
                "MAX_HOTWORD_NUM": cls.MAX_HOTWORD_NUM,
                "OPT_HOTWORD_NUM": cls.OPT_HOTWORD_NUM,
                "NFILTER": cls.NFILTER,
                "MAX_HOTWORD_LEN": cls.MAX_HOTWORD_LEN,
            }

        # ── Faiss 大词库纠错（路径 B，仅启用时）──
        if cls.ENABLE_FAISS_CORRECTION:
            cfg["Faiss纠错_路径B"] = {
                "FAISS_WINDOW_SIZES": cls.FAISS_WINDOW_SIZES,
                "FAISS_TOPK": cls.FAISS_TOPK,
                "FAISS_PINYIN_WEIGHT": cls.FAISS_PINYIN_WEIGHT,
                "FAISS_EDIT_WEIGHT": cls.FAISS_EDIT_WEIGHT,
                "FAISS_SCORE_THRESHOLD": cls.FAISS_SCORE_THRESHOLD,
                "GAP_THRESHOLD": cls.GAP_THRESHOLD,
                "FINAL_SCORE_THRESHOLD": cls.FINAL_SCORE_THRESHOLD,
            }

        # ── 词表 + 热更新（热词或 Faiss 任一启用时）──
        if cls.ENABLE_HOTWORD or cls.ENABLE_FAISS_CORRECTION:
            hw = {
                "DEFAULT_HOTWORD_PATH": cls.DEFAULT_HOTWORD_PATH,
                "HOTWORD_RELOAD_ENABLED": cls.HOTWORD_RELOAD_ENABLED,
            }
            if cls.HOTWORD_RELOAD_ENABLED:
                hw["HOTWORD_POLL_INTERVAL_s"] = cls.HOTWORD_POLL_INTERVAL
            cfg["词表_热更新"] = hw

        # ── Bucket / Batch（始终）──
        cfg["Bucket_Batch"] = {
            "BUCKET_SEQ_LENS": cls.BUCKET_SEQ_LENS,
            "VALID_BATCH_SIZES": cls.VALID_BATCH_SIZES,
            "TRT_OPT_SEQ": cls.TRT_OPT_SEQ,
            "TRT_MAX_SEQ": cls.TRT_MAX_SEQ,
        }

        # ── 本地 PT 权重目录（仅 PT 后端）──
        if backend == "pt":
            cfg["本地PT权重目录"] = {"PT_MODEL_DIR": cls.PT_MODEL_DIR}

        # ── 模型路径（始终，确认真的加载了预期产物）──
        if backend == "trt":
            cfg["模型路径"] = {
                m: (p or "（缺失/未启用）") for m, p in cls.get_trt_engine_paths().items()
            }
        elif backend == "pt":
            cfg["模型路径"] = {"pt_weights": cls.PT_MODEL_DIR}
        else:  # ort
            if cls.use_ort_split():
                paths = {"ort_mode": "分段串联（字级时间戳）"}
                paths.update({
                    m: (p or "（缺失/未启用）") for m, p in cls.get_split_onnx_paths().items()
                })
                cfg["模型路径"] = paths
            else:
                cfg["模型路径"] = {
                    "ort_mode": "整体模型",
                    "model.onnx": cls.get_asr_model_path(),
                    "model_eb.onnx": cls.get_asr_bias_model_path(),
                }
        return cfg


settings = Settings()
