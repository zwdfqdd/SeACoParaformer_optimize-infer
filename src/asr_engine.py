"""
ASR 推理引擎（TRT / ORT / PT 多后端路由）

- TRT：分段架构 engine（encoder/cif/decoder/bias_encoder + 可选 timestamp），GPU 主路径
- ORT：整体模型（model.onnx，无时间戳，兜底）；启用字级时间戳时切分段串联（ort_split）
- PT ：原生 PyTorch 推理（GPU 优先/CPU 兜底），无需转换

四种 backend：trt / ort / ort_split / pt。三后端均支持字级时间戳 + 热词 + Faiss（参数可选）。

接口对外统一（scheduler 不感知后端差异）：
    infer_batch_raw(padded_feats, lengths, bias_embeddings) → list[(logits, ts_data)]
    encode_hotwords(hotword_token_ids) → bias_embed (1, num_hw, 512) | None
"""

import os

import numpy as np
import onnxruntime as ort

from src.config import settings
from src.errors import ASRException, ErrorCode
from src.logger import logger


class ASREngine:
    """ASR 推理引擎（TRT / ORT / ort_split / PT 多后端路由）。"""

    def __init__(self):
        # ORT 后端
        self._session: ort.InferenceSession | None = None
        self._bias_session: ort.InferenceSession | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._bias_input_names: list[str] = []
        self._bias_output_names: list[str] = []

        # TRT 后端
        self._trt_engine = None  # src.trt_engine.TRTEngine 实例
        # ORT 分段后端（启用字级时间戳时用，替代整体模型）
        self._ort_split = None  # src.ort_engine.ORTSplitEngine 实例
        # PT 原生后端（MODEL_PRECISION=pt）
        self._pt_engine = None  # src.pt_engine.PTEngine 实例

        self._device: str = "cpu"
        self._backend: str = "ort"  # "ort" / "trt" / "ort_split" / "pt"

        # ── 运行时健康状态（R12/R14：/health 除加载态外反映运行时卡死/降级）──
        # 由 scheduler 在每次推理成功/失败后打点（record_infer_success/failure）。
        import threading as _threading
        self._health_lock = _threading.Lock()
        self._consecutive_failures: int = 0   # 连续失败计数（含超时），成功即清零
        self._last_success_ts: float = 0.0     # 最近一次推理成功时刻（monotonic）
        self._total_infer_ok: int = 0
        self._total_infer_fail: int = 0
        # 静默降级标记（R14）：加载阶段发生后端回退（TRT→ORT / PT→ORT / ORT分段→整体）时置位，
        # 记录原因，供 /health 与 dump_config 暴露（避免实例在低性能/降功能模式下无人察觉）。
        self._degraded_reason: str | None = None

    # ============================================================
    # 加载入口
    # ============================================================
    def load(self):
        self._device = settings.get_device()
        precision = settings.get_model_precision()
        backend = settings.get_inference_backend()
        logger.info(f"模型精度策略: {precision} (设备: {self._device}, 后端: {backend})")

        if backend == "pt":
            # PT 原生推理（GPU 优先/CPU 兜底）；加载失败回退 ORT
            if self._load_pt_engine():
                self._backend = "pt"
                self._warmup()
                return
            logger.warning("PT engine 加载失败，回退 ORT")
            self._mark_degraded("PT engine 加载失败，已回退 ORT 后端")
            self._load_ort_backend()
            self._warmup()
            return

        if backend == "trt":
            if self._load_trt_engines():
                self._backend = "trt"
            else:
                logger.warning("TRT engines 加载失败，回退到 ORT")
                self._mark_degraded("TRT engines 加载失败，已回退 ORT 后端")
                self._load_ort_backend()
        else:
            self._load_ort_backend()

        self._warmup()

    def _load_ort_backend(self):
        """加载 ORT 后端：启用字级时间戳且分段 ONNX 齐全 → 分段串联；否则整体模型。"""
        if settings.use_ort_split():
            # 静默降级告警：onnx_int8 本意是 CPU 小模型（体积-75%），但启用字级时间戳后
            # 走 fp32 分段串联（split 目录只有 fp32 产物），失去 int8 的体积/速度特性。
            if settings.get_model_precision() == "onnx_int8":
                logger.warning(
                    "onnx_int8 + ENABLE_WORD_TIMESTAMP：已切换到 fp32 分段串联"
                    "（无 int8 量化分段产物），体积/速度不再是 int8 特性。"
                    "若需 int8 体积优势请关闭字级时间戳。"
                )
            if self._load_ort_split():
                self._backend = "ort_split"
                logger.info("ORT 后端使用分段串联模式（支持字级时间戳）")
                return
            logger.warning("ORT 分段加载失败，回退整体模型（字级时间戳不可用）")
            self._mark_degraded("ORT 分段加载失败，已回退整体模型（字级时间戳不可用）")
        self._backend = "ort"
        self._load_ort_main()
        self._load_ort_bias()

    def _load_ort_split(self) -> bool:
        paths = settings.get_split_onnx_paths()
        if not (paths.get("encoder") and paths.get("cif") and paths.get("decoder")):
            logger.info(f"ORT 分段 ONNX 不齐全: {paths}")
            return False
        try:
            from src.ort_engine import ORTSplitEngine
            engine = ORTSplitEngine()
            engine.load(
                encoder_path=paths["encoder"],
                cif_path=paths["cif"],
                decoder_path=paths["decoder"],
                bias_encoder_path=paths.get("bias_encoder"),
                timestamp_path=paths.get("timestamp"),
                device=self._device,
            )
            self._ort_split = engine
            return True
        except Exception as e:
            logger.warning(f"ORT 分段 engine 加载异常: {e}")
            self._ort_split = None
            return False

    @property
    def is_loaded(self) -> bool:
        if self._backend == "trt":
            return self._trt_engine is not None and self._trt_engine.is_loaded
        if self._backend == "ort_split":
            return self._ort_split is not None and self._ort_split.is_loaded
        if self._backend == "pt":
            return self._pt_engine is not None and self._pt_engine.is_loaded
        return self._session is not None

    @property
    def has_bias_model(self) -> bool:
        if self._backend == "trt":
            return self._trt_engine is not None and self._trt_engine.has_bias_encoder
        if self._backend == "ort_split":
            return self._ort_split is not None and self._ort_split.has_bias_encoder
        if self._backend == "pt":
            return self._pt_engine is not None and self._pt_engine.has_bias_encoder
        return self._bias_session is not None

    # ============================================================
    # 运行时健康状态（R12/R14）
    # ============================================================
    def _mark_degraded(self, reason: str):
        """标记静默降级（加载阶段后端回退），供 /health 与 dump_config 暴露。"""
        with self._health_lock:
            self._degraded_reason = reason

    def record_infer_success(self):
        """记录一次推理成功（scheduler 调用）：连续失败清零 + 刷新成功时刻。"""
        import time as _t
        with self._health_lock:
            self._consecutive_failures = 0
            self._last_success_ts = _t.monotonic()
            self._total_infer_ok += 1

    def record_infer_failure(self):
        """记录一次推理失败/超时（scheduler 调用）：连续失败 +1。"""
        with self._health_lock:
            self._consecutive_failures += 1
            self._total_infer_fail += 1

    def runtime_health(self) -> dict:
        """返回运行时健康快照（供 /health 判定 + 可观测）。

        healthy 判定：连续失败数 < HEALTH_MAX_CONSECUTIVE_FAILURES 且无静默降级。
        注：静默降级（后端回退）不算 unhealthy（服务仍可用），但通过 degraded_reason
        单独暴露，便于运维察觉实例运行在低性能/降功能模式。
        """
        with self._health_lock:
            cf = self._consecutive_failures
            reason = self._degraded_reason
            snapshot = {
                "backend": self._backend,
                "consecutive_failures": cf,
                "total_infer_ok": self._total_infer_ok,
                "total_infer_fail": self._total_infer_fail,
                "degraded_reason": reason,
            }
        runtime_ok = cf < settings.HEALTH_MAX_CONSECUTIVE_FAILURES
        snapshot["runtime_ok"] = runtime_ok
        return snapshot

    def active_probe(self) -> bool:
        """主动探针：跑一次极小 dummy 推理验证 GPU 链路存活（HEALTH_ACTIVE_PROBE=true 时）。

        成功返回 True 并记 success；异常返回 False 并记 failure。同步执行（调用方在
        线程池/超时保护下调用），不引入对 scheduler 的循环依赖。
        """
        try:
            seq = min(settings.BUCKET_SEQ_LENS)
            feats = np.zeros((1, seq, settings.FEAT_DIM), dtype=np.float32)
            lengths = np.array([seq], dtype=np.int32)
            self.infer_batch_raw(feats, lengths, None)
            self.record_infer_success()
            return True
        except Exception as e:
            logger.warning(f"健康主动探针失败: {e}")
            self.record_infer_failure()
            return False

    # ============================================================
    # TRT 加载
    # ============================================================
    def _load_trt_engines(self) -> bool:
        paths = settings.get_trt_engine_paths()
        if not paths.get("encoder") or not paths.get("cif") or not paths.get("decoder"):
            logger.info(f"TRT 4 段 engine 不齐全: {paths}")
            return False

        try:
            from src.trt_engine import TRTEngine
            engine = TRTEngine()
            engine.load(
                encoder_path=paths["encoder"],
                cif_path=paths["cif"],
                decoder_path=paths["decoder"],
                bias_encoder_path=paths.get("bias_encoder"),
                timestamp_path=paths.get("timestamp"),
            )
            self._trt_engine = engine
            return True
        except Exception as e:
            logger.warning(f"TRT engine 加载异常: {e}")
            self._trt_engine = None
            return False

    # ============================================================
    # PT 加载（原生 PyTorch 推理，GPU 优先/CPU 兜底）
    # ============================================================
    def _load_pt_engine(self) -> bool:
        try:
            from src.pt_engine import PTEngine
            engine = PTEngine()
            engine.load(device=None)  # None = GPU 优先/CPU 兜底
            self._pt_engine = engine
            self._device = engine._device
            return True
        except Exception as e:
            logger.warning(f"PT engine 加载异常: {e}")
            self._pt_engine = None
            return False

    # ============================================================
    # ORT 加载（v1 整体模型路径）
    # ============================================================
    def _load_ort_main(self):
        model_path = settings.get_asr_model_path()
        if not os.path.exists(model_path):
            raise ASRException(ErrorCode.MODEL_LOAD_FAILED, f"ORT 主模型不存在: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CIF predictor 输出动态 token 数量，关闭内存复用避免 broadcast 失败
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False

        if self._device == "cpu":
            intra = settings.ORT_INTRA_OP_THREADS or (os.cpu_count() or 4)
            sess_options.intra_op_num_threads = intra
            sess_options.inter_op_num_threads = settings.ORT_INTER_OP_THREADS
            logger.info(
                f"ORT CPU 线程配置: intra_op={intra}, inter_op={settings.ORT_INTER_OP_THREADS}"
                f"（高并发请按 总核数/并发数 调小 ORT_INTRA_OP_THREADS 避免线程超额订阅）"
            )
            providers = ["CPUExecutionProvider"]
        else:
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    # 按需分配，不按 2 的幂翻倍预占 arena：多 worker 场景显存占用显著降低
                    # （kNextPowerOfTwo 会激进预占，fp32 整体模型多开时易 OOM）。
                    "arena_extend_strategy": "kSameAsRequested",
                }),
                "CPUExecutionProvider",
            ]

        try:
            self._session = ort.InferenceSession(model_path, sess_options, providers=providers)
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._output_names = [o.name for o in self._session.get_outputs()]
            logger.info(
                f"ORT 主模型加载成功: {model_path}, 设备: {self._device}, "
                f"输入: {self._input_names}, 输出: {self._output_names}"
            )
        except Exception as e:
            raise ASRException(ErrorCode.MODEL_LOAD_FAILED, f"ORT 主模型加载失败: {e}")

    def _load_ort_bias(self):
        bias_path = settings.get_asr_bias_model_path()
        if not os.path.exists(bias_path):
            logger.info(f"ORT bias encoder 不存在，热词功能不可用: {bias_path}")
            return
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._bias_session = ort.InferenceSession(
                bias_path, sess_options, providers=["CPUExecutionProvider"]
            )
            self._bias_input_names = [i.name for i in self._bias_session.get_inputs()]
            self._bias_output_names = [o.name for o in self._bias_session.get_outputs()]
            logger.info(
                f"ORT bias encoder 加载成功: {bias_path}, "
                f"输入: {self._bias_input_names}, 输出: {self._bias_output_names}"
            )
        except Exception as e:
            logger.warning(f"ORT bias encoder 加载失败（基础识别不受影响）: {e}")

    # ============================================================
    # 预热
    # ============================================================
    def _warmup(self):
        bucket_seq_lens = settings.BUCKET_SEQ_LENS
        batch_sizes = settings.VALID_BATCH_SIZES

        if self._backend == "trt" and self._trt_engine is not None:
            self._trt_engine.warmup(bucket_seq_lens, batch_sizes)
            return

        if self._backend == "ort_split" and self._ort_split is not None:
            self._ort_split.warmup(bucket_seq_lens, batch_sizes)
            return

        if self._backend == "pt" and self._pt_engine is not None:
            self._pt_engine.warmup(bucket_seq_lens, batch_sizes)
            return

        if self._session is None:
            return

        logger.info("ORT 模型预热中（bucket × batch 全组合）...")
        feat_dim = 560
        warmup_count = 0

        for seq_len in bucket_seq_lens:
            for batch in batch_sizes:
                try:
                    dummy_feats = np.random.randn(batch, seq_len, feat_dim).astype(np.float32)
                    dummy_lengths = np.full(batch, seq_len, dtype=np.int64)

                    feed = {}
                    for name in self._input_names:
                        if name == "speech":
                            feed[name] = dummy_feats
                        elif name == "speech_lengths":
                            feed[name] = dummy_lengths
                        elif "bias_embed" in name:
                            inp = next(i for i in self._session.get_inputs() if i.name == name)
                            embed_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else 512
                            feed[name] = np.zeros((batch, 1, embed_dim), dtype=np.float32)

                    self._session.run(self._output_names, feed)
                    warmup_count += 1
                except Exception as e:
                    logger.warning(f"  ORT 预热失败 batch={batch}, seq={seq_len}: {e}")

        logger.info(f"ORT 模型预热完成（{warmup_count} 个 shape）")

    # ============================================================
    # 热词编码
    # ============================================================
    def encode_hotwords(self, hotword_token_ids: np.ndarray) -> np.ndarray | None:
        """
        编码热词为 bias_embed (1, num_hotwords, 512)。

        参数：
            hotword_token_ids: (H, L) int 矩阵，pad=0。
                **注意：调用方需保证最后一行是 [sos]=[1] 哨兵**（SeACo NO_BIAS 占位机制）。

        返回：
            bias_embed: (1, num_hotwords, 512) float32，或 None（无 bias 模型）
        """
        if self._backend == "trt":
            if self._trt_engine is None or not self._trt_engine.has_bias_encoder:
                return None
            try:
                return self._trt_engine.encode_hotwords(hotword_token_ids)
            except Exception as e:
                logger.warning(f"TRT bias encoder 推理失败: {e}")
                return None

        if self._backend == "ort_split":
            if self._ort_split is None or not self._ort_split.has_bias_encoder:
                return None
            try:
                return self._ort_split.encode_hotwords(hotword_token_ids)
            except Exception as e:
                logger.warning(f"ORT 分段 bias encoder 推理失败: {e}")
                return None

        if self._backend == "pt":
            if self._pt_engine is None or not self._pt_engine.has_bias_encoder:
                return None
            try:
                return self._pt_engine.encode_hotwords(hotword_token_ids)
            except Exception as e:
                logger.warning(f"PT bias encoder 推理失败: {e}")
                return None

        # ORT 后端
        if self._bias_session is None:
            return None
        try:
            feed = {self._bias_input_names[0]: hotword_token_ids.astype(np.int64)}
            if len(self._bias_input_names) >= 2:
                lengths = np.array(
                    [(row != 0).sum() for row in hotword_token_ids], dtype=np.int64
                )
                feed[self._bias_input_names[1]] = lengths
            outputs = self._bias_session.run(self._bias_output_names, feed)
            hw_embed = outputs[0]
            # model_eb.onnx 输出 (H, D)，补 batch 维 → (1, H, D)
            if hw_embed.ndim == 2:
                hw_embed = hw_embed[np.newaxis, :, :]
            return hw_embed
        except Exception as e:
            logger.warning(f"ORT bias encoder 推理失败: {e}")
            return None

    # ============================================================
    # 主推理（scheduler 调用）
    # ============================================================
    def infer_batch_raw(
        self,
        padded_feats: np.ndarray,
        lengths: np.ndarray,
        bias_embeddings: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """
        Batch 推理（已 pad 到桶边界）。

        参数：
            padded_feats: (batch, bucket_seq_len, feat_dim) float32
            lengths: (batch,) int32 桶长度
            bias_embeddings: (1, H, 512) float32 或 None

        返回：
            (logits, ts_data) 元组列表，每 batch 一项：
                logits: (token_num, vocab_size)，token_num 由 CIF 输出决定
                ts_data: 字级时间戳数据 dict 或 None；
                        ORT 整体模型未暴露 timestamp 输出，一律 None
        """
        if self._backend == "trt" and self._trt_engine is not None:
            return self._trt_engine.infer_batch_raw(padded_feats, lengths, bias_embeddings)
        if self._backend == "ort_split" and self._ort_split is not None:
            return self._ort_split.infer_batch_raw(padded_feats, lengths, bias_embeddings)
        if self._backend == "pt" and self._pt_engine is not None:
            return self._pt_engine.infer_batch_raw(padded_feats, lengths, bias_embeddings)
        return self._infer_batch_raw_ort(padded_feats, lengths, bias_embeddings)

    def _infer_batch_raw_ort(
        self,
        padded_feats: np.ndarray,
        lengths: np.ndarray,
        bias_embeddings: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        if self._session is None:
            raise ASRException(ErrorCode.ASR_INFER_FAILED, "ASR 模型未加载")

        try:
            batch_size = padded_feats.shape[0]

            feed = {}
            for name in self._input_names:
                if name == "speech":
                    feed[name] = padded_feats
                elif name == "speech_lengths":
                    feed[name] = lengths.astype(np.int64)
                elif "bias_embed" in name:
                    if bias_embeddings is not None:
                        feed[name] = np.tile(bias_embeddings, (batch_size, 1, 1)).astype(np.float32)
                    else:
                        inp = next(i for i in self._session.get_inputs() if i.name == name)
                        embed_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else 512
                        feed[name] = np.zeros((batch_size, 1, embed_dim), dtype=np.float32)

            outputs = self._session.run(self._output_names, feed)
            logits = outputs[0]

            # ORT 整体模型：logits shape = (batch, token_num, vocab) 或 (batch, seq_len, vocab)
            # 按 token_num（如有）或全部返回，scheduler 不再截断
            results = []
            # 优先取 token_num 输出（v1 整体模型有此输出）
            token_nums = None
            for j, oname in enumerate(self._output_names):
                if "token_num" in oname.lower():
                    token_nums = np.round(outputs[j].flatten()).astype(np.int64)
                    break
            # ORT 整体模型未暴露 CIF alphas，字级时间戳不可用（alphas=None）
            for i in range(batch_size):
                if token_nums is not None and i < len(token_nums):
                    n = int(token_nums[i])
                    results.append((logits[i, :n, :], None))
                else:
                    results.append((logits[i], None))
            return results
        except Exception as e:
            raise ASRException(ErrorCode.ASR_INFER_FAILED, f"ASR 推理失败: {e}")


# 全局单例
asr_engine = ASREngine()
