"""
ASR 推理引擎（支持 ORT 和 TensorRT 双后端）

SeACo-Paraformer 双模型架构：
- model.onnx / model.engine: 主模型（encoder + predictor + decoder）
- model_eb.onnx / model_eb.engine: bias encoder（热词编码器）

推理后端选择（由 MODEL_PRECISION 决定）：
- fp32 / int8 → ONNX Runtime
- trt_fp16 / trt_int8 → TensorRT（回退 ORT fp32）

优化：
- ORT: graph_optimization_level=ORT_ENABLE_ALL, enable_mem_pattern=False
- TRT: dynamic shape profile, engine 缓存, CUDA stream 异步
- 服务启动时模型预热（dummy inference）
"""

import os

import numpy as np
import onnxruntime as ort

from src.config import settings
from src.errors import ASRException, ErrorCode
from src.logger import logger


class ASREngine:
    """
    ASR 推理引擎（ORT + TRT 双后端）。

    双模型：
    - _session: ORT 主模型 (model.onnx)
    - _trt_engine: TRT 主模型 (model.engine)
    - _bias_session: 热词 bias encoder (model_eb.onnx)

    推理优先级：TRT > ORT（TRT 加载失败自动回退 ORT）
    """

    def __init__(self):
        self._session: ort.InferenceSession | None = None
        self._trt_engine = None  # TRTEngine 实例
        self._bias_session: ort.InferenceSession | None = None
        self._device: str = "cpu"
        self._backend: str = "ort"  # "ort" 或 "trt"
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._bias_input_names: list[str] = []
        self._bias_output_names: list[str] = []

    def load(self):
        """加载 ASR 主模型和 bias encoder。"""
        self._device = settings.get_device()
        precision = settings.get_model_precision()
        backend = settings.get_inference_backend()
        logger.info(f"模型精度策略: {precision} (设备: {self._device}, 后端: {backend})")

        if backend == "trt":
            # 尝试加载 TRT engine
            if self._load_trt_engine():
                self._backend = "trt"
            else:
                # TRT 加载失败，回退 ORT fp32
                logger.warning("TRT engine 加载失败，回退到 ORT fp32")
                self._backend = "ort"
                self._load_main_model()
        else:
            self._backend = "ort"
            self._load_main_model()

        self._load_bias_model()
        self._warmup()

    @property
    def is_loaded(self) -> bool:
        if self._backend == "trt":
            return self._trt_engine is not None and self._trt_engine.is_loaded
        return self._session is not None

    @property
    def has_bias_model(self) -> bool:
        return self._bias_session is not None

    def _load_trt_engine(self) -> bool:
        """尝试加载 TRT engine，成功返回 True。"""
        engine_path = settings.get_asr_trt_engine_path()
        if not engine_path:
            logger.info("未找到匹配的 TRT engine 文件")
            return False

        try:
            from src.trt_engine import TRTEngine
            self._trt_engine = TRTEngine()
            self._trt_engine.load(engine_path)
            return True
        except Exception as e:
            logger.warning(f"TRT engine 加载异常: {e}")
            self._trt_engine = None
            return False
        return self._bias_session is not None

    def _load_main_model(self):
        """加载 ASR 主模型 (model.onnx)。"""
        model_path = settings.get_asr_model_path()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 禁用内存模式：CIF predictor 输出动态 token 数量，
        # ORT 内存缓存会因 shape 变化导致第二次推理 Mul 广播失败
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False

        if self._device == "cpu":
            cpu_count = os.cpu_count() or 4
            sess_options.intra_op_num_threads = cpu_count
            sess_options.inter_op_num_threads = 2

        if self._device == "cuda":
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                }),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            self._session = ort.InferenceSession(model_path, sess_options, providers=providers)
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._output_names = [o.name for o in self._session.get_outputs()]
            logger.info(
                f"ASR 主模型加载成功: {model_path}, 设备: {self._device}, "
                f"输入: {self._input_names}, 输出: {self._output_names}"
            )
        except Exception as e:
            raise ASRException(ErrorCode.MODEL_LOAD_FAILED, f"ASR 主模型加载失败: {e}")

    def _load_bias_model(self):
        """加载 bias encoder (model_eb.onnx)，用于热词编码。"""
        bias_path = settings.get_asr_bias_model_path()
        if not os.path.exists(bias_path):
            logger.info(f"Bias encoder 不存在，热词功能不可用: {bias_path}")
            return

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # bias encoder 始终在 CPU 运行（轻量模型）
            self._bias_session = ort.InferenceSession(
                bias_path, sess_options, providers=["CPUExecutionProvider"]
            )
            self._bias_input_names = [i.name for i in self._bias_session.get_inputs()]
            self._bias_output_names = [o.name for o in self._bias_session.get_outputs()]
            logger.info(
                f"Bias encoder 加载成功: {bias_path}, "
                f"输入: {self._bias_input_names}, 输出: {self._bias_output_names}"
            )
        except Exception as e:
            logger.warning(f"Bias encoder 加载失败（基础识别不受影响）: {e}")

    def _warmup(self):
        """
        模型预热：对所有 bucket(音频段长度) × batch 组合执行一次推理。

        plan.md 3.11 要求：
        - 音频段长度桶：2s/4s/8s → LFR 帧数约 34/67/134
        - batch：1, 2, 4, 8, 12
        - 在不同音频段、不同 batch 下都预热一次

        预热后所有 shape 命中 kernel cache，推理延迟稳定。
        """
        bucket_seq_lens = [34, 67, 134]
        batch_sizes = [1, 2, 4, 8, 12]

        if self._backend == "trt" and self._trt_engine is not None:
            self._trt_engine.warmup(bucket_seq_lens, batch_sizes)
            return

        # ORT 预热
        if self._session is None:
            return

        logger.info("ASR 模型预热中（bucket × batch 全组合）...")
        feat_dim = 560
        warmup_count = 0

        for seq_len in bucket_seq_lens:
            for batch in batch_sizes:
                try:
                    dummy_feats = np.random.randn(batch, seq_len, feat_dim).astype(np.float32)
                    dummy_lengths = np.full(batch, seq_len, dtype=np.int32)

                    feed_dict = {}
                    for name in self._input_names:
                        if name == "speech":
                            feed_dict[name] = dummy_feats
                        elif name == "speech_lengths":
                            feed_dict[name] = dummy_lengths
                        elif "bias_embed" in name:
                            inp = next(i for i in self._session.get_inputs() if i.name == name)
                            embed_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else 512
                            feed_dict[name] = np.zeros((batch, 1, embed_dim), dtype=np.float32)

                    self._session.run(self._output_names, feed_dict)
                    warmup_count += 1
                except Exception as e:
                    logger.warning(f"  预热失败 batch={batch}, seq={seq_len}: {e}")

        logger.info(f"ASR 模型预热完成（{warmup_count} 个 shape 已缓存）")

    def encode_hotwords(self, hotword_token_ids: np.ndarray) -> np.ndarray | None:
        """
        使用 bias encoder 编码热词。

        参数:
            hotword_token_ids: 热词 token ID 矩阵, shape=(num_hotwords, max_len)

        返回:
            bias embeddings, shape=(1, num_hotwords, embed_dim)，或 None（无 bias 模型时）
        """
        if self._bias_session is None:
            return None

        try:
            feed = {self._bias_input_names[0]: hotword_token_ids.astype(np.int32)}
            if len(self._bias_input_names) >= 2:
                # 可能需要 hotword lengths
                lengths = np.array(
                    [(row != 0).sum() for row in hotword_token_ids], dtype=np.int32
                )
                feed[self._bias_input_names[1]] = lengths

            outputs = self._bias_session.run(self._bias_output_names, feed)
            hw_embed = outputs[0]  # bias embeddings

            # 确保输出为 3D: (1, num_hotwords, embed_dim)
            if hw_embed.ndim == 2:
                hw_embed = hw_embed[np.newaxis, :, :]  # (num_hw, dim) → (1, num_hw, dim)

            if settings.VERBOSE:
                logger.debug(f"[Hotwords] bias_embed shape={hw_embed.shape}")

            return hw_embed
        except Exception as e:
            logger.warning(f"Bias encoder 推理失败: {e}")
            return None

    def infer_batch(
        self,
        features_list: list[np.ndarray],
        lengths: list[int],
        bias_embeddings: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        Batch 推理。

        参数:
            features_list: 特征列表，每个元素 shape=(time, feat_dim)
            lengths: 每个特征的有效长度
            bias_embeddings: 热词 bias embeddings（可选）

        返回:
            logits 列表
        """
        if self._session is None:
            raise ASRException(ErrorCode.ASR_INFER_FAILED, "ASR 模型未加载")

        try:
            # Padding 到最近 bucket 边界（命中 shape cache，避免 kernel 重编译）
            max_len = max(f.shape[0] for f in features_list)
            bucket_len = self._get_bucket_seq_len(max_len)
            feat_dim = features_list[0].shape[1]
            batch_size = len(features_list)

            padded = np.zeros((batch_size, bucket_len, feat_dim), dtype=np.float32)
            for i, feat in enumerate(features_list):
                padded[i, :feat.shape[0], :] = feat

            lengths_arr = np.array(lengths, dtype=np.int32)

            # 构建输入
            feed_dict = {self._input_names[0]: padded}
            if len(self._input_names) >= 2:
                feed_dict[self._input_names[1]] = lengths_arr

            # bias_embed 输入（模型必需，无热词时传零向量）
            for name in self._input_names:
                if "bias_embed" in name and name not in feed_dict:
                    if bias_embeddings is not None:
                        bias_batch = np.tile(bias_embeddings, (batch_size, 1, 1)).astype(np.float32)
                        feed_dict[name] = bias_batch
                    else:
                        # 推断 embed_dim
                        inp = next(i for i in self._session.get_inputs() if i.name == name)
                        embed_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else 512
                        feed_dict[name] = np.zeros((batch_size, 1, embed_dim), dtype=np.float32)

            # 推理（禁用 IO Binding，动态 shape 下不稳定）
            outputs = self._session.run(self._output_names, feed_dict)

            # 拆分 batch 结果
            logits = outputs[0]  # shape: (batch, time, vocab)
            results = []
            for i in range(batch_size):
                results.append(logits[i, :lengths[i], :])

            return results

        except ASRException:
            raise
        except Exception as e:
            raise ASRException(ErrorCode.ASR_INFER_FAILED, f"ASR 推理失败: {e}")

    def _infer_with_io_binding(self, feed_dict: dict) -> list[np.ndarray]:
        """使用 IO Binding 进行 GPU 推理。"""
        try:
            io_binding = self._session.io_binding()
            for name, data in feed_dict.items():
                tensor = ort.OrtValue.ortvalue_from_numpy(data, "cuda", 0)
                io_binding.bind_ortvalue_input(name, tensor)
            for name in self._output_names:
                io_binding.bind_output(name, "cuda")
            self._session.run_with_iobinding(io_binding)
            return io_binding.copy_outputs_to_cpu()
        except Exception:
            logger.warning("IO Binding 失败，回退到普通推理")
            return self._session.run(self._output_names, feed_dict)

    def infer_single(
        self, features: np.ndarray, bias_embeddings: np.ndarray | None = None
    ) -> np.ndarray:
        """单条推理（pad 到最近 bucket seq_len）。"""
        results = self.infer_batch([features], [features.shape[0]], bias_embeddings)
        return results[0]

    def infer_batch_raw(
        self,
        padded_feats: np.ndarray,
        lengths: np.ndarray,
        bias_embeddings: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        直接推理已 pad 好的固定 shape 输入（跳过内部 padding 逻辑）。
        用于严格 bucket 推理，输入已 pad 到桶边界。

        自动路由到 TRT 或 ORT 后端。

        参数:
            padded_feats: shape=(batch, bucket_seq_len, feat_dim) 已 pad
            lengths: shape=(batch,) 桶长度
            bias_embeddings: 热词 bias（可选）

        返回:
            logits 列表
        """
        if self._backend == "trt" and self._trt_engine is not None:
            return self._trt_engine.infer_batch_raw(padded_feats, lengths, bias_embeddings)
        return self._infer_batch_raw_ort(padded_feats, lengths, bias_embeddings)

    def _infer_batch_raw_ort(
        self,
        padded_feats: np.ndarray,
        lengths: np.ndarray,
        bias_embeddings: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        直接推理已 pad 好的固定 shape 输入（跳过内部 padding 逻辑）。
        用于严格 bucket 推理，输入已 pad 到桶边界。

        参数:
            padded_feats: shape=(batch, bucket_seq_len, feat_dim) 已 pad
            lengths: shape=(batch,) 每条的有效长度
            bias_embeddings: 热词 bias（可选）

        返回:
            logits 列表（按有效长度截断）
        """
        if self._session is None:
            raise ASRException(ErrorCode.ASR_INFER_FAILED, "ASR 模型未加载")

        try:
            batch_size = padded_feats.shape[0]

            feed_dict = {}
            for name in self._input_names:
                if name == "speech":
                    feed_dict[name] = padded_feats
                elif name == "speech_lengths":
                    feed_dict[name] = lengths
                elif "bias_embed" in name:
                    if bias_embeddings is not None:
                        feed_dict[name] = np.tile(bias_embeddings, (batch_size, 1, 1)).astype(np.float32)
                    else:
                        inp = next(i for i in self._session.get_inputs() if i.name == name)
                        embed_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else 512
                        feed_dict[name] = np.zeros((batch_size, 1, embed_dim), dtype=np.float32)

            outputs = self._session.run(self._output_names, feed_dict)

            logits = outputs[0]
            results = []
            for i in range(batch_size):
                results.append(logits[i])
            return results

        except Exception as e:
            raise ASRException(ErrorCode.ASR_INFER_FAILED, f"ASR 推理失败: {e}")

    @staticmethod
    def _get_bucket_seq_len(seq_len: int) -> int:
        """将 seq_len pad 到最近的 bucket 边界。"""
        bucket_seq_lens = [34, 67, 134]
        for b in bucket_seq_lens:
            if seq_len <= b:
                return b
        return seq_len


# 全局单例
asr_engine = ASREngine()
