"""
ASR ONNX 推理引擎

SeACo-Paraformer 双模型架构：
- model.onnx: 主模型（encoder + predictor + decoder），输入 speech features，输出 logits
- model_eb.onnx: bias encoder（热词编码器），输入 hotword token IDs，输出 bias embeddings

推理流程：
1. [可选] model_eb.onnx: hotwords → token IDs → bias embeddings
2. model.onnx: speech features [+ bias embeddings] → logits

优化：
- graph_optimization_level=ORT_ENABLE_ALL
- GPU 推理使用 IO Binding
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
    ASR ONNX Runtime 推理引擎。

    双模型：
    - _session: 主模型 (model.onnx)
    - _bias_session: 热词 bias encoder (model_eb.onnx)
    """

    def __init__(self):
        self._session: ort.InferenceSession | None = None
        self._bias_session: ort.InferenceSession | None = None
        self._device: str = "cpu"
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._bias_input_names: list[str] = []
        self._bias_output_names: list[str] = []

    def load(self):
        """加载 ASR 主模型和 bias encoder。"""
        self._device = settings.get_device()
        self._load_main_model()
        self._load_bias_model()
        self._warmup()

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def has_bias_model(self) -> bool:
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
        """模型预热。"""
        logger.info("ASR 模型预热中...")
        try:
            dummy_feats = np.random.randn(1, 100, 560).astype(np.float32)
            dummy_lengths = np.array([100], dtype=np.int32)

            feed_dict = {self._input_names[0]: dummy_feats}
            if len(self._input_names) >= 2:
                feed_dict[self._input_names[1]] = dummy_lengths

            self._session.run(self._output_names, feed_dict)
            logger.info("ASR 模型预热完成")
        except Exception as e:
            logger.warning(f"ASR 模型预热失败（不影响服务启动）: {e}")

    def encode_hotwords(self, hotword_token_ids: np.ndarray) -> np.ndarray | None:
        """
        使用 bias encoder 编码热词。

        参数:
            hotword_token_ids: 热词 token ID 矩阵, shape=(num_hotwords, max_len)

        返回:
            bias embeddings 或 None（无 bias 模型时）
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
            return outputs[0]  # bias embeddings
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
            # Padding 到同一长度
            max_len = max(f.shape[0] for f in features_list)
            feat_dim = features_list[0].shape[1]
            batch_size = len(features_list)

            padded = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
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

            # 推理
            if self._device == "cuda":
                outputs = self._infer_with_io_binding(feed_dict)
            else:
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
        """单条推理。"""
        results = self.infer_batch([features], [features.shape[0]], bias_embeddings)
        return results[0]


# 全局单例
asr_engine = ASREngine()
