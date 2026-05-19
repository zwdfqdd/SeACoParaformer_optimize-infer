"""
TensorRT 推理引擎

替代 ORT 的高性能 GPU 推理引擎，支持：
- Dynamic batch（对齐 scheduler bucket 策略）
- fp16/INT8 精度
- Engine 缓存（首次加载后序列化，后续直接反序列化）
- 回退机制：TRT 加载失败 → 回退 ORT fp32

接口对齐 asr_engine.py，scheduler.py 无需修改。
"""

import os
from pathlib import Path

import numpy as np

from src.config import settings
from src.logger import logger

try:
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


if TRT_AVAILABLE:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine:
    """
    TensorRT 推理引擎。

    加载序列化 engine 文件，执行 dynamic batch 推理。
    接口对齐 ASREngine.infer_batch_raw()。
    """

    def __init__(self):
        self._engine = None
        self._context = None
        self._stream = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._input_shapes: dict[str, tuple] = {}
        self._output_shapes: dict[str, tuple] = {}

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    def load(self, engine_path: str):
        """加载 TRT engine 文件。"""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT 未安装，无法加载 engine")

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine 文件不存在: {engine_path}")

        logger.info(f"TRT engine 加载中: {engine_path}")

        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"Engine 反序列化失败: {engine_path}")

        self._context = self._engine.create_execution_context()

        # 解析输入输出
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            shape = self._engine.get_tensor_shape(name)

            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
                self._input_shapes[name] = tuple(shape)
            else:
                self._output_names.append(name)
                self._output_shapes[name] = tuple(shape)

        logger.info(
            f"TRT engine 加载成功: 输入={self._input_names}, "
            f"输出={self._output_names}"
        )

    def infer_batch_raw(
        self,
        padded_feats: np.ndarray,
        lengths: np.ndarray,
        bias_embeddings: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        TRT 推理（接口对齐 ASREngine.infer_batch_raw）。

        使用 torch tensor 管理 GPU 内存（简洁且与现有依赖兼容）。

        参数：
            padded_feats: (batch, seq_len, 560) float32
            lengths: (batch,) int32
            bias_embeddings: (1, num_hw, 512) float32 或 None

        返回：
            logits 列表，每个元素 shape=(seq_len, vocab_size)
        """
        if self._engine is None:
            raise RuntimeError("TRT engine 未加载")

        import torch

        batch_size = padded_feats.shape[0]
        seq_len = padded_feats.shape[1]
        feat_dim = padded_feats.shape[2]

        # 准备输入 tensor（GPU）
        d_inputs = {}
        for name in self._input_names:
            if name == "speech":
                data = torch.from_numpy(padded_feats).cuda().contiguous()
                self._context.set_input_shape(name, (batch_size, seq_len, feat_dim))
            elif name == "speech_lengths":
                data = torch.from_numpy(lengths.astype(np.int32)).cuda().contiguous()
                self._context.set_input_shape(name, (batch_size,))
            elif "bias_embed" in name:
                if bias_embeddings is not None:
                    bias_data = np.tile(bias_embeddings, (batch_size, 1, 1)).astype(np.float32)
                    data = torch.from_numpy(bias_data).cuda().contiguous()
                    self._context.set_input_shape(name, bias_data.shape)
                else:
                    data = torch.zeros((batch_size, 1, 512), dtype=torch.float32, device="cuda")
                    self._context.set_input_shape(name, (batch_size, 1, 512))
            else:
                continue
            d_inputs[name] = data
            self._context.set_tensor_address(name, data.data_ptr())

        # 分配输出 tensor（GPU）
        d_outputs = {}
        for name in self._output_names:
            output_shape = tuple(self._context.get_tensor_shape(name))
            d_outputs[name] = torch.empty(output_shape, dtype=torch.float32, device="cuda")
            self._context.set_tensor_address(name, d_outputs[name].data_ptr())

        # 执行推理
        stream = torch.cuda.current_stream()
        self._context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        # 拷贝输出到 CPU
        logits = d_outputs[self._output_names[0]].cpu().numpy()

        # 拆分 batch 结果
        results = []
        for i in range(batch_size):
            results.append(logits[i])

        return results

    def warmup(self, bucket_seq_lens: list[int], batch_sizes: list[int]):
        """
        预热：对所有 bucket × batch 组合执行一次推理。
        """
        if not self.is_loaded:
            return

        logger.info("TRT engine 预热中...")
        feat_dim = 560
        count = 0

        for seq_len in bucket_seq_lens:
            for batch in batch_sizes:
                try:
                    dummy_feats = np.random.randn(batch, seq_len, feat_dim).astype(np.float32)
                    dummy_lengths = np.full(batch, seq_len, dtype=np.int32)
                    self.infer_batch_raw(dummy_feats, dummy_lengths, None)
                    count += 1
                except Exception as e:
                    logger.warning(f"  TRT 预热失败 batch={batch}, seq={seq_len}: {e}")

        logger.info(f"TRT engine 预热完成（{count} 个 shape）")


# 全局单例
trt_engine = TRTEngine()
