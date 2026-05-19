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
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 — 初始化 CUDA context

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
        self._stream = cuda.Stream()

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

        参数：
            padded_feats: (batch, seq_len, 560) float32
            lengths: (batch,) int32
            bias_embeddings: (1, num_hw, 512) float32 或 None

        返回：
            logits 列表，每个元素 shape=(seq_len, vocab_size)
        """
        if self._engine is None:
            raise RuntimeError("TRT engine 未加载")

        batch_size = padded_feats.shape[0]
        seq_len = padded_feats.shape[1]
        feat_dim = padded_feats.shape[2]

        # 设置 input shapes（dynamic batch）
        for name in self._input_names:
            if name == "speech":
                self._context.set_input_shape(name, (batch_size, seq_len, feat_dim))
            elif name == "speech_lengths":
                self._context.set_input_shape(name, (batch_size,))
            elif "bias_embed" in name:
                if bias_embeddings is not None:
                    bias_batch = np.tile(bias_embeddings, (batch_size, 1, 1)).astype(np.float32)
                    self._context.set_input_shape(name, bias_batch.shape)
                else:
                    self._context.set_input_shape(name, (batch_size, 1, 512))

        # 准备输入数据
        inputs = {}
        for name in self._input_names:
            if name == "speech":
                inputs[name] = np.ascontiguousarray(padded_feats.astype(np.float32))
            elif name == "speech_lengths":
                inputs[name] = np.ascontiguousarray(lengths.astype(np.int32))
            elif "bias_embed" in name:
                if bias_embeddings is not None:
                    inputs[name] = np.ascontiguousarray(
                        np.tile(bias_embeddings, (batch_size, 1, 1)).astype(np.float32)
                    )
                else:
                    inputs[name] = np.ascontiguousarray(
                        np.zeros((batch_size, 1, 512), dtype=np.float32)
                    )

        # 分配 GPU 内存
        d_inputs = {}
        for name, data in inputs.items():
            d_inputs[name] = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod_async(d_inputs[name], data, self._stream)

        # 分配输出内存
        d_outputs = {}
        h_outputs = {}
        for name in self._output_names:
            # 获取输出 shape（在设置 input shape 后可以推断）
            output_shape = self._context.get_tensor_shape(name)
            output_size = int(np.prod(output_shape))
            h_outputs[name] = np.empty(output_shape, dtype=np.float32)
            d_outputs[name] = cuda.mem_alloc(h_outputs[name].nbytes)

        # 绑定地址
        for name in self._input_names:
            self._context.set_tensor_address(name, int(d_inputs[name]))
        for name in self._output_names:
            self._context.set_tensor_address(name, int(d_outputs[name]))

        # 执行推理
        self._context.execute_async_v3(stream_handle=self._stream.handle)

        # 拷贝输出
        for name in self._output_names:
            cuda.memcpy_dtoh_async(h_outputs[name], d_outputs[name], self._stream)

        self._stream.synchronize()

        # 释放 GPU 内存
        for d in d_inputs.values():
            d.free()
        for d in d_outputs.values():
            d.free()

        # 拆分 batch 结果
        logits = h_outputs[self._output_names[0]]  # (batch, seq, vocab)
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
