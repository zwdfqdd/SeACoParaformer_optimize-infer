"""
VAD 模块 — 语音活动检测

使用 Silero VAD 官方 ONNX 模型进行静音检测切段。
VAD 只输出时间戳列表 [(start_ms, end_ms), ...]，不修改音频数据。
运行在 CPU 上，与 ASR 推理并行。
"""

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from src.config import settings
from src.errors import ASRException, ErrorCode
from src.logger import logger


@dataclass
class VADSegment:
    """VAD 检测到的语音段，保留原始时间戳。"""
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class SileroVAD:
    """
    Silero VAD ONNX 推理引擎。

    配置：
    - min_speech_duration: 最小语音段时长 0.5s
    - 段间合并策略：相邻段间隔 < 300ms 则合并
    """

    SAMPLE_RATE = 16000
    WINDOW_SIZE = 512  # Silero VAD 窗口大小（16kHz 下约 32ms）
    MIN_SPEECH_DURATION_MS = 500  # 最小语音段 0.5s
    MERGE_GAP_MS = 300  # 段间合并阈值

    def __init__(self):
        self._session: ort.InferenceSession | None = None
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def load(self):
        """加载 VAD ONNX 模型（CPU）。"""
        model_path = settings.get_vad_model_path()
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            logger.info(f"VAD 模型加载成功: {model_path}")
        except Exception as e:
            raise ASRException(
                ErrorCode.MODEL_LOAD_FAILED,
                f"VAD 模型加载失败: {e}",
            )

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def detect(self, pcm: np.ndarray, sample_rate: int = 16000) -> list[VADSegment]:
        """
        对 PCM 音频进行 VAD 检测。

        参数:
            pcm: float32 PCM 数据，shape=(samples,)
            sample_rate: 采样率（必须为 16000）

        返回:
            语音段时间戳列表 [(start_ms, end_ms), ...]
        """
        if self._session is None:
            raise ASRException(ErrorCode.VAD_SEGMENT_ERROR, "VAD 模型未加载")

        try:
            raw_segments = self._run_vad(pcm, sample_rate)
            merged_segments = self._merge_segments(raw_segments)
            filtered_segments = self._filter_short(merged_segments)
            return filtered_segments
        except ASRException:
            raise
        except Exception as e:
            raise ASRException(
                ErrorCode.VAD_SEGMENT_ERROR,
                f"VAD 推理异常: {e}",
            )

    def _run_vad(
        self, pcm: np.ndarray, sample_rate: int
    ) -> list[VADSegment]:
        """逐窗口运行 VAD，收集语音概率并提取语音段。"""
        num_samples = len(pcm)
        speeches: list[VADSegment] = []
        is_speech = False
        speech_start = 0
        threshold = 0.5

        # 重置状态
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        sr = np.array([sample_rate], dtype=np.int64)

        for i in range(0, num_samples, self.WINDOW_SIZE):
            chunk = pcm[i: i + self.WINDOW_SIZE]
            if len(chunk) < self.WINDOW_SIZE:
                chunk = np.pad(chunk, (0, self.WINDOW_SIZE - len(chunk)))

            input_data = chunk.reshape(1, -1).astype(np.float32)

            # Silero VAD 输入: input, sr, h, c
            inputs = {
                "input": input_data,
                "sr": sr,
                "h": h,
                "c": c,
            }

            try:
                output, hn, cn = self._session.run(None, inputs)
                h = hn
                c = cn
            except Exception:
                # 某些版本的 Silero VAD 输入格式不同，尝试备选
                inputs_alt = {"input": input_data, "sr": sr, "state": h}
                results = self._session.run(None, inputs_alt)
                output = results[0]
                if len(results) > 1:
                    h = results[1]

            prob = float(output.flatten()[0])
            current_ms = int(i / sample_rate * 1000)

            if prob >= threshold and not is_speech:
                is_speech = True
                speech_start = current_ms
            elif prob < threshold and is_speech:
                is_speech = False
                speeches.append(VADSegment(
                    start_ms=speech_start,
                    end_ms=current_ms,
                ))

        # 处理末尾未关闭的语音段
        if is_speech:
            end_ms = int(num_samples / sample_rate * 1000)
            speeches.append(VADSegment(start_ms=speech_start, end_ms=end_ms))

        return speeches

    def _merge_segments(self, segments: list[VADSegment]) -> list[VADSegment]:
        """合并相邻且间隔小于阈值的语音段。"""
        if not segments:
            return []

        merged: list[VADSegment] = [segments[0]]

        for seg in segments[1:]:
            last = merged[-1]
            gap = seg.start_ms - last.end_ms

            if gap <= self.MERGE_GAP_MS:
                # 合并：扩展上一段的结束时间
                merged[-1] = VADSegment(
                    start_ms=last.start_ms,
                    end_ms=seg.end_ms,
                )
            else:
                merged.append(seg)

        return merged

    def _filter_short(self, segments: list[VADSegment]) -> list[VADSegment]:
        """过滤掉短于最小时长的语音段。"""
        return [
            seg for seg in segments
            if seg.duration_ms >= self.MIN_SPEECH_DURATION_MS
        ]


# 全局单例
vad_engine = SileroVAD()
