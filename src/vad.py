"""
VAD 模块 — 语音活动检测

使用 Silero VAD 官方 ONNX 模型进行静音检测切段。
对齐官方 OnnxWrapper 实现（context 拼接、state shape=(2,batch,128)）。
VAD 只输出时间戳列表 [(start_ms, end_ms), ...]，不修改音频数据。
运行在 CPU 上，与 ASR 推理并行。
"""

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from src.config import settings
from src.errors import ASRException, ErrorCode
from src.logger import logger


# VAD 参数
SAMPLE_RATE = 16000
WINDOW_SIZE = 512  # 16kHz 下每窗口 512 samples
CONTEXT_SIZE = 64  # 上下文 samples（拼接到输入前）
THRESHOLD = 0.5
NEG_THRESHOLD = 0.35  # threshold - 0.15
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 100
SPEECH_PAD_MS = 30


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
    对齐官方 OnnxWrapper 实现。
    """

    def __init__(self):
        self._session: ort.InferenceSession | None = None

    def load(self):
        """加载 VAD ONNX 模型（CPU）。"""
        model_path = settings.get_vad_model_path()
        try:
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
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
        线程安全：每次调用独立维护 state，不共享实例状态。
        """
        if self._session is None:
            raise ASRException(ErrorCode.VAD_SEGMENT_ERROR, "VAD 模型未加载")

        try:
            speeches = self._get_speech_timestamps(pcm)
            segments = []
            for s in speeches:
                start_ms = int(s["start"] / SAMPLE_RATE * 1000)
                end_ms = int(s["end"] / SAMPLE_RATE * 1000)
                segments.append(VADSegment(start_ms=start_ms, end_ms=end_ms))
            return segments
        except ASRException:
            raise
        except Exception as e:
            raise ASRException(
                ErrorCode.VAD_SEGMENT_ERROR,
                f"VAD 推理异常: {e}",
            )

    def _get_speech_timestamps(self, pcm: np.ndarray) -> list[dict]:
        """
        对齐官方 get_speech_timestamps 逻辑。
        state 作为局部变量，保证线程安全。
        """
        audio_length = len(pcm)

        # 局部状态（线程安全）
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT_SIZE), dtype=np.float32)

        # 逐窗口推理获取概率
        speech_probs = []
        for i in range(0, audio_length, WINDOW_SIZE):
            chunk = pcm[i: i + WINDOW_SIZE]
            if len(chunk) < WINDOW_SIZE:
                chunk = np.pad(chunk, (0, WINDOW_SIZE - len(chunk)))
            chunk = chunk.reshape(1, -1).astype(np.float32)

            # 拼接 context
            x = np.concatenate([context, chunk], axis=1)
            ort_inputs = {
                "input": x,
                "state": state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            }
            ort_outs = self._session.run(None, ort_inputs)
            out, state = ort_outs[0], ort_outs[1]
            context = x[:, -CONTEXT_SIZE:]

            prob = float(out.flatten()[0])
            speech_probs.append(prob)

        # 后处理
        min_speech_samples = SAMPLE_RATE * MIN_SPEECH_MS / 1000
        min_silence_samples = SAMPLE_RATE * MIN_SILENCE_MS / 1000
        speech_pad_samples = SAMPLE_RATE * SPEECH_PAD_MS / 1000

        triggered = False
        speeches = []
        current_speech = {}
        temp_end = 0

        for i, prob in enumerate(speech_probs):
            cur_sample = WINDOW_SIZE * i

            if prob >= THRESHOLD and temp_end:
                temp_end = 0

            if prob >= THRESHOLD and not triggered:
                triggered = True
                current_speech["start"] = cur_sample
                continue

            if prob < NEG_THRESHOLD and triggered:
                if not temp_end:
                    temp_end = cur_sample
                if cur_sample - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech["end"] = temp_end
                    if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                        speeches.append(current_speech)
                    current_speech = {}
                    temp_end = 0
                    triggered = False
                    continue

        if current_speech and (audio_length - current_speech.get("start", audio_length)) > min_speech_samples:
            current_speech["end"] = audio_length
            speeches.append(current_speech)

        # padding
        for i, speech in enumerate(speeches):
            if i == 0:
                speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]["start"] - speech["end"]
                if silence_duration < 2 * speech_pad_samples:
                    speech["end"] += int(silence_duration // 2)
                    speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - silence_duration // 2))
                else:
                    speech["end"] = int(min(audio_length, speech["end"] + speech_pad_samples))
                    speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - speech_pad_samples))
            else:
                speech["end"] = int(min(audio_length, speech["end"] + speech_pad_samples))

        return speeches


# 全局单例
vad_engine = SileroVAD()
