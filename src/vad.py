"""
VAD 模块 — 语音活动检测

使用 Silero VAD 官方 ONNX 模型进行静音检测切段。
对齐官方 OnnxWrapper 实现（context 拼接、state shape=(2,batch,128)）。
VAD 只输出时间戳列表 [(start_ms, end_ms), ...]，不修改音频数据。
运行在 CPU 上，与 ASR 推理并行。

并发设计（Session Pool + Round-Robin）：
    ORT InferenceSession 并发调用 run() 会在内部串行化，高并发下单请求 VAD 墙钟耗时
    被放大数倍（10 并发实测从 500ms 涨到 3300ms）。
    解决方案：load() 时一次性预建 N 个 session（VAD_SESSION_POOL_SIZE），请求按
    无锁 counter round-robin 分配。多 session 之间真正并行；单 session 内即使被多
    线程调用也是安全的（ORT session.run 线程安全，只是串行化）。
    对比 threading.local 方案：
      - 无懒加载竞态（避免 ORT 首次并发创建时 double-free）
      - Session 数量固定可控（不会因线程数爆炸）
      - 内存占用可预测（每 session ~5MB × pool_size）
"""

import itertools
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

    Session 池：load() 一次性预建 N 个 session，请求按 round-robin 分配。
    """

    def __init__(self):
        self._sessions: list[ort.InferenceSession] = []
        # 无锁 round-robin 计数器（itertools.count 是线程安全的原子递增）
        self._counter = itertools.count()

    def load(self):
        """加载 VAD ONNX 模型：一次性预建 VAD_SESSION_POOL_SIZE 个 session。"""
        model_path = settings.get_vad_model_path()
        pool_size = max(1, settings.VAD_SESSION_POOL_SIZE)
        try:
            self._sessions = [self._new_session(model_path) for _ in range(pool_size)]
            logger.info(
                f"VAD 模型加载成功: {model_path}（session 池大小={pool_size}, CPU）"
            )
        except Exception as e:
            raise ASRException(
                ErrorCode.MODEL_LOAD_FAILED,
                f"VAD 模型加载失败: {e}",
            )

    @staticmethod
    def _new_session(model_path: str) -> ort.InferenceSession:
        sess_options = ort.SessionOptions()
        # 单 session 内单线程即可（多 session 之间并行，避免线程超额订阅）
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        # 禁用 arena/mem_pattern：ORT 内部 arena 分配在高并发下有已知竞态（libgomp
        # thread creation failed / free() corrupted chunks）。禁用后每次 run 独立分配，
        # 单调用略慢 5-10%，但消除多线程调 session 的并发竞态。
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        # Silero VAD 固定在 CPU：模型极小 (<10MB)，单次调用只处理 (1,576) tensor，
        # 30s 音频 937 次 session.run。压测验证过：上 GPU 反而慢 4x（PCIe 往返开销
        # 远大于 GPU kernel 计算），且 GPU sm-util 假高（大部分是 memory 等待）。
        return ort.InferenceSession(
            model_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )

    def _get_session(self) -> ort.InferenceSession:
        """无锁 round-robin 分配 session。"""
        if not self._sessions:
            raise ASRException(ErrorCode.VAD_SEGMENT_ERROR, "VAD 模型未加载")
        idx = next(self._counter) % len(self._sessions)
        return self._sessions[idx]

    @property
    def is_loaded(self) -> bool:
        return len(self._sessions) > 0

    def detect(self, pcm: np.ndarray, sample_rate: int = 16000) -> list[VADSegment]:
        """
        对 PCM 音频进行 VAD 检测。
        线程安全：每次调用独立维护 state，session 从池中 round-robin 分配。
        """
        if not self._sessions:
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

        # 获取当前线程独立的 session（首次调用时懒加载）
        session = self._get_session()

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
            ort_outs = session.run(None, ort_inputs)
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
