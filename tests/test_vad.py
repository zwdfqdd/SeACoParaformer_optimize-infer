"""
VAD 单独测试脚本

输入音频，输出语音段时间戳。
使用 Silero VAD ONNX 模型，CPU 推理。
对齐官方 OnnxWrapper 实现。

用法：
    python tests/test_vad.py --audio test_data/audio_16000_30s.wav
    python tests/test_vad.py --audio test_data/audio_16000_30s.wav --vad-model ./models/vad/silero_vad.onnx
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


# VAD 参数
SAMPLE_RATE = 16000
WINDOW_SIZE = 512  # 16kHz 下每窗口 512 samples（32ms）
CONTEXT_SIZE = 64  # 上下文 samples（官方 OnnxWrapper 拼接到输入前）
THRESHOLD = 0.5
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 100
SPEECH_PAD_MS = 30


class SileroVADRunner:
    """对齐官方 OnnxWrapper 的 ONNX 推理实现。"""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset_states()

    def reset_states(self, batch_size=1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros((batch_size, CONTEXT_SIZE), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> float:
        """
        x: shape (batch, 512) float32 PCM
        返回: 语音概率 float
        """
        # 拼接 context（官方逻辑）
        x_with_context = np.concatenate([self._context, x], axis=1)  # (batch, 576)

        ort_inputs = {
            "input": x_with_context,
            "state": self._state,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, state = ort_outs[0], ort_outs[1]
        self._state = state

        # 保存 context（最后 64 samples）
        self._context = x_with_context[:, -CONTEXT_SIZE:]

        return float(out.flatten()[0])


def get_speech_timestamps(pcm: np.ndarray, vad: SileroVADRunner) -> list[dict]:
    """
    对齐官方 get_speech_timestamps 的简化版本。
    返回 [{"start": sample, "end": sample}, ...]
    """
    audio_length = len(pcm)
    speech_probs = []

    vad.reset_states()

    # 逐窗口推理
    for i in range(0, audio_length, WINDOW_SIZE):
        chunk = pcm[i: i + WINDOW_SIZE]
        if len(chunk) < WINDOW_SIZE:
            chunk = np.pad(chunk, (0, WINDOW_SIZE - len(chunk)))
        chunk = chunk.reshape(1, -1).astype(np.float32)
        prob = vad(chunk)
        speech_probs.append(prob)

    # 后处理（对齐官方逻辑）
    neg_threshold = max(THRESHOLD - 0.15, 0.01)
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

        if prob < neg_threshold and triggered:
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


def main():
    global THRESHOLD, MIN_SPEECH_MS, MIN_SILENCE_MS, SPEECH_PAD_MS

    parser = argparse.ArgumentParser(description="VAD 测试")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--vad-model", default="./models/vad/silero_vad.onnx")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--min-speech", type=int, default=MIN_SPEECH_MS, help="最小语音段(ms)")
    parser.add_argument("--min-silence", type=int, default=MIN_SILENCE_MS, help="最小静音段(ms)")
    parser.add_argument("--speech-pad", type=int, default=SPEECH_PAD_MS, help="语音段 padding(ms)")
    args = parser.parse_args()

    THRESHOLD = args.threshold
    MIN_SPEECH_MS = args.min_speech
    MIN_SILENCE_MS = args.min_silence
    SPEECH_PAD_MS = args.speech_pad

    if not Path(args.audio).exists():
        sys.exit(f"错误：音频不存在: {args.audio}")
    if not Path(args.vad_model).exists():
        sys.exit(f"错误：VAD 模型不存在: {args.vad_model}")

    # 加载音频
    pcm, sr = sf.read(args.audio, dtype="float32")
    if len(pcm.shape) > 1:
        pcm = pcm[:, 0]
    audio_duration = len(pcm) / sr

    print("=" * 50)
    print("Silero VAD 测试")
    print("=" * 50)
    print(f"音频: {args.audio}")
    print(f"时长: {audio_duration:.2f}s ({len(pcm)} samples)")
    print(f"模型: {args.vad_model}")
    print(f"参数: threshold={THRESHOLD}, min_speech={MIN_SPEECH_MS}ms, min_silence={MIN_SILENCE_MS}ms, pad={SPEECH_PAD_MS}ms")
    print()

    # 加载模型
    vad = SileroVADRunner(args.vad_model)
    inputs = vad.session.get_inputs()
    print(f"模型输入: {[(i.name, i.shape) for i in inputs]}")

    # 运行 VAD
    t0 = time.perf_counter()
    speeches = get_speech_timestamps(pcm, vad)
    vad_time = time.perf_counter() - t0

    # 输出结果
    print(f"\nVAD 耗时: {vad_time*1000:.1f}ms (RTF: {vad_time/audio_duration:.4f}, RTX: {audio_duration/vad_time:.1f}x)")
    print(f"检测到 {len(speeches)} 个语音段")
    print()

    print("-" * 60)
    print(f"{'#':<4} {'开始(ms)':<10} {'结束(ms)':<10} {'时长(ms)':<10} {'时间范围'}")
    print("-" * 60)
    total_speech = 0
    for i, seg in enumerate(speeches):
        start_ms = int(seg["start"] / SAMPLE_RATE * 1000)
        end_ms = int(seg["end"] / SAMPLE_RATE * 1000)
        duration_ms = end_ms - start_ms
        total_speech += duration_ms
        print(f"{i:<4} {start_ms:<10} {end_ms:<10} {duration_ms:<10} {start_ms/1000:.2f}s - {end_ms/1000:.2f}s")

    print("-" * 60)
    print(f"语音总时长: {total_speech/1000:.2f}s / {audio_duration:.2f}s ({total_speech/audio_duration/10:.1f}%)")


if __name__ == "__main__":
    main()
