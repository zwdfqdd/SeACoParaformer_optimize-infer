"""
音频切段模块

VAD 后音频段处理方案：
Step 1：合并相邻短段（合并后不超过最大桶 8s）
Step 2：超长段按 8s 切分
Step 3：就近桶归类（2s/4s/8s）
Step 4：最后一段 < 2s 时合并到前一段

桶边界：2s, 4s, 8s（LFR 帧数 34, 67, 134）
时间戳保留原始 VAD 位置。
"""

from dataclasses import dataclass

import numpy as np

from src.errors import ASRException, ErrorCode
from src.vad import VADSegment


# 桶边界（毫秒）
BUCKET_MS = [2000, 4000, 8000]
MIN_BUCKET_MS = BUCKET_MS[0]   # 2s
MAX_BUCKET_MS = BUCKET_MS[-1]  # 8s


@dataclass
class ChunkMeta:
    """Chunk 元数据，维护与原始音频的映射关系。"""
    chunk_id: int
    segment_index: int
    raw_start_ms: int
    raw_end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.raw_end_ms - self.raw_start_ms

    @property
    def effective_start_ms(self) -> int:
        return self.raw_start_ms

    @property
    def effective_end_ms(self) -> int:
        return self.raw_end_ms


def segment_to_chunks(
    vad_segments: list[VADSegment],
    total_duration_ms: int,
) -> list[ChunkMeta]:
    """
    将 VAD 语音段处理为固定桶边界的 chunk。

    Step 1: 合并相邻短段（合并后 ≤ 8s）
    Step 2: 超长段按 8s 切分
    Step 3: 最后一段 < 2s 时合并到前一段
    """
    if not vad_segments:
        raise ASRException(
            ErrorCode.AUDIO_SEGMENT_ERROR,
            "无有效语音段，音频可能为静音",
        )

    # Step 1: 合并相邻短段
    merged = _merge_short_segments(vad_segments)

    # Step 2: 超长段切分
    split = _split_long_segments(merged)

    # Step 3: 最后一段 < 2s 合并到前一段
    split = _merge_trailing_short(split)

    # 生成 ChunkMeta
    chunks = []
    for i, seg in enumerate(split):
        chunks.append(ChunkMeta(
            chunk_id=i,
            segment_index=0,
            raw_start_ms=seg["start_ms"],
            raw_end_ms=seg["end_ms"],
        ))

    return chunks


def _merge_short_segments(segments: list[VADSegment]) -> list[dict]:
    """
    Step 1: 合并相邻短段。
    遍历 VAD 段，将相邻段合并直到满足最小桶长度（2s），合并后不超过最大桶（8s）。
    合并范围：第一个段的 start_ms 到最后一个段的 end_ms（包含中间静音）。
    """
    if not segments:
        return []

    merged = []
    current_start = segments[0].start_ms
    current_end = segments[0].end_ms

    for i in range(1, len(segments)):
        seg = segments[i]
        # 尝试合并：合并后总时长不超过最大桶
        merged_duration = seg.end_ms - current_start
        if merged_duration <= MAX_BUCKET_MS:
            # 合并
            current_end = seg.end_ms
        else:
            # 不能合并，保存当前段，开始新段
            merged.append({"start_ms": current_start, "end_ms": current_end})
            current_start = seg.start_ms
            current_end = seg.end_ms

    # 保存最后一段
    merged.append({"start_ms": current_start, "end_ms": current_end})

    return merged


def _split_long_segments(segments: list[dict]) -> list[dict]:
    """
    Step 2: 超长段按 8s 切分。
    """
    result = []
    for seg in segments:
        duration = seg["end_ms"] - seg["start_ms"]
        if duration <= MAX_BUCKET_MS:
            result.append(seg)
        else:
            current_start = seg["start_ms"]
            while current_start < seg["end_ms"]:
                chunk_end = min(current_start + MAX_BUCKET_MS, seg["end_ms"])
                result.append({"start_ms": current_start, "end_ms": chunk_end})
                current_start += MAX_BUCKET_MS
    return result


def _merge_trailing_short(segments: list[dict]) -> list[dict]:
    """
    Step 3: 最后一段 < 2s 时合并到前一段（如果合并后 ≤ 8s）。
    """
    if len(segments) <= 1:
        return segments

    last = segments[-1]
    last_duration = last["end_ms"] - last["start_ms"]

    if last_duration < MIN_BUCKET_MS and len(segments) >= 2:
        prev = segments[-2]
        merged_duration = last["end_ms"] - prev["start_ms"]
        if merged_duration <= MAX_BUCKET_MS:
            # 合并到前一段
            segments[-2] = {"start_ms": prev["start_ms"], "end_ms": last["end_ms"]}
            segments = segments[:-1]

    return segments


def extract_chunk_audio(
    pcm: np.ndarray, chunk: ChunkMeta, sample_rate: int = 16000
) -> np.ndarray:
    """从原始 PCM 中按 chunk 时间戳提取音频片段（slice，不拷贝）。"""
    start_sample = int(chunk.raw_start_ms * sample_rate / 1000)
    end_sample = int(chunk.raw_end_ms * sample_rate / 1000)
    start_sample = max(0, start_sample)
    end_sample = min(len(pcm), end_sample)
    return pcm[start_sample:end_sample]
