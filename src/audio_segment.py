"""
音频切段模块

负责将 VAD 输出的语音段按照规则进行二次切分：
- min=5s, opt=12s, max=15s
- 第一段 <5s 拼接到下一段前部
- 最后一段 <5s 拼接到上一段尾部
- chunk 间增加 200ms overlap
- 切分只操作时间戳区间，最后按区间从原始 PCM slice

全流程仅维护一份原始 PCM 数据（PCM 生命周期管理）。
"""

from dataclasses import dataclass, field

import numpy as np

from src.errors import ASRException, ErrorCode
from src.vad import VADSegment


# 切段参数（毫秒）
MIN_CHUNK_MS = 5000
OPT_CHUNK_MS = 12000
MAX_CHUNK_MS = 15000
OVERLAP_MS = 200


@dataclass
class ChunkMeta:
    """Chunk 元数据，维护与原始音频的映射关系。"""
    chunk_id: int
    segment_index: int  # 来源 VAD segment 索引
    raw_start_ms: int  # 原始音频中的起始时间
    raw_end_ms: int  # 原始音频中的结束时间
    overlap_left_ms: int = 0  # 左侧 overlap 长度
    overlap_right_ms: int = 0  # 右侧 overlap 长度

    @property
    def duration_ms(self) -> int:
        return self.raw_end_ms - self.raw_start_ms

    @property
    def effective_start_ms(self) -> int:
        """去除 overlap 后的有效起始时间。"""
        return self.raw_start_ms + self.overlap_left_ms

    @property
    def effective_end_ms(self) -> int:
        """去除 overlap 后的有效结束时间。"""
        return self.raw_end_ms - self.overlap_right_ms


def segment_to_chunks(
    vad_segments: list[VADSegment],
    total_duration_ms: int,
) -> list[ChunkMeta]:
    """
    将 VAD 语音段切分为适合 ASR 推理的 chunk。

    流程：
    1. 对每个 VAD segment 按 opt/max 长度切分
    2. 处理首尾短段合并
    3. 添加 overlap

    参数:
        vad_segments: VAD 检测到的语音段列表
        total_duration_ms: 原始音频总时长（毫秒）

    返回:
        ChunkMeta 列表
    """
    if not vad_segments:
        raise ASRException(
            ErrorCode.AUDIO_SEGMENT_ERROR,
            "无有效语音段，音频可能为静音",
        )

    # Step 1: 对每个 segment 按长度切分
    raw_chunks: list[ChunkMeta] = []
    chunk_id = 0

    for seg_idx, seg in enumerate(vad_segments):
        seg_chunks = _split_segment(seg, seg_idx, chunk_id)
        raw_chunks.extend(seg_chunks)
        chunk_id += len(seg_chunks)

    # Step 2: 处理首尾短段
    raw_chunks = _merge_short_edges(raw_chunks)

    # Step 3: 重新编号
    for i, chunk in enumerate(raw_chunks):
        chunk.chunk_id = i

    # Step 4: 添加 overlap
    chunks_with_overlap = _add_overlap(raw_chunks, total_duration_ms)

    return chunks_with_overlap


def _split_segment(
    segment: VADSegment, seg_idx: int, start_chunk_id: int
) -> list[ChunkMeta]:
    """将单个 VAD segment 按 opt/max 长度切分。"""
    duration = segment.duration_ms
    chunks: list[ChunkMeta] = []

    if duration <= MAX_CHUNK_MS:
        # 不需要切分
        chunks.append(ChunkMeta(
            chunk_id=start_chunk_id,
            segment_index=seg_idx,
            raw_start_ms=segment.start_ms,
            raw_end_ms=segment.end_ms,
        ))
        return chunks

    # 按 opt 长度切分
    current_start = segment.start_ms
    chunk_id = start_chunk_id

    while current_start < segment.end_ms:
        remaining = segment.end_ms - current_start

        if remaining <= MAX_CHUNK_MS:
            # 剩余部分不超过 max，直接作为最后一个 chunk
            chunks.append(ChunkMeta(
                chunk_id=chunk_id,
                segment_index=seg_idx,
                raw_start_ms=current_start,
                raw_end_ms=segment.end_ms,
            ))
            break
        else:
            # 按 opt 长度切
            chunk_end = current_start + OPT_CHUNK_MS
            chunks.append(ChunkMeta(
                chunk_id=chunk_id,
                segment_index=seg_idx,
                raw_start_ms=current_start,
                raw_end_ms=chunk_end,
            ))
            current_start = chunk_end
            chunk_id += 1

    return chunks


def _merge_short_edges(chunks: list[ChunkMeta]) -> list[ChunkMeta]:
    """处理首尾短段合并。"""
    if len(chunks) <= 1:
        return chunks

    # 第一段 <5s：拼接到下一段前部
    if chunks[0].duration_ms < MIN_CHUNK_MS and len(chunks) > 1:
        chunks[1] = ChunkMeta(
            chunk_id=chunks[1].chunk_id,
            segment_index=chunks[1].segment_index,
            raw_start_ms=chunks[0].raw_start_ms,
            raw_end_ms=chunks[1].raw_end_ms,
        )
        chunks = chunks[1:]

    if len(chunks) <= 1:
        return chunks

    # 最后一段 <5s：拼接到上一段尾部
    if chunks[-1].duration_ms < MIN_CHUNK_MS:
        chunks[-2] = ChunkMeta(
            chunk_id=chunks[-2].chunk_id,
            segment_index=chunks[-2].segment_index,
            raw_start_ms=chunks[-2].raw_start_ms,
            raw_end_ms=chunks[-1].raw_end_ms,
        )
        chunks = chunks[:-1]

    return chunks


def _add_overlap(
    chunks: list[ChunkMeta], total_duration_ms: int
) -> list[ChunkMeta]:
    """为 chunk 添加 overlap 区域。"""
    if len(chunks) <= 1:
        return chunks

    result: list[ChunkMeta] = []

    for i, chunk in enumerate(chunks):
        overlap_left = 0
        overlap_right = 0

        if i > 0:
            # 左侧 overlap：向前扩展
            overlap_left = min(OVERLAP_MS, chunk.raw_start_ms)

        if i < len(chunks) - 1:
            # 右侧 overlap：向后扩展
            overlap_right = min(OVERLAP_MS, total_duration_ms - chunk.raw_end_ms)

        result.append(ChunkMeta(
            chunk_id=chunk.chunk_id,
            segment_index=chunk.segment_index,
            raw_start_ms=chunk.raw_start_ms - overlap_left,
            raw_end_ms=chunk.raw_end_ms + overlap_right,
            overlap_left_ms=overlap_left,
            overlap_right_ms=overlap_right,
        ))

    return result


def extract_chunk_audio(
    pcm: np.ndarray, chunk: ChunkMeta, sample_rate: int = 16000
) -> np.ndarray:
    """
    从原始 PCM 中按 chunk 时间戳提取音频片段。

    全流程仅维护一份原始 PCM 数据，此处只做 slice 不做拷贝。
    """
    start_sample = int(chunk.raw_start_ms * sample_rate / 1000)
    end_sample = int(chunk.raw_end_ms * sample_rate / 1000)

    # 边界保护
    start_sample = max(0, start_sample)
    end_sample = min(len(pcm), end_sample)

    return pcm[start_sample:end_sample]
