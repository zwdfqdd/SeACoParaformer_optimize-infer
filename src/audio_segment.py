"""
音频切段模块（Uniform Chunking 均匀切段）

策略：VAD 后从第一段 start 到最后段 end 作为整体时间轴，按统一 UNIFORM_CHUNK_MS
（= TRT_OPT_SEQ 帧对应的毫秒数，默认 67 帧 × 60ms = 4020ms）线性切分。

优势：
- 所有 chunk 长度一致 → scheduler 无需桶分组 → 合批天花板显著提升（avg_batch 3-4 倍）
- 每 chunk 恰好走 TRT engine 的 opt profile → GPU 命中率最高
- 尾段不足 opt 时由 scheduler pad 到 opt（浪费仅限一个尾 chunk）

代价：
- VAD 段边界被硬切时，识别精度可能微降（Encoder self-attention 跨 chunk 断裂）
- 若 VAD 段间静音长度 < opt，静音也会被切进 chunk 送入 encoder（合理，无浪费）
- 若 VAD 段间静音 >> opt，会产生纯静音 chunk（罕见场景，可后续加静音过滤优化）

时间戳保留原始 VAD 起止位置的连续区间。
"""

from dataclasses import dataclass

import numpy as np

from src.config import settings
from src.errors import ASRException, ErrorCode
from src.vad import VADSegment


# LFR 帧对齐的毫秒尺度：1 LFR 帧 = LFR_N(6) × FRAME_SHIFT_MS(10) = 60ms
_MS_PER_LFR_FRAME = 60
# 统一切段目标：TRT opt profile 主力工作点，压中 opt kernel 使 GPU 效率最优
# 默认 67 帧 × 60ms = 4020ms（4s），可通过 TRT_OPT_SEQ 调整
UNIFORM_CHUNK_MS = settings.TRT_OPT_SEQ * _MS_PER_LFR_FRAME
# 尾块保护：切分后最后一 chunk 若小于 MIN_TAIL_MS，合并到前一 chunk（避免识别精度损失）
# 合并后前 chunk 可能超过 opt，最大不超过 UNIFORM_CHUNK_MS + MIN_TAIL_MS
MIN_TAIL_MS = 1000
# 兼容保留：其他模块可能引用
BUCKET_MS = [int(f * _MS_PER_LFR_FRAME) for f in settings.BUCKET_SEQ_LENS]
MIN_BUCKET_MS = BUCKET_MS[0]
MAX_BUCKET_MS = BUCKET_MS[-1]
# 单 chunk 最大时长上限（TRT profile max seq 帧 - 1 帧余量，防溢出）
SPLIT_MAX_MS = MAX_BUCKET_MS - _MS_PER_LFR_FRAME


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
    将 VAD 语音段均匀切分为 UNIFORM_CHUNK_MS 长度的 chunk（尾块由 scheduler pad）。

    流程：
      1. 取 VAD 的整体时间跨度：[first.start_ms, last.end_ms]
      2. 从 first.start_ms 起，每 UNIFORM_CHUNK_MS 切一刀
      3. 尾块保留实际长度（可能 < UNIFORM_CHUNK_MS），scheduler 按 TRT opt seq 帧数 pad

    所有 chunk 长度一致（除尾块）→ scheduler 无需桶分组 → 合批天花板显著提升。
    """
    if not vad_segments:
        raise ASRException(
            ErrorCode.AUDIO_SEGMENT_ERROR,
            "无有效语音段，音频可能为静音",
        )

    # 取 VAD 整体时间跨度（首段 start 到末段 end）
    speech_start = vad_segments[0].start_ms
    speech_end = vad_segments[-1].end_ms
    if speech_end <= speech_start:
        raise ASRException(
            ErrorCode.AUDIO_SEGMENT_ERROR,
            "VAD 段时间戳异常",
        )

    # 均匀切分
    chunks: list[ChunkMeta] = []
    cur = speech_start
    idx = 0
    while cur < speech_end:
        nxt = min(cur + UNIFORM_CHUNK_MS, speech_end)
        chunks.append(ChunkMeta(
            chunk_id=idx,
            segment_index=0,
            raw_start_ms=cur,
            raw_end_ms=nxt,
        ))
        cur = nxt
        idx += 1

    # 尾块保护：末段 < MIN_TAIL_MS 时并入前一段
    # 避免过短尾块因上下文不足引起识别错误/幻听；
    # 合并后前段长度最多为 UNIFORM_CHUNK_MS + MIN_TAIL_MS（默认 5020ms，未超 8s max 桶）
    if len(chunks) >= 2:
        tail = chunks[-1]
        tail_ms = tail.raw_end_ms - tail.raw_start_ms
        if tail_ms < MIN_TAIL_MS:
            prev = chunks[-2]
            merged = ChunkMeta(
                chunk_id=prev.chunk_id,
                segment_index=prev.segment_index,
                raw_start_ms=prev.raw_start_ms,
                raw_end_ms=tail.raw_end_ms,
            )
            chunks = chunks[:-2] + [merged]

    return chunks


def extract_chunk_audio(
    pcm: np.ndarray, chunk: ChunkMeta, sample_rate: int = 16000
) -> np.ndarray:
    """从原始 PCM 中按 chunk 时间戳提取音频片段（slice，不拷贝）。"""
    start_sample = int(chunk.raw_start_ms * sample_rate / 1000)
    end_sample = int(chunk.raw_end_ms * sample_rate / 1000)
    start_sample = max(0, start_sample)
    end_sample = min(len(pcm), end_sample)
    return pcm[start_sample:end_sample]
