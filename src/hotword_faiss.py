"""
路径 B：Faiss 大词库后处理纠错

触发条件：客户端未传 hotwords（默认词表恒走本路径，不再按 MAX_HOTWORD_NUM 大小切换）
    且 ENABLE_FAISS_CORRECTION=true。

流程：
    普通 ASR 输出 text
    → 滑窗切片（window=2/3/4 字）生成候选片段
    → 候选片段 → 拼音序列 → 拼音向量
    → Faiss IndexFlatIP 检索 TopK
    → final_score = 拼音相似×0.75 + 编辑距离分×0.25
    → 三重联合判定（检索分 + 区分度 + 融合分）
    → 通过则替换纠错

拼音向量化：
    词 → pypinyin 无声调音节序列 → 在音节词表上构建 multi-hot 计数向量 → L2 归一化
    IndexFlatIP 在归一化向量上即余弦相似度。

依赖（懒加载，缺失时纠错功能自动禁用，不影响主流程）：
    faiss-cpu, pypinyin, python-Levenshtein
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import numpy as np

from src.config import settings
from src.logger import logger


class _FaissState(NamedTuple):
    """索引快照（不可变，一次性原子切换，避免 correct 读到 build 中间态）。"""
    index: object                    # faiss.IndexFlatIP
    words: tuple[str, ...]           # 词表（与 index 行对齐）
    word_pinyin: tuple[str, ...]     # 各词拼音串（编辑距离用）
    syllable_to_idx: dict            # 音节 → 列索引
    dim: int
    version: int

# 懒加载第三方依赖（缺失时禁用纠错，不阻塞服务）
try:
    import faiss  # type: ignore
    from pypinyin import lazy_pinyin  # type: ignore
    import Levenshtein  # type: ignore
    _HAS_DEPS = True
except ImportError as _e:  # pragma: no cover
    faiss = None  # type: ignore
    lazy_pinyin = None  # type: ignore
    Levenshtein = None  # type: ignore
    _HAS_DEPS = False
    _IMPORT_ERR = str(_e)


def _to_pinyin_syllables(text: str) -> list[str]:
    """文本 → 无声调拼音音节列表。"""
    return [s for s in lazy_pinyin(text) if s]


class FaissCorrector:
    """大词库拼音检索纠错器。原子切换索引，支持随词表热更新重建。"""

    def __init__(self):
        # 单一状态快照引用（None=未就绪）；build 原子替换，correct 原子读取。
        self._state: _FaissState | None = None
        self._build_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        st = self._state
        return _HAS_DEPS and st is not None and len(st.words) > 0

    @property
    def version(self) -> int:
        st = self._state
        return st.version if st is not None else -1

    # --------------------------------------------------------
    # 索引构建（原子切换）
    # --------------------------------------------------------
    def build(self, words: list[str], version: int):
        """从词表构建拼音向量索引。失败不抛出，仅记录日志并保持禁用。"""
        if not _HAS_DEPS:
            logger.warning(f"Faiss 纠错依赖缺失（{_IMPORT_ERR}），路径 B 不可用")
            return

        words = [w for w in dict.fromkeys(w.strip() for w in words) if w]
        if not words:
            logger.warning("Faiss 纠错词表为空，跳过构建")
            return

        with self._build_lock:
            try:
                # 1) 构建音节词表
                word_syllables = [_to_pinyin_syllables(w) for w in words]
                syllable_to_idx: dict[str, int] = {}
                for syls in word_syllables:
                    for s in syls:
                        if s not in syllable_to_idx:
                            syllable_to_idx[s] = len(syllable_to_idx)
                dim = len(syllable_to_idx)
                if dim == 0:
                    logger.warning("Faiss 纠错：拼音音节词表为空，跳过构建")
                    return

                # 2) 词向量矩阵（multi-hot 计数 + L2 归一化）
                mat = np.zeros((len(words), dim), dtype=np.float32)
                for i, syls in enumerate(word_syllables):
                    for s in syls:
                        mat[i, syllable_to_idx[s]] += 1.0
                faiss.normalize_L2(mat)

                # 3) 构建 IndexFlatIP（内积 = 余弦，因已归一化）
                index = faiss.IndexFlatIP(dim)
                index.add(mat)

                word_pinyin = ["".join(syls) for syls in word_syllables]

                # 原子切换：打包成不可变快照，单次引用赋值（GIL 原子），
                # correct 读一次 self._state 即得一致视图，杜绝读到 index 新/词表旧的中间态
                self._state = _FaissState(
                    index=index,
                    words=tuple(words),
                    word_pinyin=tuple(word_pinyin),
                    syllable_to_idx=syllable_to_idx,
                    dim=dim,
                    version=version,
                )

                logger.info(
                    f"Faiss 纠错索引构建完成: version={version}, 词条={len(words)}, dim={dim}"
                )
            except Exception as e:
                logger.warning(f"Faiss 纠错索引构建失败: {e}")

    # --------------------------------------------------------
    # 文本纠错
    # --------------------------------------------------------
    def _vectorize(self, text: str, st: _FaissState) -> np.ndarray | None:
        """候选片段 → 拼音向量（在快照 st 的音节词表上）。"""
        syls = _to_pinyin_syllables(text)
        if not syls:
            return None
        vec = np.zeros((1, st.dim), dtype=np.float32)
        hit = False
        for s in syls:
            idx = st.syllable_to_idx.get(s)
            if idx is not None:
                vec[0, idx] += 1.0
                hit = True
        if not hit:
            return None
        faiss.normalize_L2(vec)
        return vec

    def _gen_candidates(self, text: str) -> list[tuple[int, int]]:
        """滑窗切片生成候选 (start, end) 索引区间。"""
        spans: list[tuple[int, int]] = []
        n = len(text)
        for w in settings.FAISS_WINDOW_SIZES:
            if w > n:
                continue
            for i in range(0, n - w + 1):
                spans.append((i, i + w))
        return spans

    def correct(self, text: str) -> str:
        """对 ASR 文本做拼音检索纠错，返回纠错后文本（无命中则原样返回）。"""
        corrected, _ = self.correct_with_spans(text)
        return corrected

    def correct_with_spans(self, text: str) -> tuple[str, list[tuple[int, int, str]]]:
        """
        对 ASR 文本做拼音检索纠错，返回 (纠错后文本, 替换区间列表)。

        替换区间列表元素为 (start, end, cand)：text[start:end] 被替换为 cand，
        按 start 升序、区间互不重叠。供上层把同一替换映射到字级时间戳 words，
        保证 words 与段 text 纠错后一致（I3）。无命中返回 (原文, [])。

        逐候选片段检索，三重联合判定通过则记录替换。
        同一位置只替换一次（最长片段优先），最后按非重叠区间重组文本。
        """
        st = self._state  # 一次性原子读取快照，后续全程用 st（避免中途被 build 替换）
        if not _HAS_DEPS or st is None or not st.words or not text:
            return text, []

        try:
            spans = self._gen_candidates(text)
            if not spans:
                return text, []

            # 长片段优先（更具体，减少误纠）
            spans.sort(key=lambda s: s[1] - s[0], reverse=True)

            occupied = [False] * len(text)
            # 已接受的替换：start → (end, cand)
            replacements: dict[int, tuple[int, str]] = {}
            topk = settings.FAISS_TOPK

            for (start, end) in spans:
                # 区间与已接受替换重叠则跳过
                if any(occupied[start:end]):
                    continue
                frag = text[start:end]
                vec = self._vectorize(frag, st)
                if vec is None:
                    continue

                k = min(topk, len(st.words))
                scores, idxs = st.index.search(vec, k)
                scores, idxs = scores[0], idxs[0]
                if len(idxs) == 0 or idxs[0] < 0:
                    continue

                top1_score = float(scores[0])
                top2_score = float(scores[1]) if len(scores) > 1 else 0.0
                cand = st.words[idxs[0]]

                # 编辑距离分（基于拼音串）
                frag_py = "".join(_to_pinyin_syllables(frag))
                cand_py = st.word_pinyin[idxs[0]]
                maxlen = max(len(frag_py), len(cand_py), 1)
                edit_score = 1.0 - Levenshtein.distance(frag_py, cand_py) / maxlen

                final_score = (
                    settings.FAISS_PINYIN_WEIGHT * top1_score
                    + settings.FAISS_EDIT_WEIGHT * edit_score
                )

                # 三重联合判定
                if (
                    top1_score > settings.FAISS_SCORE_THRESHOLD
                    and (top1_score - top2_score) > settings.GAP_THRESHOLD
                    and final_score > settings.FINAL_SCORE_THRESHOLD
                    and cand != frag
                ):
                    for j in range(start, end):
                        occupied[j] = True
                    replacements[start] = (end, cand)
                    logger.info(
                        f"Faiss 纠错: '{frag}' → '{cand}' "
                        f"(py={top1_score:.3f}, edit={edit_score:.3f}, final={final_score:.3f})"
                    )

            if not replacements:
                return text, []

            # 按非重叠区间重组文本 + 收集替换区间（按 start 升序）
            out_parts: list[str] = []
            span_list: list[tuple[int, int, str]] = []
            i = 0
            while i < len(text):
                if i in replacements:
                    end, cand = replacements[i]
                    out_parts.append(cand)
                    span_list.append((i, end, cand))
                    i = end
                else:
                    out_parts.append(text[i])
                    i += 1
            return "".join(out_parts), span_list
        except Exception as e:
            logger.warning(f"Faiss 纠错执行失败，返回原文: {e}")
            return text, []


# 全局单例
faiss_corrector = FaissCorrector()
