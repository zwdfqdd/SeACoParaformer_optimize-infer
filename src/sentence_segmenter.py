# -*- coding: utf-8 -*-
"""
句子级分句器（标点恢复 + 分句）

基于 ngram-punctuator（KenLM n-gram 统计语言模型 + Qwen2.5 BPE 分词，纯 CPU）对
无标点的 ASR 文本恢复标点，再按句末标点切分句子，供 main._build_response 结合字级
时间戳构造句子级时间戳（asr[] 粒度变为句）。

设计（与 Faiss 纠错一致的懒加载策略）：
    - 依赖（ngram_punctuator/kenlm/modelscope）缺失时自动禁用，不阻塞主流程；
    - 模型为扁平结构：settings.PUNC_MODEL_DIR 下直接放 prune{prune}.bin / vocab.json /
      merges.txt（无 ModelScope 嵌套缓存目录）。加载时 monkeypatch 上游
      model_file_download 返回本地扁平路径，从本地加载、不联网；缺失才自动调
      scripts/download_punc.py（HTTP 直链下载并平铺）。
    - 生产参数固化实测最优：order=3 + 中文候选标点 + ppl_drop_ratio=0.12
      （详见 scripts/benchmark_punctuator.py 实测结论）。

线程安全：punctuate 为纯 CPU 无状态计算（KenLM score），可多线程并发调用。
"""

from __future__ import annotations

import os
import threading
from typing import List, Tuple

from src.config import settings
from src.logger import logger


class SentenceSegmenter:
    """标点恢复 + 分句器（懒加载 ngram-punctuator，缺失自动禁用）。"""

    def __init__(self):
        self._punctuator = None
        self._punct_fn = None           # 兼容 punctuate / predict 两种方法名
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._punct_fn is not None

    def load(self):
        """加载标点模型（幂等）。失败仅告警并禁用，不抛异常阻塞服务。"""
        if self._loaded or self._load_failed:
            return
        with self._lock:
            if self._loaded or self._load_failed:
                return
            if not settings.ENABLE_SENTENCE_TIMESTAMP:
                return
            # 模型目录（扁平结构）：models/punc/ 下直接放 prune{prune}.bin / vocab.json /
            # merges.txt，不用 ModelScope 嵌套缓存目录（models/{org}--{repo}/...）。
            punc_dir = os.path.abspath(settings.PUNC_MODEL_DIR)
            os.makedirs(punc_dir, exist_ok=True)
            try:
                from ngram_punctuator import Punctuator
            except ImportError as e:
                logger.warning(
                    f"句子级时间戳依赖 ngram-punctuator 缺失（{e}），句子分句禁用；"
                    f"如需启用请 pip install ngram-punctuator"
                )
                self._load_failed = True
                return
            # 扁平模型文件齐全则本地加载；缺失才告警并自动下载（download_punc.py 会平铺）
            if not self._flat_files_ready(punc_dir):
                logger.warning(
                    f"标点模型未在 {punc_dir} 就绪（缺 prune/vocab/merges），"
                    f"自动启动下载脚本 scripts/download_punc.py ..."
                )
                self._auto_download(punc_dir)
            else:
                logger.info(f"标点模型已存在，跳过下载: {punc_dir}")
            try:
                self._punctuator = self._build_punctuator_local(punc_dir)
                self._punct_fn = self._resolve_fn(self._punctuator)
                self._loaded = True
                logger.info(
                    f"句子分句器加载成功: order={settings.PUNC_NGRAM_ORDER}, "
                    f"candidates={settings.PUNC_CANDIDATES}, "
                    f"ppl_drop={settings.PUNC_PPL_DROP_RATIO}, dir={punc_dir}"
                )
            except Exception as e:
                logger.warning(f"句子分句器加载失败（句子级时间戳禁用）: {e}")
                self._load_failed = True

    @staticmethod
    def _resolve_fn(punctuator):
        """兼容不同版本方法名：pip 发布版 predict，master 源码 punctuate。"""
        for name in ("punctuate", "predict"):
            fn = getattr(punctuator, name, None)
            if callable(fn):
                return fn
        raise AttributeError("Punctuator 无 punctuate/predict 方法，版本不兼容")

    @staticmethod
    def _prune_name(order: int) -> str:
        """order 对应的 n-gram 模型文件名（与上游一致）：prune{0..order-1}.bin。"""
        return f"prune{''.join(map(str, range(order)))}.bin"

    def _flat_files_ready(self, punc_dir: str) -> bool:
        """判断扁平结构模型文件是否齐全（prune*.bin + vocab.json + merges.txt）。"""
        prune = os.path.join(punc_dir, self._prune_name(settings.PUNC_NGRAM_ORDER))
        vocab = os.path.join(punc_dir, "vocab.json")
        merges = os.path.join(punc_dir, "merges.txt")
        return os.path.exists(prune) and os.path.exists(vocab) and os.path.exists(merges)

    def _build_punctuator_local(self, punc_dir: str):
        """从扁平模型目录构造 Punctuator，绕过 ModelScope 下载与嵌套缓存目录。

        上游 Punctuator/Tokenizer 通过 modelscope.model_file_download 取文件路径。
        这里临时 monkeypatch 该函数，按请求的文件名返回本地扁平路径（prune*.bin /
        vocab.json / merges.txt），使其从本地加载、不联网、不产生 {org}--{repo} 目录。
        """
        from ngram_punctuator import Punctuator
        import ngram_punctuator.punctuator as _pmod
        import ngram_punctuator.tokenizer as _tmod

        prune_path = os.path.join(punc_dir, self._prune_name(settings.PUNC_NGRAM_ORDER))
        local_map = {
            "vocab.json": os.path.join(punc_dir, "vocab.json"),
            "merges.txt": os.path.join(punc_dir, "merges.txt"),
        }

        def _local_download(repo_id: str, file_path: str, *args, **kwargs):
            base = os.path.basename(file_path)
            if base.endswith(".bin"):
                return prune_path
            if base in local_map:
                return local_map[base]
            # 未预置的文件（理论不会走到）→ 回退原始下载
            from modelscope import model_file_download as _orig
            return _orig(repo_id, file_path, *args, **kwargs)

        orig_p = getattr(_pmod, "model_file_download", None)
        orig_t = getattr(_tmod, "model_file_download", None)
        try:
            if orig_p is not None:
                _pmod.model_file_download = _local_download
            if orig_t is not None:
                _tmod.model_file_download = _local_download
            return Punctuator(
                order=settings.PUNC_NGRAM_ORDER,
                tokenizer_id=settings.PUNC_TOKENIZER_ID,
            )
        finally:
            if orig_p is not None:
                _pmod.model_file_download = orig_p
            if orig_t is not None:
                _tmod.model_file_download = orig_t

    def _auto_download(self, punc_dir: str):
        """调用 download_punc.py 下载标点模型到 punc_dir（失败仅告警，交由后续加载兜底）。"""
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent / "scripts" / "download_punc.py"
        try:
            subprocess.run(
                [sys.executable, str(script),
                 "--output-dir", punc_dir,
                 "--order", str(settings.PUNC_NGRAM_ORDER),
                 "--tokenizer-id", settings.PUNC_TOKENIZER_ID],
                check=False,
            )
        except Exception as e:
            logger.warning(f"标点模型自动下载失败: {e}（Punctuator 加载时将回退在线下载）")

    def punctuate(self, text: str) -> str:
        """对无标点文本恢复标点。未就绪或异常时返回原文（保守降级）。"""
        if not self.is_ready or not text:
            return text
        try:
            return self._punct_fn(
                text,
                puncts=settings.PUNC_CANDIDATES,
                ppl_drop_ratio=settings.PUNC_PPL_DROP_RATIO,
            )
        except Exception as e:
            logger.warning(f"标点恢复失败，返回原文: {e}")
            return text

    def split_sentences(self, text: str) -> List[Tuple[str, List[int]]]:
        """
        对无标点文本恢复标点并按句末标点切句。

        返回 [(带标点句子, [原文起始字符偏移, 原文结束字符偏移)], ...]：
            句子 = 含句末标点的完整句子文本（用于展示 / istar_asr 拼接）；
            [start, end) = 该句对应「原文 text」的字符区间，供上层映射字级时间戳。

        对齐方式（鲁棒双指针）：
            标点模型输出 = 原文字符 + 插入的标点 + format_text 插入的空格（中英文之间）。
            用双指针把带标点结果逐字与原文匹配：字符相等则推进原文游标（该字符属于原文），
            否则视为「插入字符」（标点或空格，不推进原文游标）。遇到 SENTENCE_SPLIT_PUNCTS
            即断句，得到每句对应的原文字符区间（与字级 words 严格对齐）。
            双指针错位（异常）时兜底整段一句，避免破坏输出。

        未就绪时返回单句 [(原文, [0, len(text)])]（降级为整段一句）。
        """
        if not text:
            return []
        if not self.is_ready:
            return [(text, [0, len(text)])]

        punctuated = self.punctuate(text)
        split_set = set(settings.SENTENCE_SPLIT_PUNCTS)

        sentences: List[Tuple[str, List[int]]] = []
        cur_chars: List[str] = []      # 当前句累积（含标点，去空格前）
        seg_start = 0                  # 当前句在原文中的起始偏移
        orig_pos = 0                   # 原文字符游标（双指针匹配推进）
        n = len(text)

        for ch in punctuated:
            cur_chars.append(ch)
            # 双指针：与原文当前字符相等 → 属于原文，推进游标
            if orig_pos < n and ch == text[orig_pos]:
                orig_pos += 1
            # 否则为插入字符（标点/空格），原文游标不动
            if ch in split_set:
                sentence = "".join(cur_chars).strip()
                if sentence:
                    sentences.append((sentence, [seg_start, orig_pos]))
                cur_chars = []
                seg_start = orig_pos

        # 尾部残句（无句末标点结尾）
        if cur_chars:
            sentence = "".join(cur_chars).strip()
            if sentence:
                sentences.append((sentence, [seg_start, orig_pos]))

        # 兜底：全程未断出句子 / 双指针未走完原文（错位）→ 整段一句
        if not sentences or orig_pos < n:
            if orig_pos < n:
                logger.warning("句子分句双指针未对齐原文，降级整段一句")
            return [(punctuated, [0, len(text)])]
        return sentences


# 全局单例
sentence_segmenter = SentenceSegmenter()
