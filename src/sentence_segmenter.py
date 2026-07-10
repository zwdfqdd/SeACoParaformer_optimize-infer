# -*- coding: utf-8 -*-
"""
句子级分句器（CT-Transformer 标点恢复 + 分句）

基于 CT-Transformer 标点模型（iic/punc_ct-transformer_zh-cn-common-vad_realtime-
vocab272727-onnx，纯 onnxruntime 手写推理）对无标点 ASR 文本恢复标点，再按标点切分
子句，供 main._build_response 结合字级时间戳构造句子级时间戳（asr[] 粒度变为子句）。

选型（替换早期 ngram-punctuator）：
    CT-Transformer 逐 token 分类，对长文本/重复口语无 n-gram 的困惑度阈值失效问题，
    实测 20000+ 字/秒（远快于 ngram），标点完整自然，且纯 onnxruntime（项目已有，无新依赖）。

模型（扁平存于 settings.PUNC_MODEL_DIR，默认 models/punc）：
    model_quant.onnx（量化，281MB）/ tokens.json（272727，CharTokenizer 逐字）/ config.yaml
    缺失时启动/加载阶段自动调 scripts/download_punc.py（HTTP 直链）下载。

ONNX 契约（已探测）：
    输入 inputs(int32)[1,L] / text_lengths(int32)[1] / vad_masks(f32)[1,1,L,L] /
        sub_masks(f32)[1,1,L,L]
    输出 logits[1,L,6]，punc_list=[<unk>,_,，,。,？,、]（0/1 无标点，2..5 标点）

设计（与 Faiss 纠错一致的懒加载）：依赖/模型缺失自动禁用，不阻塞主流程。
线程安全：推理为无状态 ONNX run（ORT 内部线程安全），可多线程并发调用。
"""

from __future__ import annotations

import json
import os
import threading
from typing import List, Tuple

import numpy as np

from src.config import settings
from src.logger import logger

# logits 6 类标点映射（config.yaml punc_list）
_PUNC_LIST = ["<unk>", "_", "，", "。", "？", "、"]
_NO_PUNC_IDS = {0, 1}  # <unk> / _ 视为无标点


class SentenceSegmenter:
    """CT-Transformer 标点恢复 + 分句器（懒加载纯 ONNX，缺失自动禁用）。"""

    def __init__(self):
        self._sess = None
        self._tok2id: dict[str, int] = {}
        self._unk_id: int = 0
        self._in_names: list[str] = []
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._sess is not None

    def load(self):
        """加载标点模型（幂等）。失败仅告警并禁用，不抛异常阻塞服务。"""
        if self._loaded or self._load_failed:
            return
        with self._lock:
            if self._loaded or self._load_failed:
                return
            if not settings.ENABLE_SENTENCE_TIMESTAMP:
                return
            punc_dir = os.path.abspath(settings.PUNC_MODEL_DIR)
            os.makedirs(punc_dir, exist_ok=True)

            # 模型文件缺失则自动下载（HTTP 直链，扁平结构）
            if not self._files_ready(punc_dir):
                logger.warning(
                    f"标点模型未在 {punc_dir} 就绪（缺 {settings.PUNC_ONNX_NAME}/tokens.json），"
                    f"自动启动下载脚本 scripts/download_punc.py ..."
                )
                self._auto_download(punc_dir)
            else:
                logger.info(f"标点模型已存在，跳过下载: {punc_dir}")

            try:
                import onnxruntime as ort

                onnx_path = os.path.join(punc_dir, settings.PUNC_ONNX_NAME)
                tokens_path = os.path.join(punc_dir, "tokens.json")
                with open(tokens_path, encoding="utf-8") as f:
                    toks = json.load(f)
                self._tok2id = {t: i for i, t in enumerate(toks)}
                self._unk_id = self._tok2id.get("<unk>", len(toks) - 1)

                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # 纯 CPU 标点推理，线程数跟随 ORT 默认（受 OMP 环境约束）
                self._sess = ort.InferenceSession(
                    onnx_path, so, providers=["CPUExecutionProvider"]
                )
                self._in_names = [i.name for i in self._sess.get_inputs()]
                self._loaded = True
                logger.info(
                    f"句子分句器加载成功（CT-Transformer）: onnx={settings.PUNC_ONNX_NAME}, "
                    f"vocab={len(toks)}, max_len={settings.PUNC_MAX_LEN}, dir={punc_dir}"
                )
            except Exception as e:
                logger.warning(f"句子分句器加载失败（句子级时间戳禁用）: {e}")
                self._load_failed = True

    @staticmethod
    def _files_ready(punc_dir: str) -> bool:
        """判断扁平模型文件是否齐全（onnx + tokens.json）。"""
        onnx_ok = os.path.exists(os.path.join(punc_dir, settings.PUNC_ONNX_NAME))
        tok_ok = os.path.exists(os.path.join(punc_dir, "tokens.json"))
        return onnx_ok and tok_ok

    def _auto_download(self, punc_dir: str):
        """调用 download_punc.py 下载标点模型到 punc_dir（失败仅告警）。"""
        import subprocess
        import sys
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent / "scripts" / "download_punc.py"
        try:
            subprocess.run(
                [sys.executable, str(script), "--output-dir", punc_dir],
                check=False,
            )
        except Exception as e:
            logger.warning(f"标点模型自动下载失败: {e}")

    # --------------------------------------------------------
    # 推理
    # --------------------------------------------------------
    def _tokenize(self, text: str) -> Tuple[List[str], List[int]]:
        """逐字 → (字符列表, token id 列表)。空白跳过（不计入原文对齐）。"""
        chars, ids = [], []
        for ch in text:
            if ch.isspace():
                continue
            chars.append(ch)
            ids.append(self._tok2id.get(ch, self._unk_id))
        return chars, ids

    def _run_window(self, token_ids: List[int]) -> np.ndarray:
        """单窗推理，返回每 token 的标点类别 argmax (L,)。"""
        L = len(token_ids)
        feed = {}
        for name in self._in_names:
            if name == "inputs":
                feed[name] = np.array([token_ids], dtype=np.int32)
            elif name == "text_lengths":
                feed[name] = np.array([L], dtype=np.int32)
            elif name == "vad_masks":
                feed[name] = np.ones((1, 1, L, L), dtype=np.float32)
            elif name == "sub_masks":
                feed[name] = np.ones((1, 1, L, L), dtype=np.float32)
        logits = self._sess.run(None, feed)[0]  # [1, L, 6]
        return np.argmax(logits[0], axis=-1)

    def _predict_classes(self, chars: List[str], ids: List[int]) -> List[int]:
        """对全部字符逐窗推理，返回与 chars 对齐的标点类别列表。"""
        max_len = settings.PUNC_MAX_LEN
        classes: List[int] = []
        i = 0
        n = len(ids)
        while i < n:
            seg = ids[i:i + max_len] if max_len > 0 else ids[i:]
            cls = self._run_window(seg)
            classes.extend(int(c) for c in cls)
            if max_len <= 0:
                break
            i += max_len
        return classes[:len(chars)]

    def punctuate(self, text: str) -> str:
        """对无标点文本恢复标点。未就绪或异常时返回原文（保守降级）。"""
        if not self.is_ready or not text:
            return text
        try:
            chars, ids = self._tokenize(text)
            if not ids:
                return text
            classes = self._predict_classes(chars, ids)
            out = []
            for ch, c in zip(chars, classes):
                out.append(ch)
                if c not in _NO_PUNC_IDS:
                    out.append(_PUNC_LIST[c])
            return "".join(out)
        except Exception as e:
            logger.warning(f"标点恢复失败，返回原文: {e}")
            return text

    def split_sentences(self, text: str) -> List[Tuple[str, List[int]]]:
        """
        对无标点文本恢复标点并按标点切分子句（任何标点都切，子句级）。

        返回 [(带标点子句, [原文起始字符偏移, 原文结束字符偏移)], ...]：
            子句 = 含尾部标点的片段（用于展示 / istar_asr 拼接）；
            [start, end) = 该子句对应「原文 text」的字符区间，供上层映射字级时间戳。

        实现：逐字推理标点类别；每遇到标点即在该字后收一个子句（含标点），
        原文字符偏移只按非标点字符（即原文字符）推进——标点是模型插入的，不占原文偏移。
        原文对齐用 _tokenize 的逐字序列（与字级 words 同为逐字，天然对齐）。

        未就绪时返回单句 [(原文, [0, len(有效字符)])]（降级为整段一句）。
        """
        if not text:
            return []
        # 逐字（去空白）作为对齐基准——与字级 words 口径一致（words 也是逐字/subword）
        chars, ids = self._tokenize(text)
        if not chars:
            return []
        n = len(chars)
        if not self.is_ready:
            return [("".join(chars), [0, n])]

        try:
            classes = self._predict_classes(chars, ids)
        except Exception as e:
            logger.warning(f"分句推理失败，降级整段一句: {e}")
            return [("".join(chars), [0, n])]

        sentences: List[Tuple[str, List[int]]] = []
        cur: List[str] = []
        seg_start = 0
        for i, ch in enumerate(chars):
            cur.append(ch)
            c = classes[i] if i < len(classes) else 0
            if c not in _NO_PUNC_IDS:
                cur.append(_PUNC_LIST[c])
                sentences.append(("".join(cur), [seg_start, i + 1]))
                cur = []
                seg_start = i + 1
        # 尾部残句（无标点结尾）
        if cur:
            sentences.append(("".join(cur), [seg_start, n]))

        if not sentences:
            return [("".join(chars), [0, n])]
        return sentences


# 全局单例
sentence_segmenter = SentenceSegmenter()
