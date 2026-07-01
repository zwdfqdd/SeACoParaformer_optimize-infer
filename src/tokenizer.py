"""
Tokenizer 模块（自实现，无第三方 ASR 框架依赖）

SeACo-Paraformer 使用 vocab8404 词表，包含：
- 8404 个 token（中文字、英文 subword、标点等）
- 特殊 token：<blank>=0, <sos>=1, <eos>=2

词表文件格式（tokens.json 或 tokens.txt）：
- JSON: ["<blank>", "<sos>", "<eos>", "的", "一", ...]
- TXT: 每行一个 token，行号即 ID

仅依赖 numpy + json。
"""

import json
from pathlib import Path

import numpy as np


# 特殊 token ID
BLANK_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = -1

# 需要过滤的特殊 token
SPECIAL_TOKEN_IDS = {BLANK_ID, SOS_ID, EOS_ID}


class Tokenizer:
    """
    Token ID ↔ 文本 转换器。

    加载 vocab8404 词表文件，提供 decode 功能。
    """

    def __init__(self):
        self._token_list: list[str] = []
        self._token_to_id: dict[str, int] = {}
        self._seg_dict: dict[str, list[str]] = {}  # 英文单词 → BPE subword 序列（来自 seg_dict）
        self._loaded = False

    def load(self, vocab_path: str, seg_dict_path: str | None = None):
        """
        加载词表文件。

        支持格式：
        - tokens.json: JSON 数组 ["<blank>", "<sos>", ...]
        - tokens.txt: 每行格式 "token id" 或仅 "token"（行号为 ID）

        seg_dict_path：可选，FunASR seg_dict 文件（英文单词→BPE subword 映射）。
            传 None 时自动尝试在 vocab 同目录查找名为 "seg_dict" 的文件。
            用于英文热词的正确 BPE 切分（见 encode 说明）。
        """
        path = Path(vocab_path)

        if not path.exists():
            raise FileNotFoundError(f"词表文件不存在: {vocab_path}")

        if path.suffix == ".json":
            self._load_json(path)
        elif path.suffix == ".txt":
            self._load_txt(path)
        else:
            # 尝试 JSON 格式
            try:
                self._load_json(path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._load_txt(path)

        # 加载 seg_dict（英文 BPE 切分表）：显式路径优先，否则同目录自动探测
        if seg_dict_path is None:
            auto = path.parent / "seg_dict"
            seg_dict_path = str(auto) if auto.exists() else None
        if seg_dict_path:
            self._load_seg_dict(Path(seg_dict_path))

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def vocab_size(self) -> int:
        return len(self._token_list)

    def decode(self, token_ids: np.ndarray | list[int]) -> str:
        """
        将 token ID 序列解码为文本。

        参数:
            token_ids: token ID 数组

        返回:
            解码后的文本字符串
        """
        if not self._loaded:
            return "[tokenizer 未加载]"

        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.flatten().tolist()

        tokens: list[str] = []
        for tid in token_ids:
            tid = int(tid)

            # 跳过特殊 token
            if tid in SPECIAL_TOKEN_IDS:
                continue

            # 跳过无效 ID
            if tid < 0 or tid >= len(self._token_list):
                continue

            token = self._token_list[tid]

            # 跳过特殊标记
            if token.startswith("<") and token.endswith(">"):
                continue

            tokens.append(token)

        # 拼接 token 为文本
        text = self._join_tokens(tokens)
        return text

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token ID 序列（用于 hotwords 编码）。

        分两类处理：
        - 英文单词（连续 ASCII 字母/数字/撇号）：查 seg_dict 得到正确的 BPE subword
          序列（如 android → a@@ nd@@ ro@@ id），再逐 subword 映射 ID。
          seg_dict 缺失或未命中时 fallback 到贪心最长匹配。
        - 中文及其他字符：逐字最长匹配（≤4 字符），与原行为一致。

        说明：seg_dict 是 FunASR 官方英文 BPE 切分表，集成后英文热词偏置强度正常；
        中文热词逐字命中、encode/decode 字符级一致，路径不变。
        """
        if not self._loaded:
            return []

        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            # 英文单词：连续 ASCII 字母/数字/撇号
            if ch.isascii() and (ch.isalnum() or ch == "'"):
                j = i
                while j < n and text[j].isascii() and (text[j].isalnum() or text[j] == "'"):
                    j += 1
                word = text[i:j]
                ids.extend(self._encode_english_word(word))
                i = j
            else:
                # 中文/标点/其他：逐字最长匹配（≤4 字符）
                matched = False
                for length in range(min(4, n - i), 0, -1):
                    substr = text[i: i + length]
                    if substr in self._token_to_id:
                        ids.append(self._token_to_id[substr])
                        i += length
                        matched = True
                        break
                if not matched:
                    i += 1

        return ids

    def _encode_english_word(self, word: str) -> list[int]:
        """编码单个英文单词为 subword ID 序列。

        优先查 seg_dict（小写键）得到 BPE subword 序列；未命中则 fallback
        贪心最长匹配。subword 在词表中查不到的跳过。
        """
        subwords = self._seg_dict.get(word.lower())
        if subwords:
            ids = [self._token_to_id[s] for s in subwords if s in self._token_to_id]
            if ids:
                return ids
        # fallback：贪心最长匹配（原行为）
        return self._greedy_encode(word)

    def _greedy_encode(self, text: str) -> list[int]:
        """贪心最长匹配（≤4 字符），用于 seg_dict 未命中的 fallback。"""
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(4, len(text) - i), 0, -1):
                substr = text[i: i + length]
                if substr in self._token_to_id:
                    ids.append(self._token_to_id[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                i += 1
        return ids

    def _load_json(self, path: Path):
        """加载 JSON 格式词表。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self._token_list = data
        elif isinstance(data, dict):
            # {"token": id} 格式
            max_id = max(data.values())
            self._token_list = [""] * (max_id + 1)
            for token, tid in data.items():
                self._token_list[tid] = token
        else:
            raise ValueError("不支持的 JSON 词表格式")

        self._build_token_to_id()

    def _load_txt(self, path: Path):
        """加载 TXT 格式词表。"""
        self._token_list = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    # "token id" 格式
                    token = parts[0]
                    tid = int(parts[1])
                    # 确保列表足够长
                    while len(self._token_list) <= tid:
                        self._token_list.append("")
                    self._token_list[tid] = token
                else:
                    # 仅 token，行号为 ID
                    self._token_list.append(parts[0])

        self._build_token_to_id()

    def _build_token_to_id(self):
        """构建 token → id 映射。"""
        self._token_to_id = {
            token: idx
            for idx, token in enumerate(self._token_list)
            if token
        }

    def _load_seg_dict(self, path: Path):
        """加载 FunASR seg_dict（英文单词 → BPE subword 序列）。

        格式：每行 "单词<TAB>subword1 subword2 ..."，subword 用 @@ 表示词内连接。
        例：aachen\\ta@@ ach@@ en
        解析失败时静默降级（self._seg_dict 保持空，encode 走 fallback）。
        """
        if not path.exists():
            return
        try:
            seg: dict[str, list[str]] = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    # 第一个 \t 或连续空白前是 key，其余是 subword 序列
                    parts = line.split("\t") if "\t" in line else line.split(None, 1)
                    if len(parts) < 2:
                        continue
                    word = parts[0].strip()
                    subwords = parts[1].split()
                    if word and subwords:
                        seg[word.lower()] = subwords
            self._seg_dict = seg
        except (UnicodeDecodeError, OSError):
            self._seg_dict = {}

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        """
        拼接 token 为文本。

        本词表（vocab8404）BPE 约定：
        - 中文字符：独立 token，直接拼接
        - 英文 subword：用 `@@` 后缀表示"与下一个 token 连接"（如 and@@ + roid → android）
        - 兼容 sentencepiece `▁` 前缀（若存在则替换为空格）

        拼接规则：
        - token 以 `@@` 结尾 → 去掉 `@@`，与下一 token 直接相连（无分隔）
        - 否则该 token 是词尾，英文词之间补空格、中文之间不补
        """
        parts: list[str] = []
        for tok in tokens:
            if tok.endswith("@@"):
                # BPE 连接片段：去后缀，标记不加分隔
                parts.append((tok[:-2], False))
            else:
                parts.append((tok, True))

        out = ""
        for i, (frag, word_end) in enumerate(parts):
            frag = frag.replace("▁", " ")
            out += frag
            # 词尾且后面还有内容时，若两侧都是 ASCII 字母则补空格（英文分词）
            if word_end and i < len(parts) - 1:
                nxt = parts[i + 1][0]
                if frag[-1:].isascii() and frag[-1:].isalnum() and nxt[:1].isascii() and nxt[:1].isalnum():
                    out += " "

        # 归一化空格
        out = " ".join(out.split())
        return out


# 全局单例
tokenizer = Tokenizer()
