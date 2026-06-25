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
        self._loaded = False

    def load(self, vocab_path: str):
        """
        加载词表文件。

        支持格式：
        - tokens.json: JSON 数组 ["<blank>", "<sos>", ...]
        - tokens.txt: 每行格式 "token id" 或仅 "token"（行号为 ID）
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

        简单的贪心最长匹配（最多 4 字符），优先匹配长 token。

        说明：本词表为中文字 + 英文 BPE（@@ 后缀）混合。中文热词逐字命中、
        encode/decode 字符级一致；英文热词因无 BPE merges 规则，贪心匹配可能
        切成非连接片段（如 android → and/r/o/id），仅影响英文热词的偏置强度，
        不影响中文热词与正常识别。本项目热词以中文为主，该限制可接受。
        """
        if not self._loaded:
            return []

        ids: list[int] = []
        i = 0
        while i < len(text):
            # 尝试最长匹配（最多 4 个字符）
            matched = False
            for length in range(min(4, len(text) - i), 0, -1):
                substr = text[i: i + length]
                if substr in self._token_to_id:
                    ids.append(self._token_to_id[substr])
                    i += length
                    matched = True
                    break

            if not matched:
                # 未匹配，跳过该字符
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
