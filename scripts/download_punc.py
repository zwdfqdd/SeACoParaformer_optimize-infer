# -*- coding: utf-8 -*-
"""
ngram 标点模型下载脚本（句子级时间戳用）

直接从 ModelScope HTTP 下载单文件到本地目录（扁平结构，无需 modelscope 库，
不产生 {org}--{repo} 嵌套缓存目录），供 src/sentence_segmenter.py 离线加载。

模型来源（ModelScope resolve 直链）:
    - n-gram 模型: pengzhendong/ngram-punctuator（{order}gram_trie_a22_q8_b8/prune*.bin）
    - BPE 词表:    Qwen/Qwen2.5-7B-Instruct（vocab.json / merges.txt）

最终 models/punc/ 下只保留 3 个扁平文件：
    prune{0..order-1}.bin / vocab.json / merges.txt

用法:
    python scripts/download_punc.py                      # 默认 order=3 到 models/punc
    python scripts/download_punc.py --order 4
    python scripts/download_punc.py --output-dir ./models/punc --tokenizer-id Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

# ModelScope 单文件直链模板（与 download_vad.py 一致的 resolve/master 形式）
MS_RESOLVE = "https://modelscope.cn/models/{repo}/resolve/master/{path}"
NGRAM_REPO = "pengzhendong/ngram-punctuator"


def _make_progress():
    """构造 urlretrieve 的 reporthook：单行刷新 百分比 + 已下载/总大小 + 速度。"""
    state = {"t0": time.time()}

    def _hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        elapsed = max(time.time() - state["t0"], 1e-6)
        speed = downloaded / elapsed / (1024 * 1024)  # MB/s
        dl_mb = downloaded / (1024 * 1024)
        if total_size > 0:
            total_mb = total_size / (1024 * 1024)
            pct = min(downloaded * 100 / total_size, 100.0)
            sys.stdout.write(
                f"\r       {pct:5.1f}%  {dl_mb:7.1f}/{total_mb:.1f} MB  {speed:5.1f} MB/s"
            )
        else:
            # 总大小未知（服务器未返回 Content-Length）
            sys.stdout.write(f"\r       {dl_mb:7.1f} MB  {speed:5.1f} MB/s")
        sys.stdout.flush()

    return _hook


def _download(url: str, dst: Path):
    """HTTP 下载单文件到 dst（覆盖），带进度显示。"""
    print(f"  下载: {url}")
    urllib.request.urlretrieve(url, str(dst), reporthook=_make_progress())
    sys.stdout.write("\n")  # 进度行收尾换行
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"       → {dst} ({size_mb:.1f} MB)")


def download_punc(output_dir: Path, order: int, tokenizer_id: str):
    """下载 n-gram 模型 + BPE 词表到 output_dir（扁平结构）。已就绪则跳过。"""
    assert order in (3, 4, 5, 6), f"order 必须为 3/4/5/6，当前 {order}"

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prune = "".join(map(str, range(order)))
    prune_name = f"prune{prune}.bin"
    dst_prune = output_dir / prune_name
    dst_vocab = output_dir / "vocab.json"
    dst_merges = output_dir / "merges.txt"

    # 已就绪（3 个扁平文件齐全）则跳过
    if dst_prune.exists() and dst_vocab.exists() and dst_merges.exists():
        print(f"标点模型已存在（扁平结构），跳过下载: {output_dir}")
        return output_dir

    print("=" * 60)
    print(f"下载 ngram 标点模型 (order={order}) 到 {output_dir}（扁平结构）")
    print("=" * 60)

    ngram_path = f"{order}gram_trie_a22_q8_b8/prune{prune}.bin"

    print(f"[1/3] n-gram 模型: {ngram_path}")
    if not dst_prune.exists():
        _download(MS_RESOLVE.format(repo=NGRAM_REPO, path=ngram_path), dst_prune)
    else:
        print(f"       已存在，跳过: {dst_prune}")

    print(f"[2/3] BPE 词表 vocab.json ({tokenizer_id})")
    if not dst_vocab.exists():
        _download(MS_RESOLVE.format(repo=tokenizer_id, path="vocab.json"), dst_vocab)
    else:
        print(f"       已存在，跳过: {dst_vocab}")

    print(f"[3/3] BPE 合并表 merges.txt ({tokenizer_id})")
    if not dst_merges.exists():
        _download(MS_RESOLVE.format(repo=tokenizer_id, path="merges.txt"), dst_merges)
    else:
        print(f"       已存在，跳过: {dst_merges}")

    print(f"\n扁平模型文件就绪: {[p.name for p in (dst_prune, dst_vocab, dst_merges)]}")
    print("下载完成！")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="ngram 标点模型下载（HTTP 直链，扁平结构）")
    parser.add_argument("--output-dir", type=str, default="./models/punc", help="输出目录")
    parser.add_argument("--order", type=int, default=3, help="n-gram 阶数（3/4/5/6）")
    parser.add_argument("--tokenizer-id", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="BPE 分词器 ModelScope id")
    args = parser.parse_args()

    download_punc(Path(args.output_dir), args.order, args.tokenizer_id)


if __name__ == "__main__":
    main()
