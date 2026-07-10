# -*- coding: utf-8 -*-
"""
CT-Transformer 中文标点模型下载脚本（句子级时间戳用）

直接从 ModelScope HTTP 直链下载到本地目录（扁平结构，无需 modelscope 库，
不产生 {org}--{repo} 嵌套缓存目录），供 src/sentence_segmenter.py 离线加载。

模型来源:
    iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx
    （CT-Transformer，纯 onnxruntime 推理；逐 token 标点分类，长文本/重复口语稳定）

最终 models/punc/ 下扁平文件：
    model_quant.onnx（量化，~280MB）/ tokens.json（272727 CharTokenizer）/ config.yaml

用法:
    python scripts/download_punc.py                       # 默认 models/punc
    python scripts/download_punc.py --output-dir ./models/punc
    python scripts/download_punc.py --onnx model.onnx     # 下非量化版
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

MODEL_ID = "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx"
MS_RESOLVE = "https://modelscope.cn/models/{repo}/resolve/master/{path}"

# 核心文件（onnx 按参数选量化/非量化）；config.yaml 提供 punc_list 等元信息
CORE_FILES = ("tokens.json", "config.yaml")


def _make_progress():
    """urlretrieve reporthook：单行刷新 百分比 + 已下载/总大小 + 速度。"""
    state = {"t0": time.time()}

    def _hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        elapsed = max(time.time() - state["t0"], 1e-6)
        speed = downloaded / elapsed / (1024 * 1024)
        dl_mb = downloaded / (1024 * 1024)
        if total_size > 0:
            total_mb = total_size / (1024 * 1024)
            pct = min(downloaded * 100 / total_size, 100.0)
            sys.stdout.write(
                f"\r       {pct:5.1f}%  {dl_mb:7.1f}/{total_mb:.1f} MB  {speed:5.1f} MB/s"
            )
        else:
            sys.stdout.write(f"\r       {dl_mb:7.1f} MB  {speed:5.1f} MB/s")
        sys.stdout.flush()

    return _hook


def _download(path: str, dst: Path):
    """HTTP 下载单文件到 dst（带进度）。"""
    url = MS_RESOLVE.format(repo=MODEL_ID, path=path)
    print(f"  下载: {path}")
    urllib.request.urlretrieve(url, str(dst), reporthook=_make_progress())
    sys.stdout.write("\n")
    print(f"       → {dst} ({dst.stat().st_size / (1024*1024):.1f} MB)")


def download_punc(output_dir: Path, onnx_name: str):
    """下载 CT-Transformer 标点模型到 output_dir（扁平结构）。已就绪则跳过。"""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dst_onnx = output_dir / onnx_name
    dst_tokens = output_dir / "tokens.json"
    dst_config = output_dir / "config.yaml"

    if dst_onnx.exists() and dst_tokens.exists() and dst_config.exists():
        print(f"标点模型已存在（扁平结构），跳过下载: {output_dir}")
        return output_dir

    print("=" * 60)
    print(f"下载 CT-Transformer 标点模型到 {output_dir}（扁平结构）")
    print(f"模型: {MODEL_ID}")
    print("=" * 60)

    if not dst_onnx.exists():
        _download(onnx_name, dst_onnx)
    else:
        print(f"  已存在，跳过: {onnx_name}")
    for name in CORE_FILES:
        dst = output_dir / name
        if dst.exists():
            print(f"  已存在，跳过: {name}")
            continue
        _download(name, dst)

    print(f"\n扁平模型文件就绪: {[onnx_name, 'tokens.json', 'config.yaml']}")
    print("下载完成！")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="CT-Transformer 标点模型下载（HTTP 直链，扁平结构）")
    parser.add_argument("--output-dir", type=str, default="./models/punc", help="输出目录")
    parser.add_argument("--onnx", type=str, default="model_quant.onnx",
                        help="ONNX 文件名（默认量化版 model_quant.onnx；非量化用 model.onnx）")
    args = parser.parse_args()

    download_punc(Path(args.output_dir), args.onnx)


if __name__ == "__main__":
    main()
