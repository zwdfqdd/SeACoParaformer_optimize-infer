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

# 核心文件（onnx 按参数选量化/非量化）。运行时仅需 onnx + tokens.json；
# config.yaml 仅作模型溯源/参考（punc_list 已固化于 sentence_segmenter._PUNC_LIST，运行时不解析）。
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


# 各文件最小合理大小（字节），低于此判定为残缺下载（防中断残缺文件静默通过）
_MIN_SIZE = {
    "model_quant.onnx": 200 * 1024 * 1024,  # 量化版 ~280MB
    "model.onnx": 200 * 1024 * 1024,         # 非量化同量级
    "tokens.json": 1 * 1024 * 1024,          # 272727 词表 ~4MB
    "config.yaml": 100,                       # 几百字节
}


def _download(path: str, dst: Path, retries: int = 2):
    """HTTP 下载单文件到 dst（带进度 + 完整性校验，残缺自动重试）。"""
    url = MS_RESOLVE.format(repo=MODEL_ID, path=path)
    min_size = _MIN_SIZE.get(dst.name, 0)
    for attempt in range(1, retries + 2):
        print(f"  下载: {path}" + (f"（第 {attempt} 次）" if attempt > 1 else ""))
        try:
            urllib.request.urlretrieve(url, str(dst), reporthook=_make_progress())
            sys.stdout.write("\n")
        except Exception as e:
            sys.stdout.write("\n")
            print(f"       下载出错: {e}")
            if dst.exists():
                dst.unlink()
            continue
        size = dst.stat().st_size
        if size < min_size:
            print(f"       [残缺] {dst.name} 仅 {size/(1024*1024):.1f} MB "
                  f"(< {min_size/(1024*1024):.0f} MB)，删除重试")
            dst.unlink()
            continue
        print(f"       → {dst} ({size / (1024*1024):.1f} MB)")
        return
    raise RuntimeError(f"下载 {path} 失败（残缺/网络问题），已重试 {retries} 次")


def download_punc(output_dir: Path, onnx_name: str):
    """下载 CT-Transformer 标点模型到 output_dir（扁平结构）。已就绪则跳过。"""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dst_onnx = output_dir / onnx_name
    dst_tokens = output_dir / "tokens.json"
    dst_config = output_dir / "config.yaml"

    def _ok(p: Path) -> bool:
        return p.exists() and p.stat().st_size >= _MIN_SIZE.get(p.name, 0)

    # 完整（大小达标）才跳过；残缺文件（如中断的 33MB onnx）会重新下载
    if _ok(dst_onnx) and _ok(dst_tokens) and _ok(dst_config):
        print(f"标点模型已存在（扁平结构，完整性校验通过），跳过下载: {output_dir}")
        return output_dir

    print("=" * 60)
    print(f"下载 CT-Transformer 标点模型到 {output_dir}（扁平结构）")
    print(f"模型: {MODEL_ID}")
    print("=" * 60)

    # 逐文件用 _ok（大小校验）判断：残缺文件（中断的部分下载）会重新下载，
    # 而非因"存在"被跳过（否则残缺文件永不替换，且与顶层跳过/加载侧校验形成死循环）。
    if not _ok(dst_onnx):
        _download(onnx_name, dst_onnx)
    else:
        print(f"  已存在（完整），跳过: {onnx_name}")
    for name in CORE_FILES:
        dst = output_dir / name
        if _ok(dst):
            print(f"  已存在（完整），跳过: {name}")
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
