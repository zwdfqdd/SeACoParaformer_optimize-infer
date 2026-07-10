# -*- coding: utf-8 -*-
"""
SeACo-Paraformer ASR 模型（PT 权重）下载脚本

直接从 ModelScope HTTP 下载核心文件到本地 PT_MODEL_DIR（默认 models/asr/pt，
扁平结构，无需 modelscope 库，不产生嵌套缓存目录），供 PT 后端推理 或
prepare_model.py 转换 ONNX/TRT engine 的上游。

模型来源（ModelScope resolve 直链）:
    iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

核心文件（扁平放置到 models/asr/pt/）:
    model.pt / config.yaml / am.mvn / tokens.json / seg_dict / configuration.json

用法:
    python scripts/download_asr.py                       # 默认下载到 models/asr/pt
    python scripts/download_asr.py --output-dir ./models/asr/pt
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
MS_RESOLVE = "https://modelscope.cn/models/{repo}/resolve/master/{path}"

# 需要下载的文件（PT 推理 / 转换所需）。model.pt 为权重，其余为配置。
# configuration.json 为 ModelScope 元数据，可选（失败不阻断）。
CORE_FILES = ("model.pt", "config.yaml", "am.mvn", "tokens.json", "seg_dict")
OPTIONAL_FILES = ("configuration.json",)


def _has_weights(output_dir: Path) -> bool:
    """判断目录内是否已有 PT 权重（model.pt 或任意 .pt/.pth）。"""
    if (output_dir / "model.pt").exists():
        return True
    for ext in ("*.pt", "*.pth"):
        if list(output_dir.rglob(ext)):
            return True
    return False


def _make_progress():
    """构造 urlretrieve 的 reporthook：单行刷新 百分比 + 已下载/总大小 + 速度。"""
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


def _download(path: str, dst: Path, required: bool) -> bool:
    """HTTP 下载单文件到 dst，带进度显示。required=False 时失败仅告警。返回是否成功。"""
    url = MS_RESOLVE.format(repo=MODEL_ID, path=path)
    try:
        print(f"  下载: {path}")
        urllib.request.urlretrieve(url, str(dst), reporthook=_make_progress())
        sys.stdout.write("\n")
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"       → {dst} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        level = "错误" if required else "警告"
        print(f"  [{level}] 下载失败 {path}: {e}")
        return False


def download_asr(output_dir: Path):
    """下载 SeACo-Paraformer PT 权重 + 配置到 output_dir（扁平结构）。已存在则跳过。"""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if _has_weights(output_dir):
        print(f"ASR PT 权重已存在，跳过下载: {output_dir}")
        return output_dir

    print("=" * 60)
    print(f"下载 SeACo-Paraformer ASR PT 权重到 {output_dir}（扁平结构）")
    print(f"模型: {MODEL_ID}")
    print("=" * 60)

    ok = True
    for name in CORE_FILES:
        dst = output_dir / name
        if dst.exists():
            print(f"  已存在，跳过: {name}")
            continue
        if not _download(name, dst, required=True):
            ok = False
    for name in OPTIONAL_FILES:
        dst = output_dir / name
        if not dst.exists():
            _download(name, dst, required=False)

    _verify(output_dir)
    print("\n下载完成！" if ok else "\n[警告] 部分核心文件下载失败，请检查上方日志")
    return output_dir


def _verify(output_dir: Path):
    """核对核心文件是否齐全。"""
    print("\n核对核心文件:")
    missing = []
    for name in CORE_FILES:
        exists = (output_dir / name).exists()
        print(f"  {name}: {'OK' if exists else '缺失'}")
        if not exists:
            missing.append(name)
    if missing:
        print(f"[警告] 缺失核心文件: {missing}（可能影响 PT 推理/转换）")


def main():
    parser = argparse.ArgumentParser(description="SeACo-Paraformer ASR PT 权重下载（HTTP 直链）")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join("models", "asr", "pt"),
                        help="输出目录（PT_MODEL_DIR）")
    args = parser.parse_args()

    download_asr(Path(args.output_dir))


if __name__ == "__main__":
    main()
