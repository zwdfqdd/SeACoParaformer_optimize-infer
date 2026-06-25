"""
Silero VAD 模型下载脚本

直接下载官方 ONNX 格式的 Silero VAD 模型，无需自行转换。
模型来源: https://github.com/snakers4/silero-vad
"""

import argparse
from pathlib import Path


def download_silero_vad(output_dir: Path):
    """
    下载 Silero VAD ONNX 模型。
    优先从 GitHub 直接下载 ONNX 文件，备选使用 torch.hub。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "silero_vad.onnx"

    if onnx_path.exists():
        print(f"VAD 模型已存在: {onnx_path}")
        _verify_model(onnx_path)
        return onnx_path

    # 方式一：直接从 modelscope 下载 ONNX 文件（推荐，无需 torch 依赖）
    try:
        import urllib.request

        url = "https://modelscope.cn/models/pengzhendong/silero-vad/resolve/master/silero_vad.onnx"
        print(f"正在从 ModelScope 下载 Silero VAD ONNX...")
        print(f"   URL: {url}")
        urllib.request.urlretrieve(url, str(onnx_path))
        print(f"   下载完成: {onnx_path}")
        _verify_model(onnx_path)
        return onnx_path
    except Exception as e:
        print(f"   直接下载失败 ({e})，尝试github方式...")


    # 方式二：直接从 GitHub 下载 ONNX 文件（推荐，无需 torch 依赖）
    try:
        import urllib.request

        url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
        print(f"正在从 GitHub 下载 Silero VAD ONNX...")
        print(f"   URL: {url}")
        urllib.request.urlretrieve(url, str(onnx_path))
        print(f"   下载完成: {onnx_path}")
        _verify_model(onnx_path)
        return onnx_path
    except Exception as e:
        print(f"   直接下载失败 ({e})，尝试torch.hub方式...")


    # 方式三：通过 torch.hub 获取
    try:
        import torch
        import shutil
        import glob

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=True,
        )

        # 查找 hub cache 中的 onnx 文件
        hub_dir = torch.hub.get_dir()
        vad_onnx_files = glob.glob(
            f"{hub_dir}/**/silero_vad.onnx", recursive=True
        )

        if vad_onnx_files:
            shutil.copy2(vad_onnx_files[0], onnx_path)
            print(f"   VAD 模型已保存: {onnx_path}")
            _verify_model(onnx_path)
            return onnx_path
    except Exception as e:
        print(f"   torch.hub 方式也失败: {e}")

    # 方式四：通过 pip 包 silero-vad 获取内置 ONNX
    try:
        import silero_vad
        import shutil

        pkg_dir = Path(silero_vad.__file__).parent
        pkg_onnx = pkg_dir / "data" / "silero_vad.onnx"
        if pkg_onnx.exists():
            shutil.copy2(pkg_onnx, onnx_path)
            print(f"   从 silero-vad 包复制: {onnx_path}")
            _verify_model(onnx_path)
            return onnx_path
    except ImportError:
        pass

    print("错误：所有下载方式均失败，请手动下载 silero_vad.onnx")
    raise RuntimeError("无法获取 Silero VAD ONNX 模型")


def _verify_model(onnx_path: Path):
    """验证 ONNX 模型可正常加载。"""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    inputs = [f"{i.name}({i.shape})" for i in session.get_inputs()]
    print(f"   模型验证通过，输入: {inputs}")
    print(f"   模型大小: {onnx_path.stat().st_size / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Silero VAD 模型下载")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/vad",
        help="输出目录",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Silero VAD ONNX 模型下载")
    print("=" * 60)

    download_silero_vad(output_dir)

    print()
    print("下载完成！")


if __name__ == "__main__":
    main()
