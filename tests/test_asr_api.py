"""
ASR HTTP 接口冒烟测试（最简单的单文件请求）

输入支持：
    .wav  → 读取并 base64 编码
    .txt  → 直接读取已 base64 的字符串

依赖：仅 Python 标准库（urllib），推理镜像无需额外安装。

用法：
    python tests/test_asr_api.py test_data/audio_16000_10s.wav
    python tests/test_asr_api.py test_data/audio_16000_10s.wav --url http://localhost:8099 --hotwords 埃文 账号
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
import urllib.error


def wav2b64(wavpath: str) -> str:
    """将 WAV 文件转换为 base64 字符串。"""
    with open(wavpath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_b64_data(b64_file: str) -> str:
    """从文本文件读取 base64 字符串。"""
    with open(b64_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def _post_json(url: str, payload: dict, timeout: int = 120):
    """POST JSON，返回 (status_code, body)。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        status = e.code
        raw = e.read().decode("utf-8")
    try:
        body = json.loads(raw)
    except Exception:
        body = raw
    return status, body


def main():
    parser = argparse.ArgumentParser(description="ASR HTTP 接口冒烟测试")
    parser.add_argument("file_path", help="WAV 或 含 base64 的 txt 文件")
    parser.add_argument("--url", default="http://localhost:8080", help="服务地址")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    args = parser.parse_args()

    if args.file_path.endswith(".txt"):
        data = get_b64_data(args.file_path)
    elif args.file_path.endswith(".wav"):
        data = wav2b64(args.file_path)
    else:
        sys.exit("仅支持 .wav 或 .txt 输入")

    payload = {"b64": data}
    if args.hotwords:
        payload["hotwords"] = args.hotwords

    t0 = time.time()
    status, body = _post_json(f"{args.url}/chinese_asr", payload, timeout=120)
    elapsed = time.time() - t0

    print(f"HTTP {status} | 耗时 {elapsed:.3f}s")
    if isinstance(body, (dict, list)):
        print(json.dumps(body, ensure_ascii=False, indent=2))
    else:
        print(body)


if __name__ == "__main__":
    main()
