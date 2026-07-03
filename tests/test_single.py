"""
单次请求测试脚本（简化版）

输入示例：
    {
        "base64": "UklGRi4AAABXQVZFZm10IBAAAA...",
        "hotwords": ["张三", "李四"]
    }

输出示例（成功）：
    {
        "code": 0,
        "article_url": null,
        "istar_asr": "今天天气真好适合出去走走",
        "asr": [
            {"idx": 0, "slid": "", "text": "今天天气真好", "speaker": "", "timestamp": [0.0, 5.2]},
            {"idx": 1, "slid": "", "text": "适合出去走走", "speaker": "", "timestamp": [5.2, 9.8]}
        ]
    }

输出示例（失败）：
    {
        "code": 1001,
        "article_url": null,
        "istar_asr": "",
        "asr": [],
        "error": "DECODE_FAILED",
        "message": "音频解码失败，请确认为16kHz单声道WAV格式"
    }

用法：
    python tests/test_single.py --audio test_data/audio_16000_30s.wav
    python tests/test_single.py --audio test_data/audio_16000_30s.wav --url http://localhost:8099

依赖：仅 Python 标准库（urllib），推理镜像无需额外安装。
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="单次 ASR 请求测试")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--url", default="http://localhost:8080", help="服务地址")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"音频不存在: {args.audio}")

    # Base64 编码
    with open(args.audio, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode()

    payload = {"base64": b64_audio}
    if args.hotwords:
        payload["hotwords"] = args.hotwords

    # 发送请求
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{args.url}/chinese_asr", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        status = e.code
        raw = e.read().decode("utf-8")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        result = json.loads(raw)
        pretty = json.dumps(result, ensure_ascii=False, indent=2)
    except Exception:
        pretty = raw
    print(f"状态: {status} | 耗时: {elapsed_ms:.0f}ms")
    print(pretty)


if __name__ == "__main__":
    main()
