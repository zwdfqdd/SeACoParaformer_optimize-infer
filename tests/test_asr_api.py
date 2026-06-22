"""
v1 ASR HTTP 服务测试

@author: zw
"""
import base64
import sys
import time

import requests

api_url = "http://0.0.0.0:8080/chinese_asr"


def wav2b64(wavpath: str) -> str:
    """将 WAV 文件转换为 base64 字符串。"""
    with open(wavpath, "rb") as f:
        wav_data = f.read()
    return base64.b64encode(wav_data).decode("utf-8")


def get_b64_data(b64_file: str) -> str:
    """从文本文件读取 base64 字符串。"""
    return open(b64_file, "r").read()


def req_main(file_path: str):
    t0 = time.time()

    if file_path.endswith(".txt"):
        data = get_b64_data(file_path)
    elif file_path.endswith(".wav"):
        data = wav2b64(file_path)
    else:
        print("inputs error")
        sys.exit()

    r = requests.post(api_url, json={"base64": data, "article_url": None}, timeout=60)
    t1 = time.time()
    print(r.json())
    print(f"time: {t1 - t0:.3f}s")


if __name__ == "__main__":
    file_path = sys.argv[1]
    req_main(file_path)
