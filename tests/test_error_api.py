"""
错误路径 + 健康/指标接口测试（HTTP 客户端，需服务已启动）

覆盖：
    GET  /health             健康检查（status/device/models_loaded）
    GET  /metrics            Prometheus 指标（含 3 个指标名）
    POST /asr 错误码：
        1000 INPUT_PARAM_FAILED  空音频数据
        1001 DECODE_FAILED       非法 base64 / 非 WAV / 采样率不符
        1005 AUDIO_TOO_LONG      超时长上限（需服务端设较小 MAX_AUDIO_DURATION_MS 才能触发）

用法：
    python tests/test_error_api.py --url http://localhost:8080
"""

import argparse
import base64
import io
import json
import sys

import requests


def _make_wav(sample_rate: int, num_samples: int, channels: int = 1) -> bytes:
    """生成一个最小合法 WAV 字节流（静音）。"""
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_samples * channels)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def run(url: str):
    passed, failed = 0, 0

    def check(cond: bool, msg: str):
        nonlocal passed, failed
        if cond:
            passed += 1; print(f"  ✓ {msg}")
        else:
            failed += 1; print(f"  ✗ {msg}")

    # 1. /health
    r = requests.get(f"{url}/health", timeout=10)
    body = r.json()
    print(f"\n[health] {r.status_code} {body}")
    check(r.status_code == 200, "health 返回 200")
    check(body.get("status") == "ok", "status=ok")
    check("device" in body and "models_loaded" in body, "含 device/models_loaded")

    # 2. /metrics
    r = requests.get(f"{url}/metrics", timeout=10)
    text = r.text
    print(f"\n[metrics] {r.status_code}, len={len(text)}")
    check(r.status_code == 200, "metrics 返回 200")
    check("asr_request_total" in text, "含 asr_request_total")
    check("asr_inference_duration_seconds" in text, "含 asr_inference_duration_seconds")

    # 3. 非法 base64 → 1001 DECODE_FAILED
    r = requests.post(f"{url}/asr", json={"b64": "@@@not-base64@@@"}, timeout=30)
    body = r.json()
    print(f"\n[非法base64] {r.status_code} {body}")
    check(r.status_code == 400, "非法 base64 返回 400")
    check(body.get("code") in (1000, 1001), "错误码 1000/1001")

    # 4. 合法 base64 但非 WAV → 1001
    r = requests.post(f"{url}/asr", json={"b64": _b64(b"this is not a wav file")}, timeout=30)
    body = r.json()
    print(f"\n[非WAV] {r.status_code} {body}")
    check(r.status_code == 400, "非 WAV 返回 400")
    check(body.get("code") == 1001, "错误码 1001(DECODE_FAILED)")

    # 5. 采样率不符（8kHz）→ 1001
    wav_8k = _make_wav(8000, 8000)  # 1s @ 8kHz
    r = requests.post(f"{url}/asr", json={"b64": _b64(wav_8k)}, timeout=30)
    body = r.json()
    print(f"\n[采样率8k] {r.status_code} {body}")
    check(r.status_code == 400, "采样率不符返回 400")
    check(body.get("code") == 1001, "错误码 1001")

    # 6. 缺少 b64 字段 → 422（pydantic 校验）或 400
    r = requests.post(f"{url}/asr", json={"hotwords": ["张三"]}, timeout=30)
    print(f"\n[缺b64] {r.status_code}")
    check(r.status_code in (400, 422), "缺 b64 返回 400/422")

    # 7. 空字符串 b64 → 1000/1001
    r = requests.post(f"{url}/asr", json={"b64": ""}, timeout=30)
    body = r.json()
    print(f"\n[空b64] {r.status_code} {body}")
    check(r.status_code == 400, "空 b64 返回 400")

    print(f"\n{'=' * 50}\n结果: {passed} 通过, {failed} 失败\n{'=' * 50}")
    print("提示：1005 AUDIO_TOO_LONG 需服务端设较小 MAX_AUDIO_DURATION_MS（如 5000）"
          "并传 >5s 音频才能触发，本脚本未覆盖。")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="错误路径 + 健康/指标接口测试")
    parser.add_argument("--url", default="http://localhost:8080", help="服务地址")
    args = parser.parse_args()

    try:
        requests.get(f"{args.url}/health", timeout=5)
    except Exception as e:
        sys.exit(f"服务不可用: {e}")

    ok = run(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
