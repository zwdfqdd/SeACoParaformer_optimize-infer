"""
错误路径 + 健康/指标接口测试（HTTP 客户端，需服务已启动）

覆盖：
    GET  /health             健康检查（status/device/models_loaded）
    GET  /metrics            Prometheus 指标（含 3 个指标名）
    POST /chinese_asr 错误码：
        1000 INPUT_PARAM_FAILED  空音频数据
        1001 DECODE_FAILED       非法 base64 / 非 WAV / 采样率不符
        1005 AUDIO_TOO_LONG      超时长上限（需服务端设较小 MAX_AUDIO_DURATION_MS 才能触发）

依赖：仅 Python 标准库（urllib），推理镜像无需额外安装。

用法：
    python tests/test_error_api.py --url http://localhost:8080
"""

import argparse
import base64
import io
import json
import sys
import urllib.request
import urllib.error


def _http(method: str, url: str, payload=None, timeout: int = 30):
    """发起 HTTP 请求，返回 (status_code, body, raw_text)。"""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
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
    return status, body, raw


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
    status, body, _ = _http("GET", f"{url}/health", timeout=10)
    print(f"\n[health] {status} {body}")
    check(status == 200, "health 返回 200")
    check(isinstance(body, dict) and body.get("status") == "ok", "status=ok")
    check(isinstance(body, dict) and "device" in body and "models_loaded" in body, "含 device/models_loaded")

    # 2. /metrics
    status, _, text = _http("GET", f"{url}/metrics", timeout=10)
    print(f"\n[metrics] {status}, len={len(text)}")
    check(status == 200, "metrics 返回 200")
    check("asr_request_total" in text, "含 asr_request_total")
    check("asr_inference_duration_seconds" in text, "含 asr_inference_duration_seconds")

    # 3. 非法 base64 → 1001 DECODE_FAILED
    status, body, _ = _http("POST", f"{url}/chinese_asr", {"b64": "@@@not-base64@@@"}, timeout=30)
    print(f"\n[非法base64] {status} {body}")
    check(status == 400, "非法 base64 返回 400")
    check(isinstance(body, dict) and body.get("code") in (1000, 1001), "错误码 1000/1001")

    # 4. 合法 base64 但非 WAV → 1001
    status, body, _ = _http("POST", f"{url}/chinese_asr", {"b64": _b64(b"this is not a wav file")}, timeout=30)
    print(f"\n[非WAV] {status} {body}")
    check(status == 400, "非 WAV 返回 400")
    check(isinstance(body, dict) and body.get("code") == 1001, "错误码 1001(DECODE_FAILED)")

    # 5. 采样率不符（8kHz）→ 1001
    wav_8k = _make_wav(8000, 8000)  # 1s @ 8kHz
    status, body, _ = _http("POST", f"{url}/chinese_asr", {"b64": _b64(wav_8k)}, timeout=30)
    print(f"\n[采样率8k] {status} {body}")
    check(status == 400, "采样率不符返回 400")
    check(isinstance(body, dict) and body.get("code") == 1001, "错误码 1001")

    # 6. 缺少 b64 字段 → 422（pydantic 校验）或 400
    status, body, _ = _http("POST", f"{url}/chinese_asr", {"hotwords": ["张三"]}, timeout=30)
    print(f"\n[缺b64] {status}")
    check(status in (400, 422), "缺 b64 返回 400/422")

    # 7. 空字符串 b64 → 1000/1001
    status, body, _ = _http("POST", f"{url}/chinese_asr", {"b64": ""}, timeout=30)
    print(f"\n[空b64] {status} {body}")
    check(status == 400, "空 b64 返回 400")

    print(f"\n{'=' * 50}\n结果: {passed} 通过, {failed} 失败\n{'=' * 50}")
    print("提示：1005 AUDIO_TOO_LONG 需服务端设较小 MAX_AUDIO_DURATION_MS（如 5000）"
          "并传 >5s 音频才能触发，本脚本未覆盖。")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="错误路径 + 健康/指标接口测试")
    parser.add_argument("--url", default="http://localhost:8080", help="服务地址")
    args = parser.parse_args()

    try:
        _http("GET", f"{args.url}/health", timeout=5)
    except Exception as e:
        sys.exit(f"服务不可用: {e}")

    ok = run(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
