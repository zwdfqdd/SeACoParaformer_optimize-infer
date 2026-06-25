"""
热词管理接口测试（HTTP 客户端，需服务已启动）

覆盖：
    GET  /hotwords/status    查看当前词表版本状态
    POST /hotwords/reload    重载词表（words / reload_from_file / expected_version）
    POST /hotwords/rollback  回滚到上一版内容

验证点：
    - status 返回 version/md5/count/route/cache_ready
    - reload 成功后 version 递增、count 正确、route 随词表大小切换（≤256=A，>256=B）
    - expected_version 乐观并发：旧版本号被拒（409 / code=1008）
    - 空词表 / 全 OOV 校验失败（400 / code=1000）
    - rollback 以旧内容发布新版本（version 继续递增）

用法：
    python tests/test_hotword_api.py --url http://localhost:8080
    python tests/test_hotword_api.py --url http://localhost:8099
"""

import argparse
import json
import sys

import requests


def _print(title: str, resp: requests.Response):
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"\n[{title}] HTTP {resp.status_code}")
    print(json.dumps(body, ensure_ascii=False, indent=2))
    return body


def run(url: str):
    passed, failed = 0, 0

    def check(cond: bool, msg: str):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  ✓ {msg}")
        else:
            failed += 1
            print(f"  ✗ {msg}")

    # 1. status 初始状态
    r = requests.get(f"{url}/hotwords/status", timeout=10)
    body = _print("status 初始", r)
    check(r.status_code == 200, "status 返回 200")
    check("version" in body and "route" in body, "status 含 version/route 字段")
    base_version = body.get("version", 0)

    # 2. reload 小词表（≤256 → 路径 A）
    small = [f"测试词{i}" for i in range(10)]
    r = requests.post(f"{url}/hotwords/reload", json={"words": small}, timeout=60)
    body = _print("reload 小词表(路径A)", r)
    check(r.status_code == 200, "reload 小词表返回 200")
    check(body.get("route") == "A", "小词表 route=A")
    check(body.get("count") == 10, "count=10")
    check(body.get("version", 0) > base_version, "version 递增")
    v_after_small = body.get("version", 0)

    # 3. reload 大词表（>256 → 路径 B）
    large = [f"词条{i}" for i in range(300)]
    r = requests.post(f"{url}/hotwords/reload", json={"words": large}, timeout=120)
    body = _print("reload 大词表(路径B)", r)
    check(r.status_code == 200, "reload 大词表返回 200")
    check(body.get("route") == "B", "大词表 route=B")
    check(body.get("count") == 300, "count=300")
    v_after_large = body.get("version", 0)

    # 4. 乐观并发：用过时 expected_version 应被拒
    r = requests.post(
        f"{url}/hotwords/reload",
        json={"words": ["新词"], "expected_version": base_version},  # 故意用旧版本号
        timeout=30,
    )
    body = _print("reload 版本冲突", r)
    check(r.status_code == 409, "版本冲突返回 409")
    check(body.get("code") == 1008, "错误码 1008(HOTWORD_VERSION_CONFLICT)")

    # 5. 空词表校验失败
    r = requests.post(f"{url}/hotwords/reload", json={"words": []}, timeout=30)
    body = _print("reload 空词表", r)
    check(r.status_code == 400, "空词表返回 400")
    check(body.get("code") == 1000, "错误码 1000(INPUT_PARAM_FAILED)")

    # 6. 既不传 words 也不 reload_from_file
    r = requests.post(f"{url}/hotwords/reload", json={}, timeout=30)
    body = _print("reload 参数缺失", r)
    check(r.status_code == 400, "参数缺失返回 400")

    # 7. rollback（回到上一版内容，version 继续递增）
    r = requests.post(f"{url}/hotwords/rollback", timeout=60)
    body = _print("rollback", r)
    check(r.status_code == 200, "rollback 返回 200")
    check(body.get("version", 0) > v_after_large, "rollback 后 version 继续递增")

    print(f"\n{'=' * 50}\n结果: {passed} 通过, {failed} 失败\n{'=' * 50}")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="热词管理接口测试")
    parser.add_argument("--url", default="http://localhost:8080", help="服务地址")
    args = parser.parse_args()

    # 健康检查
    try:
        h = requests.get(f"{args.url}/health", timeout=5).json()
        print(f"服务状态: {h}")
    except Exception as e:
        sys.exit(f"服务不可用: {e}")

    ok = run(args.url)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
