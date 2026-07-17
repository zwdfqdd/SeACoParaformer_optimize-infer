"""
Prometheus 指标 QPS 监控脚本

从服务 /metrics 端点抓取 fastapi_requests_total 计数器（labels:
method/endpoint/status/http_status），按采样间隔做差分计算 QPS
（本地版 rate()，等价于 Prometheus 的 rate(fastapi_requests_total[interval])）。

多进程部署（WORKERS>1 + PROMETHEUS_MULTIPROC_DIR）下 /metrics 已聚合所有 worker，
本脚本抓到的即全进程累计值，QPS 反映整机真实吞吐（不会像单 worker 那样偏低 1/WORKERS）。

计数器语义：
- Counter 只增不减，进程重启会归零；两次采样差分若为负判定为服务重启，跳过该窗口
- status="success"（http_status<400）/ "error"（>=400）

用法：
    # 持续监控（每 5s 采样一次，Ctrl+C 结束并打印汇总）
    python tests/test_metrics_qps.py --url http://localhost:8080

    # 指定采样间隔与次数（每 2s 采样，采 30 次后退出）
    python tests/test_metrics_qps.py --url http://localhost:8080 --interval 2 --count 30

    # 只打印当前指标快照（不算 QPS，看累计计数分布）
    python tests/test_metrics_qps.py --url http://localhost:8080 --once

    # 只关注某个 endpoint
    python tests/test_metrics_qps.py --url http://localhost:8080 --endpoint /chinese_asr

配合压测：另开一个终端跑 test_service.py 施压，本脚本实时观察各 endpoint QPS。
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict


# 兼容两种样本名：fastapi_requests_total 或（prometheus_client 对已含 _total 的名字
# 再补一次后缀导致的）fastapi_requests_total_total；排除 _created 时间戳样本。
_METRIC = "fastapi_requests_total"
_LINE_RE = re.compile(
    r'^(?P<name>' + re.escape(_METRIC) + r'(?:_total)?)\{(?P<labels>[^}]*)\}\s+(?P<val>[0-9.eE+-]+)\s*$'
)
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def _fetch_metrics(url: str, timeout: float = 10.0) -> str:
    """抓取 /metrics 原始 Prometheus 文本。

    本服务 /metrics 用 JSONResponse 包裹，返回体是 JSON 编码的字符串，
    需先 json.loads 剥壳还原真实 Prometheus 文本。
    """
    endpoint = url.rstrip("/") + "/metrics"
    with urllib.request.urlopen(endpoint, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    # JSONResponse 包裹 → 以双引号开头；直接文本 → 以 # 或指标名开头
    stripped = raw.lstrip()
    if stripped.startswith('"'):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _parse(text: str) -> dict:
    """解析 fastapi_requests_total 各 series 计数。

    返回 {(method, endpoint, status, http_status): value}
    """
    series = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        labels = dict(_LABEL_RE.findall(m.group("labels")))
        key = (
            labels.get("method", ""),
            labels.get("endpoint", ""),
            labels.get("status", ""),
            labels.get("http_status", ""),
        )
        # 多进程聚合可能同 key 多分片，累加
        series[key] = series.get(key, 0.0) + float(m.group("val"))
    return series


def _totals(series: dict):
    """按 status 汇总：返回 (total, success, error)。"""
    total = sum(series.values())
    success = sum(v for k, v in series.items() if k[2] == "success")
    error = sum(v for k, v in series.items() if k[2] == "error")
    return total, success, error


def _print_snapshot(series: dict, endpoint_filter: str | None):
    """打印当前累计计数分布（按 endpoint 分组）。"""
    by_ep = defaultdict(lambda: defaultdict(float))
    for (method, ep, status, http_status), v in series.items():
        if endpoint_filter and ep != endpoint_filter:
            continue
        by_ep[ep][(method, status, http_status)] += v

    print("\n当前累计计数（fastapi_requests_total）:")
    print("-" * 72)
    for ep in sorted(by_ep):
        ep_total = sum(by_ep[ep].values())
        print(f"  {ep}  (合计 {int(ep_total)})")
        for (method, status, http_status), v in sorted(by_ep[ep].items()):
            print(f"      {method:5s} status={status:8s} http={http_status:4s}  {int(v)}")
    print("-" * 72)


def _diff_qps(prev: dict, cur: dict, dt: float, endpoint_filter: str | None):
    """两次采样差分算 QPS。返回 (total_qps, success_qps, error_qps, per_ep)。

    per_ep: {endpoint: (total_qps, success_qps, error_qps)}
    """
    def _sel(series):
        if not endpoint_filter:
            return series
        return {k: v for k, v in series.items() if k[1] == endpoint_filter}

    prev_s, cur_s = _sel(prev), _sel(cur)

    # 全局
    p_total, p_ok, p_err = _totals(prev_s)
    c_total, c_ok, c_err = _totals(cur_s)
    d_total, d_ok, d_err = c_total - p_total, c_ok - p_ok, c_err - p_err

    # 计数器回退（服务重启）→ 本窗口无效
    restarted = d_total < 0 or d_ok < 0 or d_err < 0

    per_ep = {}
    eps = {k[1] for k in cur_s} | {k[1] for k in prev_s}
    for ep in eps:
        pt, po, pe = _totals({k: v for k, v in prev_s.items() if k[1] == ep})
        ct, co, ce = _totals({k: v for k, v in cur_s.items() if k[1] == ep})
        per_ep[ep] = ((ct - pt) / dt, (co - po) / dt, (ce - pe) / dt)

    if restarted:
        return None, None, None, per_ep
    return d_total / dt, d_ok / dt, d_err / dt, per_ep


def main():
    parser = argparse.ArgumentParser(description="Prometheus fastapi_requests_total QPS 监控")
    parser.add_argument("--url", default="http://localhost:8080", help="服务地址")
    parser.add_argument("--interval", type=float, default=5.0, help="采样间隔（秒），默认 5")
    parser.add_argument("--count", type=int, default=0, help="采样次数，0=无限直到 Ctrl+C")
    parser.add_argument("--once", action="store_true", help="只打印当前快照，不算 QPS")
    parser.add_argument("--endpoint", default=None, help="只统计指定 endpoint（如 /chinese_asr）")
    args = parser.parse_args()

    # 连通性 + 首次抓取
    try:
        text = _fetch_metrics(args.url)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"无法访问 {args.url}/metrics: {e}")
        sys.exit(1)

    series = _parse(text)
    if not series:
        print(f"警告: /metrics 未找到 {_METRIC} 指标（服务可能刚启动，尚无请求计数）")

    print("=" * 72)
    print("Prometheus QPS 监控 — fastapi_requests_total")
    print("=" * 72)
    print(f"服务:     {args.url}")
    print(f"采样间隔: {args.interval}s")
    if args.endpoint:
        print(f"过滤:     endpoint={args.endpoint}")

    if args.once:
        _print_snapshot(series, args.endpoint)
        return

    print(f"采样次数: {'无限（Ctrl+C 结束）' if args.count == 0 else args.count}")
    print("-" * 72)

    prev = series
    prev_t = time.time()
    peak_qps = 0.0
    ok_qps_samples = []
    i = 0
    try:
        while args.count == 0 or i < args.count:
            time.sleep(args.interval)
            i += 1
            try:
                cur = _parse(_fetch_metrics(args.url))
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                print(f"[{i}] 抓取失败: {e}")
                continue
            now = time.time()
            dt = now - prev_t

            total_qps, ok_qps, err_qps, per_ep = _diff_qps(prev, cur, dt, args.endpoint)
            ts = time.strftime("%H:%M:%S")
            if total_qps is None:
                print(f"[{ts}] 检测到计数回退（服务重启？），跳过本窗口")
            else:
                peak_qps = max(peak_qps, ok_qps)
                ok_qps_samples.append(ok_qps)
                print(
                    f"[{ts}] QPS 总={total_qps:7.2f}  成功={ok_qps:7.2f}  "
                    f"失败={err_qps:6.2f}  (窗口 {dt:.1f}s)"
                )
                if not args.endpoint and len(per_ep) > 1:
                    for ep in sorted(per_ep):
                        t_q, o_q, e_q = per_ep[ep]
                        if t_q > 0.01:
                            print(f"           └ {ep:20s} 总={t_q:7.2f} 成功={o_q:7.2f} 失败={e_q:6.2f}")

            prev, prev_t = cur, now
    except KeyboardInterrupt:
        print("\n(已中断)")

    # 汇总
    print("-" * 72)
    if ok_qps_samples:
        avg = sum(ok_qps_samples) / len(ok_qps_samples)
        print(f"汇总: 成功 QPS 平均={avg:.2f}  峰值={peak_qps:.2f}  (采样 {len(ok_qps_samples)} 次)")
    else:
        print("汇总: 无有效 QPS 采样")
    print("=" * 72)


if __name__ == "__main__":
    main()
