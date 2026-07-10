"""
推理服务性能测试脚本

测试指标：
- RTF (Real-Time Factor): 推理耗时 / 音频时长（越小越好）
- RTX (加速比): 音频时长 / 推理耗时（越大越好）
- 吞吐量 (Throughput): 单位时间处理的音频秒数（audio_seconds/wall_seconds）
- QPS (Queries Per Second): 每秒完成的请求数
- 延迟分布: P50/P90/P95/P99

支持并发压测，模拟多客户端同时请求。

用法：
    # 单请求延迟测试
    python tests/test_service.py --audio test_data/audio_16000_30s.wav --url http://localhost:8080

    # 并发压测（10 并发，共 50 请求）
    python tests/test_service.py --audio test_data/audio_16000_30s.wav --concurrency 10 --total 50

    # 自定义服务地址
    python tests/test_service.py --audio test_data/audio_16000_30s.wav --url http://localhost:8099 --concurrency 20 --total 100
"""

import argparse
import base64
import json
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class GPUMonitor:
    """压测期间后台采样 GPU 利用率 + 显存（基于 nvidia-smi，零依赖）。

    每 interval 秒采样一次 utilization.gpu(%) 与 memory.used(MiB)，
    停止后给出均值/峰值。nvidia-smi 不可用（无 GPU/CPU 部署）时自动禁用，不报错。
    多卡时默认监测 --gpu-index 指定的卡（默认 0）。
    """

    def __init__(self, gpu_index: int = 0, interval: float = 0.5):
        self._gpu_index = gpu_index
        self._interval = interval
        self._util_samples: list[float] = []
        self._mem_samples: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._available = self._probe()

    @staticmethod
    def _probe() -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
            return True
        except Exception:
            return False

    def _sample_once(self):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 f"--id={self._gpu_index}",
                 "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if not out:
                return
            # 取第一行（单卡）：形如 "45, 12345"
            first = out.splitlines()[0]
            util_s, mem_s = [x.strip() for x in first.split(",")[:2]]
            self._util_samples.append(float(util_s))
            self._mem_samples.append(float(mem_s))
        except Exception:
            pass

    def _loop(self):
        while self._running:
            self._sample_once()
            time.sleep(self._interval)

    def start(self):
        if not self._available:
            print("[GPU监测] nvidia-smi 不可用，跳过 GPU 利用率采样")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if not self._available:
            return {}
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if not self._util_samples:
            return {}
        return {
            "gpu_index": self._gpu_index,
            "samples": len(self._util_samples),
            "util_avg": sum(self._util_samples) / len(self._util_samples),
            "util_max": max(self._util_samples),
            "mem_used_avg_mib": sum(self._mem_samples) / len(self._mem_samples),
            "mem_used_max_mib": max(self._mem_samples),
        }


def _percentile(sorted_vals: list, pct: float) -> float:
    """线性插值百分位（替代 numpy.percentile，零依赖）。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def send_request(url: str, payload: dict) -> dict:
    """发送单个 ASR 请求（同步，urllib），返回结果和耗时。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/chinese_asr", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            elapsed = time.perf_counter() - t0
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
        return {
            "success": status == 200,
            "status": status,
            "elapsed_s": elapsed,
            "text_len": len(body.get("istar_asr", "")) if status == 200 else 0,
            "error": body.get("error", "") if status != 200 else "",
        }
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        try:
            body = json.loads(e.read().decode("utf-8"))
            err = body.get("error", f"HTTP {e.code}")
        except Exception:
            err = f"HTTP {e.code}"
        return {"success": False, "status": e.code, "elapsed_s": elapsed, "text_len": 0, "error": err}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "status": 0,
            "elapsed_s": elapsed,
            "text_len": 0,
            "error": f"{type(e).__name__}: {e}",
        }


def run_benchmark(url: str, payload: dict, concurrency: int, total: int,
                  audio_duration: float, gpu_index: int = 0):
    """并发压测（线程池模拟多客户端），期间后台采样 GPU 利用率。"""
    print(f"开始压测: 并发={concurrency}, 总请求={total}")
    print(f"目标: {url}/chinese_asr")
    print()

    gpu = GPUMonitor(gpu_index=gpu_index)
    gpu.start()
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: send_request(url, payload), range(total)))
    wall_time = time.perf_counter() - wall_start
    gpu_stats = gpu.stop()

    return results, wall_time, gpu_stats


def compute_metrics(results: list, wall_time: float, audio_duration: float, concurrency: int):
    """计算性能指标。"""
    success_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    total = len(results)
    success_count = len(success_results)

    if not success_results:
        print("所有请求失败！")
        for r in failed_results[:5]:
            print(f"  错误: {r['error']}")
        return {}

    latencies = sorted([r["elapsed_s"] for r in success_results])

    # 基础统计
    avg_latency = sum(latencies) / len(latencies)
    p50 = _percentile(latencies, 50)
    p90 = _percentile(latencies, 90)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    min_latency = min(latencies)
    max_latency = max(latencies)

    # 性能指标
    rtf = avg_latency / audio_duration  # 单请求 RTF
    rtx = audio_duration / avg_latency  # 单请求加速比
    qps = success_count / wall_time  # 每秒完成请求数
    throughput = qps * audio_duration  # 每秒处理的音频秒数

    metrics = {
        "total_requests": total,
        "success": success_count,
        "failed": len(failed_results),
        "concurrency": concurrency,
        "wall_time_s": wall_time,
        "audio_duration_s": audio_duration,
        "avg_latency_ms": avg_latency * 1000,
        "p50_ms": p50 * 1000,
        "p90_ms": p90 * 1000,
        "p95_ms": p95 * 1000,
        "p99_ms": p99 * 1000,
        "min_ms": min_latency * 1000,
        "max_ms": max_latency * 1000,
        "rtf": rtf,
        "rtx": rtx,
        "qps": qps,
        "throughput_audio_s_per_s": throughput,
    }

    return metrics


def print_report(metrics: dict):
    """打印性能报告。"""
    if not metrics:
        return

    print("=" * 60)
    print("推理服务性能测试报告")
    print("=" * 60)
    print()

    print(f"请求统计:")
    print(f"  总请求: {metrics['total_requests']}")
    print(f"  成功: {metrics['success']}")
    print(f"  失败: {metrics['failed']}")
    print(f"  并发数: {metrics['concurrency']}")
    print(f"  总耗时: {metrics['wall_time_s']:.2f}s")
    print(f"  音频时长: {metrics['audio_duration_s']:.2f}s")
    print()

    print(f"延迟分布:")
    print(f"  平均: {metrics['avg_latency_ms']:.1f}ms")
    print(f"  P50:  {metrics['p50_ms']:.1f}ms")
    print(f"  P90:  {metrics['p90_ms']:.1f}ms")
    print(f"  P95:  {metrics['p95_ms']:.1f}ms")
    print(f"  P99:  {metrics['p99_ms']:.1f}ms")
    print(f"  Min:  {metrics['min_ms']:.1f}ms")
    print(f"  Max:  {metrics['max_ms']:.1f}ms")
    print()

    print(f"性能指标:")
    print(f"  RTF (单请求):  {metrics['rtf']:.4f}")
    print(f"  RTX (加速比):  {metrics['rtx']:.2f}x")
    print(f"  QPS:           {metrics['qps']:.2f} req/s")
    print(f"  吞吐量:        {metrics['throughput_audio_s_per_s']:.2f} audio_s/s")
    print()

    # 吞吐量解读
    throughput = metrics['throughput_audio_s_per_s']
    if throughput > 100:
        print(f"  → 每秒可处理 {throughput:.0f} 秒音频（极高吞吐）")
    elif throughput > 10:
        print(f"  → 每秒可处理 {throughput:.0f} 秒音频（高吞吐）")
    else:
        print(f"  → 每秒可处理 {throughput:.1f} 秒音频")

    # GPU 利用率（压测期间后台采样）
    gpu = metrics.get("gpu")
    if gpu:
        print()
        print(f"GPU 利用率（卡 {gpu['gpu_index']}，采样 {gpu['samples']} 次）:")
        print(f"  利用率 平均: {gpu['util_avg']:.1f}%  峰值: {gpu['util_max']:.0f}%")
        print(f"  显存占用 平均: {gpu['mem_used_avg_mib']:.0f} MiB  "
              f"峰值: {gpu['mem_used_max_mib']:.0f} MiB")


def main():
    parser = argparse.ArgumentParser(description="推理服务性能测试")
    parser.add_argument("--audio", required=True, help="WAV 16kHz 单声道音频")
    parser.add_argument("--url", default="http://localhost:8080", help="服务地址")
    parser.add_argument("--concurrency", type=int, default=1, help="并发数")
    parser.add_argument("--total", type=int, default=10, help="总请求数")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    parser.add_argument("--output", default="service_benchmark.json", help="结果输出文件")
    parser.add_argument("--gpu-index", type=int, default=0, help="监测的 GPU 卡号（nvidia-smi）")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"错误：音频不存在: {args.audio}")

    # 读取音频时长（标准库 wave，仅用于 RTF/RTX 计算）
    import wave
    try:
        with wave.open(args.audio, "rb") as w:
            audio_duration = w.getnframes() / w.getframerate()
    except Exception as e:
        sys.exit(f"错误：无法读取音频时长（需 WAV 格式）: {e}")

    # 读取原始字节做 base64
    with open(args.audio, "rb") as f:
        b64_audio = base64.b64encode(f.read()).decode()

    payload = {"base64": b64_audio}
    if args.hotwords:
        payload["hotwords"] = args.hotwords

    print("=" * 60)
    print("SeACo-Paraformer 推理服务性能测试")
    print("=" * 60)
    print(f"音频: {args.audio} ({audio_duration:.2f}s)")
    print(f"服务: {args.url}")
    print(f"并发: {args.concurrency}")
    print(f"总请求: {args.total}")
    if args.hotwords:
        print(f"热词: {args.hotwords}")
    print()

    # 健康检查（等待服务就绪）
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"{args.url}/health", timeout=5)
        health = json.loads(resp.read())
        if not health.get("models_loaded"):
            print("等待模型加载...")
            for _ in range(30):
                time.sleep(2)
                try:
                    resp = urllib.request.urlopen(f"{args.url}/health", timeout=5)
                    health = json.loads(resp.read())
                    if health.get("models_loaded"):
                        break
                except Exception:
                    pass
            else:
                sys.exit("模型加载超时")
        print(f"服务状态: {health}")
    except Exception as e:
        sys.exit(f"服务不可用: {e}")
    print()

    # 运行压测（含 GPU 利用率采样）
    results, wall_time, gpu_stats = run_benchmark(
        args.url, payload, args.concurrency, args.total, audio_duration, args.gpu_index
    )

    # 计算指标
    metrics = compute_metrics(results, wall_time, audio_duration, args.concurrency)
    if metrics and gpu_stats:
        metrics["gpu"] = gpu_stats

    # 打印报告
    print_report(metrics)

    # 保存结果
    if metrics:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
