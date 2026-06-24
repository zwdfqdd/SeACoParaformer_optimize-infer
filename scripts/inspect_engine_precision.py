"""
TRT engine 层精度诊断

用 EngineInspector 导出每层的实际精度（INT8/FP16/FP32），
统计各精度的层数占比，判断 INT8 量化是否真正生效。

用法：
    python scripts/inspect_engine_precision.py --engine models/asr/trt/2080_ti_encoder_int8.engine
    python scripts/inspect_engine_precision.py --engine models/asr/trt/2080_ti_encoder_int8.engine --dump layers.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    sys.exit("需要 tensorrt")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def main():
    parser = argparse.ArgumentParser(description="TRT engine 层精度诊断")
    parser.add_argument("--engine", required=True, help="engine 文件路径")
    parser.add_argument("--dump", default=None, help="导出完整层信息 JSON")
    parser.add_argument("--show-layers", type=int, default=0,
                        help="打印前 N 层详情（0=不打印）")
    parser.add_argument("--raw", action="store_true",
                        help="打印 inspector 原始输出的前 2000 字符（用于排查格式）")
    args = parser.parse_args()

    if not Path(args.engine).exists():
        sys.exit(f"engine 不存在: {args.engine}")

    runtime = trt.Runtime(TRT_LOGGER)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        sys.exit("engine 反序列化失败")

    inspector = engine.create_engine_inspector()
    # JSON 格式只有层名，DETAILED 格式才含 precision/tactic 等信息
    fmt = trt.LayerInformationFormat.ONELINE
    try:
        info_str = inspector.get_engine_information(fmt)
    except Exception:
        info_str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)

    if args.raw:
        print("=" * 60)
        print("inspector 原始输出（前 3000 字符）：")
        print("=" * 60)
        print(info_str[:3000])
        print("\n... [截断]" if len(info_str) > 3000 else "")
        print("=" * 60)

    precision_counter = Counter()
    layers = []

    # ONELINE 格式：每行一个 layer，形如
    #   "Layer(...): <name>, Tactic: ..., precision: INT8, ..."
    # 按行扫描 precision 关键字
    for line in info_str.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r'[Pp]recision[:\s]+([A-Za-z0-9]+)', line)
        if m:
            prec = m.group(1).upper()
            precision_counter[prec] += 1
            layers.append({"name": line[:80], "precision": prec})

    # 如果 ONELINE 没抓到，降级用 JSON 整体正则
    if not precision_counter:
        found = re.findall(r'"[Pp]recision"\s*:\s*"([^"]+)"', info_str)
        if not found:
            found = re.findall(r'[Pp]recision[:=]\s*([A-Za-z0-9]+)', info_str)
        for p in found:
            precision_counter[p.upper()] += 1

    total = sum(precision_counter.values())
    print("=" * 60)
    print(f"Engine: {args.engine}")
    print(f"大小: {Path(args.engine).stat().st_size / (1024*1024):.1f} MB")
    print(f"总层数: {total}")
    print("=" * 60)
    print(f"{'精度':<12} {'层数':>8} {'占比':>8}")
    print("-" * 60)
    for prec, cnt in precision_counter.most_common():
        pct = cnt / total * 100 if total else 0
        print(f"{prec:<12} {cnt:>8} {pct:>7.1f}%")

    # INT8 判断
    int8_cnt = precision_counter.get("INT8", 0) + precision_counter.get("Int8", 0)
    print("-" * 60)
    if int8_cnt == 0:
        print("⚠ INT8 层数为 0：量化完全未生效（全部 fall back fp16/fp32）")
    elif int8_cnt < total * 0.3:
        print(f"⚠ INT8 层占比偏低（{int8_cnt}/{total}）：大部分层 fall back")
    else:
        print(f"✓ INT8 层占比正常（{int8_cnt}/{total}）：量化生效")

    if args.show_layers > 0:
        print("\n前 {} 层详情：".format(args.show_layers))
        for L in layers[:args.show_layers]:
            print(f"  [{L['precision']:<8}] {L['name']}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(layers if layers else {"raw": info_str}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n完整层信息已导出: {args.dump}")


if __name__ == "__main__":
    main()
