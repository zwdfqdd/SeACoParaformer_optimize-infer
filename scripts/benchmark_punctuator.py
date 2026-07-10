# -*- coding: utf-8 -*-
"""
ngram-punctuator 中文标点模型 推理速度基准测试

背景（预备给下一阶段：句子级时间戳 sentences[]）：
    ngram-punctuator 是纯 CPU 的 N-gram 统计语言模型（KenLM + Qwen2.5 BPE 分词），
    无 GPU 需求。本脚本单独评估其推理速度，为后续接入「文本加标点分句」选型/定参。

核心算法（见 ngram_punctuator/punctuator.py）：
    punctuate() 实际走 divide_and_conquer（分治），beam_search 在源码中被注释未启用。
    热点在 kenlm.Model.score（困惑度打分）：try_punctuate 对长度 N 序列做
    N × len(punct_ids) 次 score 调用，故 puncts 候选集合大小直接线性影响单层耗时。

速度杠杆（本脚本网格验证）：
    - order：n-gram 阶数（3/4/5/6），越高模型越大、KenLM 查询越慢
    - puncts：候选标点集合，越小内层循环越短（11 全集 → 中文 3 个约 3.6x）
    - ppl_drop_ratio：困惑度下降阈值，越大剪枝越狠、候选越少 → 越快（但可能漏标点）
    - max_puncts：标点数上限，越大分治递归越深 → 越慢（防超长文本爆炸）

用法（在有依赖的环境，如服务器运行）：
    pip install ngram-punctuator     # 或将本仓库 ngram-punctuator 加入 PYTHONPATH
    python scripts/benchmark_punctuator.py                     # 全默认网格
    python scripts/benchmark_punctuator.py --order 3           # 固定 order 只扫其它维度
    python scripts/benchmark_punctuator.py --rounds 5 --warmup 2
    python scripts/benchmark_punctuator.py --quick             # 快速小网格

说明：
    - 首次运行会从 ModelScope 下载 n-gram 模型 + Qwen2.5 词表（联网，有缓存）。
    - 模型加载耗时单独统计，不计入推理测速。
    - 输出各参数组合的 单条延迟（均值/P50/P90）与 吞吐（条/秒、字/秒），
      并按延迟排序给出最优参数组合。
    - pip 发布版方法名为 predict，master 源码为 punctuate，脚本自动兼容（_get_punctuate_fn）。

实测结论（服务器 CPU，3 样本 22/57/106 字，rounds=5）：
    速度杠杆（影响从大到小）：
      1) puncts 候选集合：默认 11 种 → 中文 3 种，延迟 ~50-134ms → ~14-37ms（约 3-4x）。
         因 try_punctuate 内层循环长度 = 候选标点数，线性影响单层 score 调用次数。
      2) ppl_drop_ratio：0.08 → 0.15，同配置延迟约减半（剪枝更狠、候选更少）。
      3) order：3 vs 4 差异很小（13.9 vs 14.9ms），order=3 模型更小、速度微优。
    速度-精度权衡（越快越易漏标点，长文本尤明显）：
      - ppl_drop=0.15（最快 13.9ms/4435 字/秒）：长句漏断明显，仅追求极限吞吐时用。
      - ppl_drop=0.12（19.5ms/3165 字/秒）：★推荐，速度与断句完整度均衡。
      - ppl_drop=0.08（36.9ms/1671 字/秒）：断句最完整，追求质量优先时用（仍远快于 ASR 主链路）。
    结论：即便最保守组合也有 ~1671 字/秒；ASR 单请求输出通常数十~数百字，标点后处理
          延迟仅几~几十毫秒，不会成为三级流水线瓶颈。
    ★生产推荐参数：order=3 + puncts=["，","。","？"] + ppl_drop_ratio=0.12
"""

import argparse
import statistics
import time
from typing import List, Optional

# ── 中文常用标点子集（收窄候选加速）──
PUNCTS_FULL = None                       # None = 用模型默认 11 种全集
PUNCTS_ZH = ["，", "。", "？"]           # 中文最常用 3 种
PUNCTS_ZH_EXT = ["，", "。", "？", "！", "、"]  # 中文扩展 5 种

# ── 测试样本（不含标点的中文 ASR 风格长文本，覆盖不同长度）──
SAMPLES = [
    # 短（~25 字）
    "今天天气很好我们一起去公园散步吧顺便买点水果",
    # 中（~55 字）
    "人工智能技术正在深刻改变我们的生活方式从智能手机到自动驾驶汽车"
    "从医疗诊断到金融风控人工智能的应用已经渗透到各个领域",
    # 长（~110 字）
    "中华文明有着五千年的悠久历史从夏商周到秦汉唐宋元明清每个朝代"
    "都留下了丰富的文化遗产长城故宫兵马俑敦煌莫高窟这些都是中华民族"
    "的宝贵财富值得我们好好保护和传承同时我们也要面向未来不断创新"
    "让古老的文明焕发新的生机与活力",
    # 中英混合（英文术语密集，验证 puncts=中文子集 时英文位置能否正确断句）
    "这个new feature的UI设计需要optimize一下user experience特别是mobile端的"
    "responsive design要考虑cross platform compatibility还有API的integration"
    "问题我们要做AB testing来validate hypothesis",
]


def _get_punctuate_fn(punctuator):
    """兼容不同版本方法名：pip 发布版为 predict，master 源码为 punctuate。"""
    for name in ("punctuate", "predict"):
        fn = getattr(punctuator, name, None)
        if callable(fn):
            return fn, name
    raise AttributeError("Punctuator 既无 punctuate 也无 predict 方法，请检查 ngram-punctuator 版本")


def _percentile(sorted_vals: List[float], p: float) -> float:
    """线性插值分位数（sorted_vals 已升序，p in [0,1]）。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = p * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bench_one(
    punct_fn,
    samples: List[str],
    puncts: Optional[List[str]],
    ppl_drop_ratio: float,
    max_puncts: Optional[int],
    rounds: int,
    warmup: int,
):
    """对单组参数在所有样本上跑 rounds 轮，返回统计 dict。

    punct_fn：已解析的推理函数（punctuate 或 predict，见 _get_punctuate_fn）。
    """
    # 预热（触发 kenlm 缓存 / BPE cache）
    for _ in range(warmup):
        for s in samples:
            punct_fn(
                s, puncts=puncts, ppl_drop_ratio=ppl_drop_ratio, max_puncts=max_puncts
            )

    latencies_ms: List[float] = []      # 每条样本每轮的延迟
    total_chars = 0
    sample_outputs = {}
    for r in range(rounds):
        for s in samples:
            t0 = time.perf_counter()
            out = punct_fn(
                s, puncts=puncts, ppl_drop_ratio=ppl_drop_ratio, max_puncts=max_puncts
            )
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt)
            total_chars += len(s)
            if r == 0:
                sample_outputs[s] = out

    lat_sorted = sorted(latencies_ms)
    total_time_s = sum(latencies_ms) / 1000.0
    n_infer = len(latencies_ms)
    return {
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": _percentile(lat_sorted, 0.50),
        "p90_ms": _percentile(lat_sorted, 0.90),
        "max_ms": lat_sorted[-1],
        "qps": n_infer / total_time_s if total_time_s > 0 else 0.0,
        "chars_per_s": total_chars / total_time_s if total_time_s > 0 else 0.0,
        "outputs": sample_outputs,
    }


def main():
    parser = argparse.ArgumentParser(description="ngram-punctuator 推理速度基准")
    parser.add_argument("--order", type=int, default=None,
                        help="固定 n-gram 阶数（不传则扫 3/4）")
    parser.add_argument("--rounds", type=int, default=3, help="每组参数测速轮数")
    parser.add_argument("--warmup", type=int, default=1, help="预热轮数")
    parser.add_argument("--tokenizer-id", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="BPE 分词器 ModelScope id")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（更小网格）")
    args = parser.parse_args()

    from ngram_punctuator import Punctuator

    # ── 参数网格 ──
    if args.order is not None:
        orders = [args.order]
    else:
        orders = [3] if args.quick else [3, 4]

    puncts_grid = [
        ("默认11种", PUNCTS_FULL),
        ("中文3种", PUNCTS_ZH),
    ]
    if not args.quick:
        puncts_grid.append(("中文5种", PUNCTS_ZH_EXT))

    ppl_grid = [0.08] if args.quick else [0.08, 0.12, 0.15]
    max_puncts_grid = [None]  # 默认 len/4；如需限制可加具体值

    print("=" * 78)
    print("ngram-punctuator 推理速度基准测试")
    print("=" * 78)
    print(f"样本数: {len(SAMPLES)}（长度 {[len(s) for s in SAMPLES]} 字）, "
          f"轮数: {args.rounds}, 预热: {args.warmup}")
    print(f"网格: order={orders} × puncts={[n for n,_ in puncts_grid]} × "
          f"ppl_drop={ppl_grid} × max_puncts={max_puncts_grid}")

    results = []  # (延迟均值, 描述, 统计)
    _last_fn = None  # 保留最后加载的推理函数，供中英混合对比段复用
    for order in orders:
        print(f"\n{'─'*78}\n加载 order={order} 模型（首次会从 ModelScope 下载 + 缓存）...")
        t_load = time.perf_counter()
        punctuator = Punctuator(order=order, tokenizer_id=args.tokenizer_id)
        load_ms = (time.perf_counter() - t_load) * 1000.0
        punct_fn, fn_name = _get_punctuate_fn(punctuator)
        _last_fn = punct_fn
        print(f"模型加载耗时: {load_ms:.0f} ms（推理方法: {fn_name}）")

        for pname, puncts in puncts_grid:
            for ppl in ppl_grid:
                for mp in max_puncts_grid:
                    stat = bench_one(
                        punct_fn, SAMPLES, puncts, ppl, mp,
                        args.rounds, args.warmup,
                    )
                    desc = f"order={order} | 标点={pname} | ppl_drop={ppl} | max_puncts={mp}"
                    results.append((stat["mean_ms"], desc, stat))
                    print(f"\n[{desc}]")
                    print(f"  单条延迟 均值={stat['mean_ms']:.1f}ms  P50={stat['p50_ms']:.1f}ms  "
                          f"P90={stat['p90_ms']:.1f}ms  Max={stat['max_ms']:.1f}ms")
                    print(f"  吞吐 {stat['qps']:.1f} 条/秒  {stat['chars_per_s']:.0f} 字/秒")

    # ── 汇总最优 ──
    results.sort(key=lambda x: x[0])
    print(f"\n{'='*78}\n速度排名（延迟均值升序，越靠前越快）\n{'='*78}")
    for i, (mean_ms, desc, stat) in enumerate(results):
        tag = " ★最快" if i == 0 else ""
        print(f"{i+1:2d}. {mean_ms:7.1f}ms  {stat['chars_per_s']:6.0f}字/秒  | {desc}{tag}")

    # ── 最优组合的标点效果（人工核对精度，避免只快不准）──
    best_mean, best_desc, best_stat = results[0]
    print(f"\n{'='*78}\n最优参数标点效果示例（核对精度，勿只看速度）\n{'='*78}")
    print(f"参数: {best_desc}")
    for src, out in best_stat["outputs"].items():
        print(f"\n  原文（{len(src)}字）: {src}")
        print(f"  加标点  : {out}")

    # ── 中英混合样本：中文3种 vs 默认11种 断句对比 ──
    # 验证「puncts 收窄到中文子集」是否影响英文位置断句：
    #   断句位置由困惑度决定，与 token 语言无关 → 英文位置照样能断；
    #   puncts 只决定「用什么符号」→ 中文3种输出中文标点，默认11种可能出英文半角标点。
    mixed_text = (
        "这个new feature的UI设计需要optimize一下user experience特别是mobile端的"
        "responsive design要考虑cross platform compatibility还有API的integration"
        "问题我们要做AB testing来validate hypothesis"
    )
    if _last_fn is not None:
        print(f"\n{'='*78}\n中英混合断句对比（中文3种 vs 默认11种，ppl_drop=0.12）\n{'='*78}")
        print(f"原文: {mixed_text}")
        out_zh = _last_fn(mixed_text, puncts=PUNCTS_ZH, ppl_drop_ratio=0.12)
        out_full = _last_fn(mixed_text, puncts=PUNCTS_FULL, ppl_drop_ratio=0.12)
        print(f"\n  中文3种 : {out_zh}")
        print(f"  默认11种: {out_full}")
        print("\n  观察点：两者断句「位置」应基本一致（英文术语边界照样能断），"
              "差异主要在标点「符号」中英风格。")

    print(f"\n{'='*78}")
    print("建议：结合速度排名 + 上述标点效果综合选参。")
    print("中文 ASR 场景经验：order=3 + 中文标点子集 + ppl_drop_ratio≈0.12 通常兼顾速度与准确。")
    print("中英混合场景：中文子集不影响英文位置断句，仅统一为中文标点风格（多为优点）；")
    print("             如需英文半角标点风格，用默认11种或自定义含英文标点的 puncts。")
    print("=" * 78)


if __name__ == "__main__":
    main()
