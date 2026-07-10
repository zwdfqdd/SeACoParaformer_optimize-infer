# -*- coding: utf-8 -*-
"""
CT-Transformer 句子级分句器测试（标点效果 + 推理性能）

直接复用 src/sentence_segmenter.py 的真实实现（CT-Transformer 纯 onnxruntime），
不重复实现推理逻辑，保证测试与线上行为一致。

模型：iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx
      扁平存于 PUNC_MODEL_DIR（默认 models/punc：model_quant.onnx + tokens.json + config.yaml），
      缺失时自动经 scripts/download_punc.py（HTTP 直链）下载。

用法：
    python tests/test_punctuator.py                 # 标点效果 + 测速（默认样本）
    python tests/test_punctuator.py --rounds 20     # 指定测速轮数
    python tests/test_punctuator.py --text "自定义无标点文本"

验证点：
    1. 标点恢复效果（含之前 ngram 失效的长/重复口语文本）
    2. 子句切分（split_sentences）：任何标点都切成独立子句 + 字符区间
    3. 推理延迟（均值/P50/P90，字/秒吞吐）
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

# 允许从项目根导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 强制开启句子级（否则 segmenter.load 直接跳过）
os.environ.setdefault("ENABLE_SENTENCE_TIMESTAMP", "true")

# 测试样本（覆盖短句 / 长文本 / 重复口语——后者是早期 ngram 断句失效的场景）
SAMPLES = {
    "短句": "今天天气很好我们一起去公园散步吧你说这个主意怎么样",
    "重复口语": (
        "埃文有麻烦了埃文凯尔有麻烦了重要的是说三遍从目前来看啊据说是埃文的前小助理"
        "带着账号带着钱跑了不带不你带着账号带着钱跑了不是这到底发生什么了重要的是什"
        "说三遍从目前来看啊据说是埃文的前小助理带着账号儿带的想跑了不是这到底发生什么了"
        "我昨天刚我昨天刚发埃文有麻烦了埃文凯尔有麻烦了重要的是说三遍从目前来看啊"
        "据说是埃文的前小小助理带着账号带着钱跑了带着钱这到底发生什么了我昨天刚发现"
    ),
    "长文本": (
        "水滴筹是大家非常熟悉的网络个人大病求助平台但近期一段网络视频却爆出水滴筹工作人员"
        "诱导了病人家属申请水滴筹你好我是水滴筹的志愿者你这边看一下有没有困难需要帮助的"
        "就是有方法就是生了大病难以负担治疗费用的这些自称水滴筹志愿者的一段对话都让那些"
        "曾经通过水滴筹捐款的爱心人士大感不安"
    ),
}


def main():
    ap = argparse.ArgumentParser(description="CT-Transformer 句子级分句器测试")
    ap.add_argument("--rounds", type=int, default=10, help="每样本测速轮数")
    ap.add_argument("--warmup", type=int, default=3, help="预热轮数")
    ap.add_argument("--text", default=None, help="自定义无标点文本（覆盖默认样本）")
    args = ap.parse_args()

    from src.config import settings
    from src.sentence_segmenter import sentence_segmenter

    print("=" * 70)
    print("CT-Transformer 句子级分句器测试")
    print(f"模型目录: {settings.PUNC_MODEL_DIR}  ONNX: {settings.PUNC_ONNX_NAME}  "
          f"MAX_LEN: {settings.PUNC_MAX_LEN}")
    print("=" * 70)

    t0 = time.perf_counter()
    sentence_segmenter.load()
    print(f"模型加载: {(time.perf_counter() - t0) * 1000:.0f}ms  ready={sentence_segmenter.is_ready}")
    if not sentence_segmenter.is_ready:
        sys.exit("分句器未就绪（模型缺失/加载失败）")

    samples = {"自定义": args.text} if args.text else SAMPLES

    for name, text in samples.items():
        # 预热
        for _ in range(args.warmup):
            sentence_segmenter.split_sentences(text)
        # 测速
        lat = []
        sents = []
        for _ in range(args.rounds):
            t = time.perf_counter()
            sents = sentence_segmenter.split_sentences(text)
            lat.append((time.perf_counter() - t) * 1000)
        avg = statistics.mean(lat)
        p50 = statistics.median(lat)
        p90 = sorted(lat)[min(len(lat) - 1, int(len(lat) * 0.9))]

        print(f"\n【{name}】{len(text)} 字 → {len(sents)} 子句")
        print(f"  延迟 均值={avg:.1f}ms P50={p50:.1f}ms P90={p90:.1f}ms  "
              f"({len(text) / avg * 1000:.0f} 字/秒)")
        # 校验：子句区间连续 + 覆盖全文（去空白逐字对齐）
        n_chars = len([c for c in text if not c.isspace()])
        contiguous = all(sents[i][1][1] == sents[i + 1][1][0] for i in range(len(sents) - 1))
        cover_ok = sents and sents[0][1][0] == 0 and sents[-1][1][1] == n_chars
        print(f"  区间连续={contiguous}  覆盖全文={cover_ok}")
        print("  分句结果:")
        for st, span in sents:
            print(f"    {span} {st}")

    print("\n" + "=" * 70)
    print("完成。子句级：任何标点（，。？、）都切成独立子句，字符区间与字级 words 对齐。")
    print("=" * 70)


if __name__ == "__main__":
    main()
