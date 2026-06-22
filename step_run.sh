#!/usr/bin/env bash
# ============================================================
# SeACo-Paraformer v2 阶段 1 完整执行流程
# 最终方案：opset 17 + clamp 30000 + 纯 fp16
# ============================================================

step 1:
    # PT baseline 推理验证（在转换容器内）
    python tests/test_pt_inference_v2.py --audio test_data/audio_16000_10s.wav --hotwords 埃文 账号

step 2:
    # PT → ONNX 分段导出
    #   --opset 17：原生 LayerNormalization 单节点（TRT 内部 fp32 累加）
    #   --clamp-value 30000：encoder 残差 Add clamp，防 fp16 上限 65504 溢出
    #     - 阈值 ≫ PT 真实峰值 ~7554（不影响 PT 数学）
    #     - 阈值 ≪ fp16 上限 65504（保证 fp16 不 inf）
    python scripts/export_onnx_split.py --output-dir ./models/asr/split --clamp-value 30000

    # ORT 验证（无热词 + 含热词）
    python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_10s.wav --device cuda
    python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_10s.wav --device cuda --hotwords 埃文 账号

step 3:
    # ONNX → TRT
    rm -f models/asr/trt/*.engine

    # ─── fp32 baseline ──────────────────────────────────────
    python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp32 --profile encoder
    python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp32 --profile cif
    python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp32 --profile decoder
    python scripts/convert_trt.py --input ./models/asr/split/bias_encoder.onnx --precision fp32 --profile bias

    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav --hotwords 埃文 账号

    # ─── 纯 fp16（无任何 fp32 fallback） ─────────────────────
    python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp16 --profile encoder
    python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp16 --profile cif
    python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp16 --profile decoder
    python scripts/convert_trt.py --input ./models/asr/split/bias_encoder.onnx --precision fp16 --profile bias

    # 全 fp16 端到端
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav \
        --encoder-precision fp16 --cif-precision fp16 \
        --decoder-precision fp16 --bias-precision fp16 \
        --hotwords 埃文 账号

    python tests/test_trt_pipeline.py --audio test_data/audio_16000_12s.wav \
        --encoder-precision fp16 --cif-precision fp16 \
        --decoder-precision fp16 --bias-precision fp16 \
        --hotwords 埃文 账号

    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav \
        --encoder-precision fp16 --cif-precision fp16 \
        --decoder-precision fp16 --bias-precision fp16

# ============================================================
# 关键技术点（详见 docs/README.md）
# ============================================================
# 1. opset 17 LayerNormalization 单节点：
#    - PyTorch 2.5 trace 自动识别 nn.LayerNorm，导出为单个 LayerNormalization 节点
#    - TRT 10.6 对该算子内部自动 fp32 累加（mean/var/Pow 都用 fp32 计算）
#    - 不需要任何 fp32 fallback
#
# 2. encoder 残差 Add 后 clamp 30000（在 seaco_paraformer/encoder.py 实现）：
#    - 通过 EncoderLayerSANM(clamp_value=...) 构造参数控制
#    - PT 推理时不传（保持数学等价）
#    - 导出 ONNX 时由 export_onnx_split.py 注入
#
# 3. 纯 trtexec --fp16 转换：
#    - 无需 Python TRT API
#    - 无需 OBEY_PRECISION_CONSTRAINTS
#    - 无需任何手动 fp32 fallback
# ============================================================
