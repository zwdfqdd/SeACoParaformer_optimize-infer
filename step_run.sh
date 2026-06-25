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


# ============================================================
# v2 阶段 2 — INT8 量化（QDQ Explicit Quantization）
# ============================================================
# 量化对象：encoder + decoder（cif/bias 保持 fp16）
# 校准数据：calib_data/audio_data 目录下 16kHz 单声道 WAV（300 条）
# 量化方法：QDQ Explicit（nvidia-modelopt 插入 QuantizeLinear/DequantizeLinear 节点）
#
# 关键背景：
#   - 方案 2（IInt8EntropyCalibrator2 Implicit）在 SeACo 架构上无效！
#     encoder 主干 MatMul 被 TRT myelin 融合进带 LayerNorm 的大 kernel，
#     融合 kernel 不支持部分 INT8 → 全部 fall back fp16 → engine 体积不降（实测 INT8 层=0）
#   - 方案 1（QDQ Explicit）有效：
#     Q/DQ 节点显式标记量化边界，TRT 不敢融合掉 → INT8 真正生效
#     encoder 337MB → 187MB，decoder 159MB → 112MB
#
# 环境依赖（仅 INT8 导出需要，转换容器内）：
#   pip install nvidia-modelopt==0.21.0 torchprofile --extra-index-url https://pypi.nvidia.com
#   注意：不要装 nvidia-modelopt[torch]！0.44+ 会把 torch 顶到 2.12+cu130 破坏环境

step 4:
    # ─── 步骤 1：encoder QDQ 量化 ─────────────────────────────────
    # modelopt INT8 量化 + 校准 → 导出含 QDQ 节点的 ONNX
    python scripts/export_encoder_qdq.py \
        --calib-data ./calib_data/audio_data \
        --output ./models/asr/split/encoder_qdq.onnx

    # QDQ ONNX → INT8 engine（QDQ 自带 scale，不需要 calibrator）
    python scripts/convert_trt.py \
        --input ./models/asr/split/encoder_qdq.onnx \
        --precision int8 --profile encoder \
        --output ./models/asr/trt/2080_ti_encoder_int8_qdq.engine

    # 验证：encoder int8 + 其余 fp16（应识别正确，CER≈0）
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav \
        --encoder-precision int8 \
        --cif-precision fp16 --decoder-precision fp16 --bias-precision fp16 \
        --encoder-engine ./models/asr/trt/2080_ti_encoder_int8_qdq.engine \
        --hotwords 埃文 账号

    # ─── 步骤 2：decoder QDQ 量化（encoder 通过后） ─────────────────
    # decoder 校准需用 fp16 encoder+cif 跑出中间结果（acoustic_embeds/encoder_out）
    # 默认排除 SeACo 热词路径（seaco_decoder/hotword_output_layer）保持 fp16，
    # 否则 INT8 会破坏热词修正（实测全量化时"埃文"→"艾文"）
    python scripts/export_decoder_qdq.py \
        --encoder-engine ./models/asr/trt/2080_ti_encoder_fp16.engine \
        --cif-engine ./models/asr/trt/2080_ti_cif_fp16.engine \
        --output ./models/asr/split/decoder_qdq.onnx

    python scripts/convert_trt.py \
        --input ./models/asr/split/decoder_qdq.onnx \
        --precision int8 --profile decoder \
        --output ./models/asr/trt/2080_ti_decoder_int8_qdq.engine

    # 验证：encoder + decoder 都 int8（cif/bias 保持 fp16）
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav \
        --encoder-precision int8 --decoder-precision int8 \
        --cif-precision fp16 --bias-precision fp16 \
        --encoder-engine ./models/asr/trt/2080_ti_encoder_int8_qdq.engine \
        --decoder-engine ./models/asr/trt/2080_ti_decoder_int8_qdq.engine \
        --hotwords 埃文 账号

    # ─── 步骤 3：数据集级 CER 评测 ─────────────────────────────────
    # 基准 fp16（与 PT baseline 一致）vs 待测 int8，阈值 3%
    # 注：当前无真实标注，以 fp16 输出作为参考基准（偏乐观，待后续用标注测试集复核）
    python scripts/evaluate_cer.py \
        --audio-dir calib_data/audio_data \
        --threshold 0.03 \
        --csv report_cer.csv

    # 含热词评测
    python scripts/evaluate_cer.py \
        --audio-dir calib_data/audio_data \
        --hotwords 埃文 账号 \
        --threshold 0.03

step 5:
    # ─── 全 INT8（trt_int8）：补充 cif + bias QDQ ──────────────────
    # 仅在需要 trt_int8（4 段全 int8）时执行；线上推荐 trt_int8_enc 无需此步。
    # cif/bias int8 精度需实测，未达标可回退 fp16（即 trt_int8_enc）。

    # cif QDQ（需 fp16 encoder engine 生成校准输入；cumsum 路径天然不量化）
    python scripts/export_cif_qdq.py \
        --calib-data ./calib_data/audio_data \
        --encoder-engine ./models/asr/trt/2080_ti_encoder_fp16.engine \
        --output ./models/asr/split/cif_qdq.onnx
    python scripts/convert_trt.py \
        --input ./models/asr/split/cif_qdq.onnx \
        --precision int8 --profile cif \
        --output ./models/asr/trt/2080_ti_cif_int8_qdq.engine

    # bias QDQ（自包含，用词表编码 token 校准）
    python scripts/export_bias_qdq.py \
        --hotword-file ./models/asr/hotwords.txt \
        --output ./models/asr/split/bias_encoder_qdq.onnx
    python scripts/convert_trt.py \
        --input ./models/asr/split/bias_encoder_qdq.onnx \
        --precision int8 --profile bias \
        --output ./models/asr/trt/2080_ti_bias_encoder_int8_qdq.engine

    # 验证：4 段全 int8
    python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav \
        --encoder-precision int8 --cif-precision int8 \
        --decoder-precision int8 --bias-precision int8 \
        --encoder-engine ./models/asr/trt/2080_ti_encoder_int8_qdq.engine \
        --cif-engine ./models/asr/trt/2080_ti_cif_int8_qdq.engine \
        --decoder-engine ./models/asr/trt/2080_ti_decoder_int8_qdq.engine \
        --bias-engine ./models/asr/trt/2080_ti_bias_encoder_int8_qdq.engine \
        --hotwords 埃文 账号

# ============================================================
# v2 阶段 2 INT8 成果（2080 Ti）
# ============================================================
# 方案：QDQ Explicit Quantization（nvidia-modelopt 0.21）
# 体积：encoder 337MB→187MB，decoder 159MB→112MB（cif/bias 保持 fp16）
# 精度：encoder int8 单模块 CER≈0（与 baseline 一致）
#       encoder+decoder int8（SeACo 路径 fp16）单条样本 CER≈3%
# 注意：2080 Ti（Turing）小 batch 下 INT8 因 Q/DQ 开销速度无明显提升，
#       INT8 价值在显存占用（减半）和大 batch 吞吐，A10/T4 上预期有速度收益
#
# 待完成任务（TODO）：
#   - 用真实标注测试集复核 CER（当前以 fp16 输出为参考基准，偏乐观）
#   - 若 CER 超标，方向 B：decoder 额外排除 src_attn（cross-attention）保持 fp16
#       python scripts/export_decoder_qdq.py ... \
#           --exclude-patterns seaco_decoder hotword_output_layer src_attn
# ============================================================
