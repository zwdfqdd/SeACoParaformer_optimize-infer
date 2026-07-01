#!/usr/bin/env bash
# ============================================================
# SeACo-Paraformer v2 阶段 1 完整执行流程
# 最终方案：opset 17 + clamp 60000 + 纯 fp16
# ============================================================

step 1:
    # PT baseline 推理验证（在转换容器内）
    python tests/test_pt_inference_v2.py --audio test_data/audio_16000_10s.wav --hotwords 埃文 账号

step 2:
    # PT → ONNX 分段导出
    #   --opset 17：原生 LayerNormalization 单节点（TRT 内部 fp32 累加）
    #   --clamp-value 60000：encoder 残差 Add clamp，防 fp16 上限 65504 溢出
    #     - encoder 后段层残差激活峰值高达 ~48万 >> fp16 上限 65504
    #     - 60000 贴近上限最大化保留信息（clamp=30000 裁剪过狠致截断输入解码错乱，已弃用）
    python scripts/export_onnx_split.py --output-dir ./models/asr/split --clamp-value 60000

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
# 2. encoder 残差 Add 后 clamp 60000（在 seaco_paraformer/encoder.py 实现）：
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
#   pip install nvidia-modelopt==0.21.0 torchprofile pulp regex --extra-index-url https://pypi.nvidia.com
#   注意：不要装 nvidia-modelopt[torch]！0.44+ 会把 torch 顶到 2.12+cu130 破坏环境
#   pulp/regex：modelopt 0.21 隐性依赖，缺失会报误导性的 "Please install optional [torch] dependencies"
#   验证安装：python -c "import modelopt.torch.quantization as mtq; print('modelopt OK')"

step 4:
    # ─── 步骤 1：encoder QDQ 量化 ─────────────────────────────────
    # modelopt INT8 量化 + 校准 → 导出含 QDQ 节点的 ONNX
    # --model-id 指向本地 PT 目录（避免联网下载，转换/推理容器均无 modelscope）
    # --cmvn-path 指向 models/asr/pt/am.mvn（配置文件实际位置）
    python scripts/export_encoder_qdq.py \
        --calib-data ./calib_data/audio_data \
        --model-id ./models/asr/pt \
        --cmvn-path ./models/asr/pt/am.mvn \
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
        --model-id ./models/asr/pt \
        --cmvn-path ./models/asr/pt/am.mvn \
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
    # 实测：4 段 engine 可正常运行，但精度损失较大（cif cumsum + bias LSTM 量化），
    #       不推荐线上，仅显存极度紧张且可接受精度下降时使用。

    # cif QDQ（需 fp16 encoder engine 生成校准输入；cumsum 路径天然不量化）
    python scripts/export_cif_qdq.py \
        --calib-data ./calib_data/audio_data \
        --encoder-engine ./models/asr/trt/2080_ti_encoder_fp16.engine \
        --model-id ./models/asr/pt \
        --cmvn-path ./models/asr/pt/am.mvn \
        --output ./models/asr/split/cif_qdq.onnx
    python scripts/convert_trt.py \
        --input ./models/asr/split/cif_qdq.onnx \
        --precision int8 --profile cif \
        --output ./models/asr/trt/2080_ti_cif_int8_qdq.engine

    # bias QDQ（自包含，用词表编码 token 校准）
    python scripts/export_bias_qdq.py \
        --hotword-file ./models/asr/hotwords.txt \
        --model-id ./models/asr/pt \
        --tokens-path ./models/asr/pt/tokens.json \
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


# ============================================================
# v1 ORT 整体模型路径（onnx_fp32 / onnx_int8）导出 + 全功能测试
# ============================================================
# 与上面 TRT 分段路径独立。整体模型 = encoder + 向量化 predictor + decoder + SeACo，
# 单文件 model.onnx（3 输入：speech/speech_lengths/bias_embed）+ model_eb.onnx（热词编码）。
# 关键：predictor 用向量化 CIF（无 Loop），支持 batch>1（多 chunk 合批）。

step 6:
    # ─── 整体 ONNX 导出（fp32）+ int8 动态量化 ───────────────────
    # 产物：models/asr/fp32/{model.onnx, model_eb.onnx}
    python scripts/export_onnx_whole.py --skip-fp16 --output-dir ./models/asr

    # fp32 → int8 动态量化（CPU 线上用，遍历 fp32/ 下所有 onnx）
    python scripts/convert_onnx_int8_dynamic.py \
        --input-dir ./models/asr/fp32 --output-dir ./models/asr/int8

    # ─── ONNX 等价性验证（PT vs ONNX，CER 对比） ─────────────────
    python scripts/verify_onnx.py --audio test_data/audio_16000_10s.wav \
        --onnx-dir ./models/asr/fp32 --device cuda

    # ─── 模型直推全链路（跳过服务，特征→ONNX→解码） ─────────────
    # fp32（GPU）
    python tests/test_model.py --audio test_data/audio_16000_30s.wav \
        --config-dir ./models/asr/pt \
        --model-dir ./models/asr/fp32 --device cuda
    python tests/test_model.py --audio test_data/audio_16000_10s.wav \
        --model-dir ./models/asr/fp32 --config-dir ./models/asr/pt \
        --device cuda --hotwords 埃文 账号

    # int8（CPU）
    python tests/test_model.py --audio test_data/audio_16000_30s.wav \
        --model-dir ./models/asr/int8 --config-dir ./models/asr/pt --device cpu

step 7:
    # ─── VAD 单独测试（语音段时间戳） ───────────────────────────
    python tests/test_vad.py --audio test_data/audio_16000_30s.wav \
        --vad-model ./models/vad/silero_vad.onnx

step 8:
    # ─── 启动服务（手动在另一终端执行，或后台 nohup） ───────────
    # bash run.sh
    # 健康检查：curl http://localhost:8080/health

    # ─── ASR 接口冒烟（标准库 urllib，无需 requests） ───────────
    # 单 chunk（10s）
    python tests/test_asr_api.py test_data/audio_16000_10s.wav --url http://localhost:8080
    # 含热词
    python tests/test_asr_api.py test_data/audio_16000_10s.wav \
        --url http://localhost:8080 --hotwords 埃文 账号
    # 多 chunk 合批（30s，验证 batch>1 不越界）
    python tests/test_asr_api.py test_data/audio_16000_30s.wav --url http://localhost:8080

    # ─── 热词管理三接口（status/reload/rollback，标准库 urllib） ──
    python tests/test_hotword_api.py --url http://localhost:8080

    # ─── 健康/指标/错误码路径（标准库 urllib） ──────────────────
    python tests/test_error_api.py --url http://localhost:8080

    # ─── 单次请求详细输出（标准库 urllib） ─────────────────────
    python tests/test_single.py --audio test_data/audio_16000_30s.wav --url http://localhost:8080

    # ─── 性能/并发压测（标准库 urllib + 线程池） ────────────────
    # 单请求延迟
    python tests/test_service.py --audio test_data/audio_16000_30s.wav --url http://localhost:8080
    # 并发压测（10 并发，共 50 请求）
    python tests/test_service.py --audio test_data/audio_16000_30s.wav \
        --url http://localhost:8080 --concurrency 10 --total 50

# ============================================================
# 测试脚本依赖说明
# ============================================================
# 无额外依赖（标准库 urllib，推理镜像可直接跑）：
#   tests/test_asr_api.py / test_hotword_api.py / test_error_api.py
#   tests/test_single.py / test_service.py
# 需 onnxruntime + torch + numpy + soundfile（推理镜像已含）：
#   tests/test_model.py / test_vad.py / verify_onnx.py
# 转换环境专用（需 funasr/torch）：
#   tests/test_pt_inference_v2.py / test_split_onnx_pipeline.py / test_trt_pipeline.py
# ============================================================
