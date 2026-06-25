# 贡献指南

## 模型更新流程

当需要更新 ASR 模型版本或重新导出 ONNX/TRT engine 时，参考本指南。

### 1. 模型代码框架

项目内置完整的 SeACo-Paraformer 模型定义（`seaco_paraformer/` 目录），**不依赖 FunASR 运行时**：

```
seaco_paraformer/
├── __init__.py          # 包入口
├── model.py             # SeacoParaformer 主模型
├── encoder.py           # SANMEncoder + EncoderLayerSANM（含 clamp_value 参数）
├── decoder.py           # ParaformerSANMDecoder + DecoderLayerSANM
├── predictor.py         # CifPredictorV3 + cif / cif_v1_export（向量化，TRT 兼容）
├── attention.py         # SANM Self-Attention / Cross-Attention（FSMN）
├── layers.py            # LayerNorm / FFN / SinusoidalPositionEncoder
├── utils.py             # MultiSequential / repeat / make_pad_mask
└── load_model.py        # 加载本地 PT 权重（PT_MODEL_DIR）
```

PT 权重提前下载并打包进 `models/asr/pt/`（默认 `PT_MODEL_DIR`），不在运行时下载。

### 2. 启动编排（推荐方式）

线上不手动逐条执行转换命令，统一由 `prepare_model.py` 按 `MODEL_PRECISION`
检查产物，缺失则从本地 PT 权重按依赖链逐级转换：

```bash
# 检查 + 按需构建（容器 entrypoint.sh 自动调用）
python scripts/prepare_model.py --precision trt_int8_enc

# 仅检查不构建
python scripts/prepare_model.py --precision trt_fp16 --check-only
```

依赖链：
```
PT 权重 → 分段 ONNX（export_onnx_split.py）→ TRT fp32/fp16 engine（convert_trt.py）
                                          → QDQ ONNX（export_{encoder,cif,decoder,bias}_qdq.py）→ TRT int8 engine
       → 整体 ONNX（export_onnx_whole.py）→ int8 动态量化（convert_onnx_int8_dynamic.py）
```

### 3. 手动导出 + 转换（开发调试）

完整流程见 `step_run.sh`，关键命令：

```bash
# 分段 ONNX 导出（opset 17 + clamp 30000，纯 fp16 关键）
python scripts/export_onnx_split.py --output-dir ./models/asr/split --clamp-value 30000

# TRT fp16（纯 fp16，无 fp32 fallback）
python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp16 --profile encoder
python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp16 --profile cif
python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp16 --profile decoder
python scripts/convert_trt.py --input ./models/asr/split/bias_encoder.onnx --precision fp16 --profile bias

# 整体 ONNX（v1 ORT 路径）+ int8 动态量化（CPU）
python scripts/export_onnx_whole.py --output-dir ./models/asr --skip-fp16
python scripts/convert_onnx_int8_dynamic.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/int8

# VAD 模型
python scripts/download_vad.py --output-dir ./models/vad
```

### 4. 精度验证

```bash
# PT baseline（独立包推理）
python tests/test_pt_inference_v2.py --audio test_data/audio_16000_10s.wav --hotwords 埃文 账号

# ORT 分段串联验证
python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_10s.wav --device cuda --hotwords 埃文 账号

# TRT 分段验证（各段独立精度）
python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav \
    --encoder-precision fp16 --cif-precision fp16 \
    --decoder-precision fp16 --bias-precision fp16 --hotwords 埃文 账号

# 数据集级 CER 评测（基准 fp16 vs 待测 int8）
python scripts/evaluate_cer.py --audio-dir calib_data/audio_data --csv report_cer.csv
```

---

## v2 TRT 分段模型架构

模型拆分为四个子模型独立转换（含热词支持）：

| 子模型 | 功能 | 推荐精度 | 说明 |
|--------|------|----------|------|
| encoder.onnx | 语音编码 | fp16 | opset 17 + clamp 30000，纯 fp16 不崩溃 |
| cif.onnx | CIF 预测器 | fp16 | 向量化实现（cumsum+bmm），TRT 兼容 |
| decoder.onnx | 解码器+SeACo | fp16 | 含 ASF + SeACo decoder + 热词合并 |
| bias_encoder.onnx | 热词编码器 | fp16 | LSTM 编码热词 token IDs |

> 历史问题：早期 encoder/decoder 全 fp16 会精度崩溃（残差 Add 溢出 inf）。
> 现已通过 **opset 17 原生 LayerNormalization + encoder 残差 Add clamp 30000**
> 实现纯 fp16，无需任何 fp32 fallback（详见 docs/README.md）。

### 纯 fp16 三大关键技术

1. **opset 17 LayerNormalization 单节点**：TRT 10.6 内部对该算子自动 fp32 累加
2. **encoder 残差 Add 后 clamp 30000**：≫ PT 真实峰值 ~7554，≪ fp16 上限 65504
3. **纯 trtexec --fp16**：无需 Python TRT API / OBEY_PRECISION_CONSTRAINTS / 手动 fallback

### INT8 量化（QDQ Explicit）

- 量化库：`nvidia-modelopt==0.21.0`（必须钉版本，0.44+ 破坏 torch 环境）
- 方案：QDQ Explicit（插入 Q/DQ 节点显式标记量化边界），Calibrator Implicit 在 SeACo 架构上无效
- encoder QDQ：`export_encoder_qdq.py`；decoder QDQ：`export_decoder_qdq.py`（默认排除 SeACo 路径保 fp16）
- cif QDQ：`export_cif_qdq.py`（trt_int8 用）；bias QDQ：`export_bias_qdq.py`（trt_int8 用）

### 热词推理流程

```
hotword_ids → bias_encoder → hw_embed → 按长度取最后时间步 → bias_embed (1, H, 512)
                                                                    ↓
speech → encoder → cif → acoustic_embeds + encoder_out + bias_embed → decoder+SeACo → logits
```

SeACo 内部：
1. 主 decoder → logits + hidden
2. ASF（注意力分数过滤）→ top-NFILTER(50) 热词
3. SeACo decoder × 2（query=acoustic_embeds / hidden，memory=filtered_hotwords）
4. merged → hotword_output_layer → dha_logits
5. NO_BIAS mask 合并：`logits * mask + dha_logits * (1-mask)`

Engine 产物（按 GPU 命名）：
```
models/asr/trt/
├── 2080_ti_encoder_fp16.engine
├── 2080_ti_cif_fp16.engine
├── 2080_ti_decoder_fp16.engine
├── 2080_ti_bias_encoder_fp16.engine
└── {gpu}_{module}_int8_qdq.engine   # int8 QDQ 产物
```

> **注意**：TRT engine 与 GPU 硬件绑定，不同 GPU 需分别构建。

### engine 层精度诊断

```bash
# 查看 engine 各层精度分布（判断 INT8 是否真正生效）
python scripts/inspect_engine_precision.py --engine models/asr/trt/2080_ti_encoder_int8_qdq.engine
```

---

## 代码规范

- Python 文件名：英文小写，下划线分隔
- 文档和日志：中文
- 不自动生成测试文件
- 未明确要求创建新文件时，在原文件上修改
- 涉及中文内容的文件编辑只用代码编辑工具（避免 shell 文本替换导致编码损坏）

## 目录结构

| 目录 | 用途 |
|------|------|
| seaco_paraformer/ | 模型代码框架（独立，不依赖 FunASR 运行时） |
| src/ | 服务源代码 |
| scripts/ | 工具脚本（导出、转换、量化、评测、编排） |
| models/ | 模型文件（不纳入 Git） |
| configs/ | 配置文件 |
| docs/ | 文档 |
| logs/ | 运行日志（按天轮转，多 worker 按 PID 分文件） |
| tests/ | 测试代码 |
