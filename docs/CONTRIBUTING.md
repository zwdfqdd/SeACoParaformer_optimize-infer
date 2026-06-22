# 贡献指南

## 模型更新流程

当需要更新 ASR 模型版本时：

### 1. 模型代码框架

项目内置了完整的 SeACo-Paraformer 模型定义（`seaco_paraformer/` 目录），不依赖 FunASR：

```
seaco_paraformer/
├── __init__.py          # 包入口
├── model.py             # SeACoParaformer 主模型
├── encoder.py           # SANMEncoder (50层 Pre-Norm)
├── decoder.py           # ParaformerSANMDecoder (16+1层)
├── predictor.py         # CifPredictorV3 + cif_v1_export
├── attention.py         # MultiHeadedAttentionSANM (FSMN+Attention)
├── layers.py            # LayerNorm, FFN, SinusoidalPositionEncoder
├── utils.py             # MultiSequential, repeat, make_pad_mask
└── load_model.py        # 从 ModelScope 加载权重
```

依赖：`torch` + `torchaudio` + `numpy` + `modelscope`（仅下载权重）

### 2. 导出 ONNX 模型

```bash
# 分段导出 encoder/cif/decoder（推荐，用于 TRT）
python scripts/export_onnx_split.py --output-dir ./models/asr/split

# 截断 encoder 导出（验证前 N 层精度）
python scripts/export_encoder_truncated.py --num-layers 40

# 截断 decoder 导出
python scripts/export_decoder_truncated.py --num-layers 12

# encoder clamp 导出（fp16 安全版，残差 Add 后 clamp）
python scripts/export_encoder_clamped.py --num-layers 40 --clamp-start 2

# v1 整体导出（完整模型）
python scripts/export_onnx.py --output-dir ./models/asr
```

### 3. 精度验证

```bash
# 验证截断后精度（对比全模型 vs 截断模型 CER）
python scripts/verify_truncated.py --audio test_data/audio_16000_30s.wav --encoder-layers 40
python scripts/verify_truncated.py --audio test_data/audio_16000_30s.wav --decoder-layers 12

# 分析 encoder fp16 溢出（值域分析）
python scripts/analyze_encoder_value_range.py --onnx models/asr/split/encoder.onnx --audio test_data/audio_16000_10s.wav

# PT 模型推理测试
python tests/test_pt_inference.py --audio test_data/audio_16000_30s.wav --device cuda
```

### 4. TRT Engine 转换

```bash
# Encoder — fp32（fp16 需要 clamp 版本）
python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp32 --profile encoder

# Encoder — fp16（使用 clamped 版本）
python scripts/convert_trt.py --input ./models/asr/split/encoder_40layers_clamped.onnx --precision fp16 --profile encoder

# CIF — fp16
python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp16 --profile cif

# Decoder — fp16
python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp16 --profile decoder
```

### 5. int8 量化（CPU 部署）

```bash
python scripts/convert_int8.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/int8
```

### 6. 下载 VAD 模型

```bash
python scripts/download_vad.py --output-dir ./models/vad
```

---

## v2 TRT 分段模型导出与转换

### 概述

v2 使用 TensorRT 替代 ORT 进行 GPU 推理，将模型拆分为四个子模型独立转换（含热词支持）：

| 子模型 | 功能 | 精度 | 说明 |
|--------|------|------|------|
| encoder.onnx | 语音编码 | fp32 | fp16 精度崩溃，需混合精度优化 |
| cif.onnx | CIF 预测器 | fp16 | 向量化实现（cumsum+bmm），TRT 兼容 |
| decoder.onnx | 解码器+SeACo | fp32 | 含 ASF + SeACo decoder + 热词合并 |
| bias_encoder.onnx | 热词编码器 | fp16 | LSTM 编码热词 token IDs |

### 分段导出流程

```bash
# 在转换容器内执行（需要 funasr==1.3.1）
python scripts/export_onnx_split.py --output-dir ./models/asr/split
```

导出产物：
```
models/asr/split/
├── encoder.onnx        # Encoder（~604MB）
├── cif.onnx            # CIF Predictor（~22MB）
├── decoder.onnx        # Decoder + SeACo（~287MB）
└── bias_encoder.onnx   # 热词编码器（~32MB）
```

### TRT Engine 转换

```bash
# 在 TRT 容器内执行（nvcr.io/nvidia/tensorrt:24.11-py3）

# Encoder — fp32（fp16 精度崩溃）
python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp32 --profile encoder

# CIF — fp16
python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp16 --profile cif

# Decoder+SeACo — fp32（fp16 精度崩溃）
python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp32 --profile decoder

# Bias Encoder — fp16
python scripts/convert_trt.py --input ./models/asr/split/bias_encoder.onnx --precision fp16 --profile bias
```

### 验证分段模型

```bash
# PT Export 模式验证（含热词）
python tests/test_pt_export_inference.py --audio test_data/audio_16000_30s.wav --hotwords 埃文 账号

# ORT 验证
python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_30s.wav --device cuda --hotwords 埃文 账号

# TRT 验证（fp32，含热词）
python tests/test_trt_pipeline.py --audio test_data/audio_16000_30s.wav --precision fp32 --hotwords 埃文 账号
```

### 热词推理流程

```
hotword_ids → bias_encoder → hw_embed → 按长度取最后时间步 → bias_embed (1, H, 512)
                                                                    ↓
speech → encoder → cif → acoustic_embeds + encoder_out + bias_embed → decoder+SeACo → logits
```

SeACo 内部：
1. 主 decoder → logits + hidden
2. ASF（注意力分数过滤）→ top-51 热词
3. SeACo decoder × 2（query=acoustic_embeds / hidden，memory=filtered_hotwords）
4. merged → hotword_output_layer → dha_logits
5. NO_BIAS mask 合并：`logits * mask + dha_logits * (1-mask)`

Engine 产物（按 GPU 命名）：
```
models/asr/trt/
├── 2080_ti_encoder_fp32.engine
├── 2080_ti_cif_fp16.engine
├── 2080_ti_decoder_fp32.engine
└── 2080_ti_bias_encoder_fp16.engine
```

> **注意**：TRT engine 与 GPU 硬件绑定，不同 GPU 需分别构建。

### Encoder 精度分析（混合精度优化）

Encoder 全 fp16 会导致精度崩溃（残差连接溢出 inf），需要逐层分析定位敏感层：

```bash
# 第一轮：全 fp16 分析，定位问题层
python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_10s.wav

# 第二轮：指定问题层 fallback fp32，继续分析
python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_10s.wav \
    --fp32-layers-from report_encoder_precision.json

# 按类别批量 fallback
python scripts/analyze_encoder_precision.py --audio test_data/audio_16000_10s.wav \
    --fp32-pattern "norm" "Softmax"
```

---

## 代码规范

- Python 文件名：英文小写，下划线分隔
- 文档和日志：中文
- 不自动生成测试文件
- 未明确要求创建新文件时，在原文件上修改

## 目录结构

| 目录 | 用途 |
|------|------|
| src/ | 服务源代码 |
| scripts/ | 工具脚本（导出、转换、验证、分析） |
| models/ | 模型文件（不纳入 Git） |
| configs/ | 配置文件 |
| docs/ | 文档 |
| logs/ | 运行日志（按天轮转） |
| tests/ | 测试代码 |
