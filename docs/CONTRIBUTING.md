# 贡献指南

## 模型更新流程

当需要更新 ASR 模型版本时：

### 1. 构建转换镜像

```bash
docker build --target converter -t seaco-asr-converter .
```

### 2. 启动转换容器并导出模型

```bash
# 启动交互式容器，挂载本地 models 目录
docker run -it --gpus all \
  -v ./models:/app/models \
  seaco-asr-converter bash

# ===== 在容器内执行 =====

# Step 1: Patch FunASR 源码（消除 CIF Loop 算子）
cp /usr/local/lib/python3.12/dist-packages/funasr/models/bicif_paraformer/cif_predictor.py /tmp/cif_predictor_backup.py

sed -i '1a from funasr.models.paraformer.cif_predictor import cif_v1_export as _cif_v1_export, cif_wo_hidden_v1 as _cif_wo_hidden_v1' \
  /usr/local/lib/python3.12/dist-packages/funasr/models/bicif_paraformer/cif_predictor.py

sed -i 's/acoustic_embeds, cif_peak = cif_export(hidden, alphas, self.threshold)/acoustic_embeds, cif_peak = _cif_v1_export(hidden, alphas, self.threshold)/' \
  /usr/local/lib/python3.12/dist-packages/funasr/models/bicif_paraformer/cif_predictor.py

sed -i 's/us_cif_peak = cif_wo_hidden_export(us_alphas, self.threshold - 1e-4)/us_cif_peak = _cif_wo_hidden_v1(us_alphas, self.threshold - 1e-4)/' \
  /usr/local/lib/python3.12/dist-packages/funasr/models/bicif_paraformer/cif_predictor.py

# Step 2: 导出 fp32 模型
python scripts/export_onnx.py --skip-fp16 --output-dir ./models/asr

# Step 3: int8 量化（CPU 部署用）
python scripts/convert_int8.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/int8

# Step 4: 下载 VAD 模型
python scripts/download_vad.py --output-dir ./models/vad

# 完成后退出容器
exit
```

> 说明：sed patch 将 CIF predictor 中带 for 循环的 `cif_export`（导出为 ONNX Loop 算子）替换为向量化的 `cif_v1_export`（使用 cumsum，无 Loop）。

导出产物：
```
models/asr/fp32/model.onnx      # ASR 主模型 fp32（GPU 线上部署）
models/asr/fp32/model_eb.onnx   # 热词 bias encoder fp32
models/asr/int8/model.onnx      # int8 动态量化（CPU 线上部署）
models/asr/int8/model_eb.onnx   # int8 bias encoder
models/vad/silero_vad.onnx      # VAD 模型
```

### 3. 精度验证（在转换容器内执行）

```bash
python scripts/verify_onnx.py \
  --audio test_data/audio_16000_30s.wav \
  --onnx-dir ./models/asr/fp32
```

验证逻辑：
- PT 推理：FunASR AutoModel.generate()（基准）
- ONNX 推理：onnxruntime + 内联自实现特征提取 + tokenizer（模拟线上部署路径）
- 脚本完全自包含，不依赖 src/ 目录
- onnx-dir 下自动查找 model.onnx、am.mvn、tokens.json
- 对比 CER ≤ 1% 为通过

### 4. 更新 CMVN 和词表

如果模型版本变更导致前端参数变化，需同步更新：
- `models/asr/am.mvn` — CMVN 归一化参数
- `models/asr/tokens.json` — 词表文件

### 5. 重新构建镜像

```bash
docker-compose build
docker-compose up -d
```

---

## ONNX 重导出流程

### 完整流程

```bash
# 1. 导出 fp32
python scripts/export_onnx.py --skip-fp16 --output-dir ./models/asr

# 2. int8 量化（CPU 部署）
python scripts/convert_int8.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/int8

# 3. 验证 fp32（GPU）
python scripts/verify_onnx.py --audio test.wav --onnx-dir ./models/asr/fp32 --device cuda

# 4. 验证 int8（CPU）
python scripts/verify_onnx.py --audio test.wav --onnx-dir ./models/asr/int8 --device cpu
```

### 关于 fp16

fp16 模型在 GPU 原生 fp16 kernel 下 CIF cumsum 精度崩溃（输出乱码），**不可用于 GPU 推理**。
仅在 CPU 推理时可用（ORT 自动 cast 回 fp32），但无实际意义（不如直接用 int8）。
GPU 精度优化留待 v2 使用 TensorRT 选择性量化解决。

---

## VAD 模型更新

```bash
python scripts/download_vad.py --output-dir ./models/vad
```

Silero VAD 更新频率较低，通常无需频繁更新。

---

---

## v2 TRT 分段模型导出与转换

### 概述

v2 使用 TensorRT 替代 ORT 进行 GPU 推理，将模型拆分为三个子模型独立转换：

| 子模型 | 功能 | 精度 | 说明 |
|--------|------|------|------|
| encoder.onnx | 语音编码 | fp32 | fp16 精度崩溃，需混合精度优化 |
| cif.onnx | CIF 预测器 | fp16 | 含 cumsum，fp16 可用 |
| decoder.onnx | 解码器 | fp16 | 含 SANM Conv，fp16 可用 |

### 分段导出流程

```bash
# 在转换容器内执行
docker run -it --gpus all \
  -v ./models:/app/models \
  seaco-asr-converter bash

# 导出分段 ONNX（不含热词）
python scripts/export_onnx_split.py --output-dir ./models/asr/split
```

导出产物：
```
models/asr/split/
├── encoder.onnx    # Encoder（~604MB）
├── cif.onnx        # CIF Predictor（~23MB）
└── decoder.onnx    # Decoder（~254MB）
```

### TRT Engine 转换

```bash
# 在 TRT 容器内执行（nvcr.io/nvidia/tensorrt:24.11-py3）

# Encoder — fp32（fp16 精度不可用，待混合精度优化）
python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp32 --profile encoder

# CIF — fp16
python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp16 --profile cif

# Decoder — fp16
python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp16 --profile decoder
```

Engine 产物（按 GPU 命名）：
```
models/asr/trt/
├── 2080_ti_encoder_fp32.engine
├── 2080_ti_cif_fp16.engine
└── 2080_ti_decoder_fp16.engine
```

> **注意**：TRT engine 与 GPU 硬件绑定，不同 GPU 需分别构建。

### 验证分段模型

```bash
# ORT 验证（分段 ONNX）
python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_10s.wav --device cuda

# TRT 验证（encoder fp32 + cif/decoder fp16）
python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav --precision fp16
```

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
