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

# Step 1: Patch FunASR 源码（消除 CIF Loop 算子，使 fp16 可用）
cp /usr/local/lib/python3.12/dist-packages/funasr/models/bicif_paraformer/cif_predictor.py /tmp/cif_predictor_backup.py

sed -i '1a from funasr.models.paraformer.cif_predictor import cif_v1_export as _cif_v1_export, cif_wo_hidden_v1 as _cif_wo_hidden_v1' \
  /usr/local/lib/python3.12/dist-packages/funasr/models/bicif_paraformer/cif_predictor.py

sed -i 's/acoustic_embeds, cif_peak = cif_export(hidden, alphas, self.threshold)/acoustic_embeds, cif_peak = _cif_v1_export(hidden, alphas, self.threshold)/' \
  /usr/local/lib/python3.12/dist-packages/funasr/models/bicif_paraformer/cif_predictor.py

sed -i 's/us_cif_peak = cif_wo_hidden_export(us_alphas, self.threshold - 1e-4)/us_cif_peak = _cif_wo_hidden_v1(us_alphas, self.threshold - 1e-4)/' \
  /usr/local/lib/python3.12/dist-packages/funasr/models/bicif_paraformer/cif_predictor.py

# Step 2: 导出模型（fp32 + fp16）
python scripts/export_onnx.py --output-dir ./models/asr

# Step 3: 下载 VAD 模型
python scripts/download_vad.py --output-dir ./models/vad

# 完成后退出容器
exit
```

> 说明：sed patch 将 CIF predictor 中带 for 循环的 `cif_export`（导出为 ONNX Loop 算子）替换为向量化的 `cif_v1_export`（使用 cumsum，无 Loop），使 fp16 转换不再出现 Sequence 类型冲突。

导出产物：
```
models/asr/fp32/model.onnx      # ASR 主模型 fp32（当前线上部署）
models/asr/fp32/model_eb.onnx   # 热词 bias encoder fp32
models/asr/fp16/model.onnx      # fp16 模型（待修复）
models/asr/fp16/model_eb.onnx   # fp16 bias encoder（待修复）
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

# 2. 自定义 op_block_list 转 fp16
python scripts/export_onnx.py \
  --output-dir ./models/asr \
  --op-block-list LayerNormalization Softmax ReduceMean

# 3. 验证
python scripts/verify_onnx.py --audio test.wav
```

### 调整精度敏感算子

默认 op_block_list 为：`Range`

仅 Range 算子需要保留 fp32（不支持 fp16 输入），其余算子均可安全转换。

---

## VAD 模型更新

```bash
python scripts/download_vad.py --output-dir ./models/vad
```

Silero VAD 更新频率较低，通常无需频繁更新。

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
| scripts/ | 工具脚本（导出、下载、验证） |
| models/ | 模型文件（不纳入 Git） |
| configs/ | 配置文件 |
| docs/ | 文档 |
| logs/ | 运行日志（按天轮转） |
| tests/ | 测试代码 |
