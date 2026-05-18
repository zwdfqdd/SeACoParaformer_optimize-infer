# SeACo-Paraformer ASR 服务

基于 SeACo-Paraformer 的工业级中文语音识别服务，支持热词定制、动态 Batch 推理、GPU 加速。

## 环境准备

### 系统要求

- Python >= 3.12
- CUDA 12.1 + cuDNN 9（GPU 推理）
- onnxruntime-gpu == 1.19.2
- Docker + Docker Compose

### 安装依赖

```bash
# 推理环境
pip install -r requirements-infer.txt

# 模型转换环境（仅导出 ONNX 时需要）
pip install -r requirements-convert.txt
```

### 模型准备

```bash
# 下载并导出 ASR 模型（需要转换环境）
python scripts/export_onnx.py --output-dir ./models/asr

# 下载 VAD 模型
python scripts/download_vad.py --output-dir ./models/vad
```

导出后目录结构：
```
models/
├── asr/
│   ├── fp32/
│   │   ├── model.onnx       # ASR 主模型（fp32，GPU 线上部署）
│   │   └── model_eb.onnx    # 热词 bias encoder（fp32）
│   ├── int8/
│   │   ├── model.onnx       # ASR 主模型（int8 动态量化，CPU 线上部署）
│   │   └── model_eb.onnx    # bias encoder（int8）
│   ├── am.mvn               # CMVN 归一化参数
│   ├── config.yaml          # 模型配置
│   ├── configuration.json   # 模型元信息
│   ├── seg_dict             # 分词词典
│   └── tokens.json          # 词表文件
└── vad/
    └── silero_vad.onnx      # VAD 模型
```

## 启动服务

### 本地启动

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8080
```

### Docker 启动

```bash
# 构建并启动（宿主机 8099 → 容器 8080）
docker-compose up -d

# 查看日志
docker-compose logs -f seaco-asr

# 自定义宿主机端口
HOST_PORT=9000 docker-compose up -d
```

### 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| HOST_PORT | 8099 | 宿主机映射端口 |
| WORKS | 1 | uvicorn workers（GPU 服务必须为 1） |
| BATCH | 12 | 最大 batch size（合法值：1,2,4,8,12） |
| BATCH_TIMEOUT | 10 | batch 等待超时（毫秒） |
| LOG_LEVEL | INFO | 日志级别 |
| MAX_CONCURRENT_REQUESTS | 2000 | 最大并发请求数 |
| MODEL_PRECISION | auto | 模型精度（auto/fp32/int8） |
| VERBOSE | 0 | 详细日志输出（1=开启） |

> 容器内部固定端口 8080，通过 HOST_PORT 映射到宿主机。
> GPU 推理服务 WORKS 必须为 1（单进程），靠 asyncio + 线程池实现并发。
> MODEL_PRECISION=auto 时：GPU 环境自动选 fp32，CPU 环境优先选 int8（若存在）。

## API 示例

### curl

```bash
# 识别音频（宿主机访问）
curl -X POST http://localhost:8099/asr \
  -H "Content-Type: application/json" \
  -d '{
    "b64": "'$(base64 -w0 test.wav)'",
    "hotwords": ["张三", "李四"]
  }'
```

### Python

```python
import base64
import requests

# 读取音频并编码
with open("test.wav", "rb") as f:
    b64_audio = base64.b64encode(f.read()).decode()

# 发送请求（宿主机端口 8099）
response = requests.post(
    "http://localhost:8099/asr",
    json={
        "b64": b64_audio,
        "hotwords": ["张三", "李四"],
    },
)

result = response.json()
print(f"识别结果: {result['text']}")
for idx, detail in result["detail"].items():
    print(f"  段{idx}: [{detail['start_ms']}ms-{detail['end_ms']}ms] {detail['text']}")
```

### 健康检查

```bash
curl http://localhost:8099/health
# {"status":"ok","device":"cuda","models_loaded":true}
```

## 精度验证

在转换容器内执行：

```bash
python scripts/verify_onnx.py --audio test.wav --onnx-dir ./models/asr/fp32
```

验证逻辑：
- PT 推理：FunASR AutoModel.generate()（原始 PyTorch 模型）
- ONNX 推理：onnxruntime + 内联 torchaudio 特征提取 + 内联 tokenizer（模拟线上部署路径）
- 脚本自包含，不依赖 src/ 目录
- 对比两者识别文本的 CER（字符错误率）

## 项目结构

```
SeACoParaformer/
├── src/                    # 推理服务源代码（线上部署）
│   ├── main.py             # FastAPI 入口（三级流水线架构）
│   ├── config.py           # 配置管理
│   ├── errors.py           # 错误码定义
│   ├── schemas.py          # 请求/响应模型
│   ├── logger.py           # 结构化日志（JSON + 按天轮转）
│   ├── feature_extractor.py # 特征提取（torchaudio kaldi fbank）
│   ├── tokenizer.py        # Token 解码（vocab8404）
│   ├── vad.py              # VAD 语音检测（Silero VAD ONNX）
│   ├── audio_segment.py    # 音频切段（固定桶边界切分 2s/4s/8s）
│   ├── asr_engine.py       # ONNX 推理引擎（双模型：model + model_eb）
│   └── scheduler.py        # GPU Scheduler（bucket 分桶 + dynamic batch）
├── scripts/                # 导出环境工具脚本
│   ├── export_onnx.py      # ONNX 导出（CIF 向量化 + fp16 转换）
│   ├── convert_int8.py     # fp32 → int8 动态量化（CPU 部署用）
│   ├── convert_fp16.py     # fp32 → fp16 混合精度转换
│   ├── download_vad.py     # VAD 模型下载
│   ├── verify_onnx.py      # 精度验证（PT vs ONNX，支持指定设备）
│   └── benchmark.py        # 性能基准测试（VAD/PT/fp32/fp16）
├── tests/                  # 测试脚本
│   ├── test_service.py     # 服务压测（并发/RTF/QPS）
│   ├── test_single.py      # 单次请求测试
│   ├── test_model.py       # 模型直接推理测试
│   └── test_vad.py         # VAD 单独测试
├── models/                 # 模型文件（不纳入版本控制）
│   ├── asr/                # ASR 配置 + 模型
│   │   ├── am.mvn, tokens.json, config.yaml  # 配置文件
│   │   ├── fp32/model.onnx, model_eb.onnx    # fp32 模型（GPU 线上部署）
│   │   └── int8/model.onnx, model_eb.onnx    # int8 模型（CPU 线上部署）
│   └── vad/silero_vad.onnx # VAD 模型
├── docs/                   # 文档
├── logs/                   # 日志（按天轮转，保留7天）
├── Dockerfile              # 多阶段构建（CUDA 12.1 + cuDNN 9）
├── docker-compose.yml      # 服务编排
├── .env                    # 环境变量配置
└── requirements-*.txt      # 依赖文件
```

## 架构说明

三级流水线并行架构，多请求间各级独立并行：

```
请求 → [Stage 1: CPU 线程池]     → [Stage 2: CPU 线程池]   → [Stage 3: GPU Scheduler] → 返回
        音频解码 + VAD + 切段       特征提取(torchaudio)      batch 推理(ORT)
        (多请求并行)                (多请求并行)              (跨请求合并 batch)
```

- CPU 和 GPU 同时满载（VAD/特征提取占 CPU，ASR 占 GPU）
- 请求间不互相阻塞（流水线各级独立）
- 单请求延迟 ≈ max(Stage1, Stage2, Stage3)，而非三者之和
- 吞吐量随并发线性增长直到 GPU 饱和

### GPU Scheduler 调度策略

- VAD 后音频段经合并/切分处理，强制归入固定桶边界：2s / 4s / 8s（LFR 帧数 34 / 67 / 134）
- `audio_segment.py` 按 2s/4s/8s 边界合并和切分 VAD 段（最小段 ≥ 2s，最大段 ≤ 8s）
- Scheduler 将 chunk 特征 pad 到桶边界（≤34 帧 → 34，≤67 帧 → 67，≤134 帧 → 134）
- 在 BATCH_TIMEOUT 窗口内持续收集同桶 chunk
- 达到合法 batch size（1,2,4,8,12）立即触发推理
- 超时后按实际数量 pad 到最近合法 batch size 推理
- OOM Fallback：减半 batch 重试 → 逐条推理 → 返回 ASR_INFER_FAILED 错误
