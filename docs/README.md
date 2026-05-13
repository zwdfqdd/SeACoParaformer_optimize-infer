# SeACo-Paraformer ASR 服务

基于 SeACo-Paraformer 的工业级中文语音识别服务，支持热词定制、动态 Batch 推理、GPU 加速。

## 环境准备

### 系统要求

- Python >= 3.12
- CUDA 12.1（GPU 推理）
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
│   ├── fp16/
│   │   ├── model.onnx       # ASR 主模型（fp16）
│   │   └── model_eb.onnx    # 热词 bias encoder（fp16）
│   ├── fp32/
│   │   ├── model.onnx       # ASR 主模型（fp32，验证用）
│   │   └── model_eb.onnx    # bias encoder（fp32）
│   ├── am.mvn               # CMVN 归一化参数
│   └── tokens.json          # 词表文件
└── vad/
    └── silero_vad.onnx      # VAD 模型
```

## 启动服务

### 本地启动

```bash
uvicorn src.main:app --host 0.0.0.0 --port 30960 --workers 1
```

### Docker 启动

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f seaco-asr
```

### 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| WORKS | 1 | uvicorn workers 数量 |
| BATCH | 1 | 每个 worker 最大 batch size |
| PORT | 30960 | 服务端口 |
| MODEL_DIR | ./models | 模型文件目录 |
| DEVICE | auto | 推理设备（auto/cuda/cpu） |
| BATCH_TIMEOUT | 10 | batch 等待超时（毫秒） |
| LOG_LEVEL | INFO | 日志级别 |
| MAX_BATCH_DURATION | 30 | 单次 batch 最大总音频时长（秒） |
| MAX_CONCURRENT_REQUESTS | 2000 | 最大并发请求数 |

## API 示例

### curl

```bash
# 识别音频
curl -X POST http://localhost:30960/asr \
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

# 发送请求
response = requests.post(
    "http://localhost:30960/asr",
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
curl http://localhost:30960/health
# {"status":"ok","device":"cuda","models_loaded":true}
```

## 精度验证

在转换容器内执行：

```bash
python scripts/verify_onnx.py --audio test.wav --onnx-dir ./models/asr/fp16
```

验证逻辑：
- PT 推理：FunASR AutoModel.generate()（原始 PyTorch 模型）
- ONNX 推理：onnxruntime + 内联自实现特征提取 + tokenizer（模拟线上部署路径）
- 脚本自包含，不依赖 src/ 目录
- 对比两者识别文本的 CER（字符错误率）

## 项目结构

```
SeACoParaformer/
├── src/                    # 服务源代码
│   ├── main.py             # FastAPI 入口
│   ├── config.py           # 配置管理
│   ├── errors.py           # 错误码定义
│   ├── schemas.py          # 请求/响应模型
│   ├── logger.py           # 结构化日志
│   ├── feature_extractor.py # 特征提取（自实现）
│   ├── tokenizer.py        # Token 解码（自实现）
│   ├── vad.py              # VAD 语音检测
│   ├── audio_segment.py    # 音频切段
│   ├── asr_engine.py       # ONNX 推理引擎
│   └── scheduler.py        # GPU 调度器
├── scripts/                # 工具脚本
├── models/                 # 模型文件
├── configs/                # 配置文件
├── docs/                   # 文档
├── logs/                   # 日志（按天轮转，保留7天）
├── Dockerfile              # 多阶段构建
├── docker-compose.yml      # 服务编排
└── requirements-*.txt      # 依赖文件
```
