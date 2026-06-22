# SeACo-Paraformer ASR 服务

基于 SeACo-Paraformer 的工业级中文语音识别服务，支持热词定制、动态 Batch 推理、GPU 加速。

## 环境准备

### 系统要求

- Python ≥ 3.12
- CUDA 12.1 + cuDNN 9（v1，ORT）
- CUDA 12.6（v2，TensorRT 10.6）
- onnxruntime-gpu == 1.19.2（v1）
- TensorRT 10.6（v2）
- Docker + Docker Compose

### 安装依赖

```bash
# v1 推理环境（ORT）
pip install -r requirements-infer.txt

# v2 推理环境（TensorRT）
pip install -r requirements-infer-trt.txt

# 模型转换环境（仅导出 ONNX 时需要）
pip install -r requirements-convert.txt
```

---

## v2 推理路径（推荐：纯 fp16）

### 总体技术路径

通过 3 个独立技术点叠加，实现纯 fp16 推理（无任何 fp32 fallback）：

| # | 技术点 | 实现位置 | 作用 |
|---|---|---|---|
| 1 | **opset 17 LayerNormalization 单节点** | `scripts/export_onnx_split.py` 默认 `--opset 17` | PyTorch 2.5 trace 自动识别 `nn.LayerNorm`，导出为单节点；TRT 10.6 内部对该节点自动 fp32 累加，避免 fp16 LayerNorm 内 `(x-mean)²` 溢出 |
| 2 | **encoder 残差 Add 后 clamp 30000** | `seaco_paraformer/encoder.py` 的 `EncoderLayerSANM(clamp_value=30000)` | 30000 ≫ PT 真实激活峰值 ~7554（不影响 PT 数学），≪ fp16 上限 65504（fp16 残差 Add 不溢出 inf） |
| 3 | **纯 trtexec --fp16 转换** | `scripts/convert_trt.py --precision fp16` | 不需要 Python TRT API、不需要 OBEY_PRECISION_CONSTRAINTS、不需要任何手动 fp32 fallback |

### 完整执行流程

```bash
# Step 1：PT baseline 验证（转换容器内）
python tests/test_pt_inference_v2.py \
    --audio test_data/audio_16000_10s.wav \
    --hotwords 埃文 账号

# Step 2：导出分段 ONNX（注入 clamp=30000）
python scripts/export_onnx_split.py \
    --output-dir ./models/asr/split \
    --clamp-value 30000

# Step 3：ORT 验证 ONNX 等价性
python tests/test_split_onnx_pipeline.py \
    --audio test_data/audio_16000_10s.wav \
    --device cuda \
    --hotwords 埃文 账号

# Step 4：清理旧 engine（可选）
rm -f models/asr/trt/*.engine

# Step 5：转 fp32 baseline（可选，用于性能对比）
python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp32 --profile encoder
python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp32 --profile cif
python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp32 --profile decoder
python scripts/convert_trt.py --input ./models/asr/split/bias_encoder.onnx --precision fp32 --profile bias

# Step 6：转 fp16（生产方案）
python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp16 --profile encoder
python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp16 --profile cif
python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp16 --profile decoder
python scripts/convert_trt.py --input ./models/asr/split/bias_encoder.onnx --precision fp16 --profile bias

# Step 7：端到端验证
python tests/test_trt_pipeline.py \
    --audio test_data/audio_16000_10s.wav \
    --encoder-precision fp16 --cif-precision fp16 \
    --decoder-precision fp16 --bias-precision fp16 \
    --hotwords 埃文 账号
```

### 精度验证标准（2080 Ti，10s 音频）

| 指标 | 期望值 | 含义 |
|---|---|---|
| token_num | 与 PT baseline 一致（=61） | CIF predictor 数值稳定 |
| encoder max | ~0.37（PT baseline 0.3716） | 数值量级一致 |
| nan/inf | False | 无溢出 |
| 识别文本 | 字符级与 PT baseline 一致 | 推理完整等价 |
| RTX | ~96-100x | 性能符合预期（2080 Ti） |

### 备选方案（追求最高速度）

```bash
# encoder fp16 + 其余 fp32 → RTX 124-157x
python tests/test_trt_pipeline.py \
    --audio test_data/audio_16000_10s.wav \
    --encoder-precision fp16 \
    --cif-precision fp32 --decoder-precision fp32 --bias-precision fp32 \
    --hotwords 埃文 账号
```

---

## v1 推理路径（ORT）

### 模型准备

```bash
# 整体导出 fp32 ONNX
python scripts/export_onnx.py --output-dir ./models/asr

# fp32 → int8 动态量化（CPU 部署）
python scripts/convert_int8.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/int8

# 下载 VAD 模型
python scripts/download_vad.py --output-dir ./models/vad
```

### 启动服务

```bash
# 本地启动
python -m uvicorn src.main:app --host 0.0.0.0 --port 8080

# Docker 启动（v1）
docker-compose up -d

# Docker 启动（v2，TRT）
docker-compose -f docker-compose.trt.yml up -d
```

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| HOST_PORT | 8099 | 宿主机映射端口 |
| WORKS | 1 | uvicorn workers（GPU 服务必须 1） |
| BATCH | 12 | 最大 batch size（合法值：1,2,4,8,12） |
| BATCH_TIMEOUT | 10 | batch 等待超时（毫秒） |
| LOG_LEVEL | INFO | 日志级别 |
| MAX_CONCURRENT_REQUESTS | 2000 | 最大并发请求数 |
| MODEL_PRECISION | auto | 模型精度（auto / fp32 / int8 / trt_fp16） |

> 容器内部固定端口 8080，通过 HOST_PORT 映射到宿主机。GPU 推理服务 WORKS 必须为 1（单进程），靠 asyncio + 线程池实现并发。

---

## API 示例

### curl

```bash
curl -X POST http://localhost:8099/asr \
    -H "Content-Type: application/json" \
    -d '{
        "b64": "'"$(base64 -w0 test.wav)"'",
        "hotwords": ["张三", "李四"]
    }'
```

### Python

```python
import base64
import requests

with open("test.wav", "rb") as f:
    b64_audio = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8099/asr",
    json={"b64": b64_audio, "hotwords": ["张三", "李四"]},
)
print(response.json())
```

### 健康检查

```bash
curl http://localhost:8099/health
```

---

## 项目结构

```
SeACoParaformer/
├── seaco_paraformer/         # 模型代码框架（独立，不依赖 FunASR 运行时）
│   ├── __init__.py
│   ├── model.py              # SeacoParaformer 主模型
│   ├── encoder.py            # SANMEncoder + EncoderLayerSANM（含 clamp_value 参数）
│   ├── decoder.py            # ParaformerSANMDecoder + DecoderLayerSANM
│   ├── predictor.py          # CifPredictorV3 + cif / cif_v1_export
│   ├── attention.py          # SANM Self-Attention / Cross-Attention
│   ├── layers.py             # LayerNorm / FFN / SinusoidalPositionEncoder
│   ├── utils.py              # MultiSequential / repeat / make_pad_mask
│   └── load_model.py         # ModelScope 下载 + 加载权重
├── src/                      # 推理服务源代码
│   ├── main.py               # FastAPI 入口（三级流水线）
│   ├── config.py
│   ├── errors.py
│   ├── schemas.py
│   ├── logger.py             # 结构化日志
│   ├── feature_extractor.py  # torchaudio kaldi fbank + LFR + CMVN
│   ├── tokenizer.py          # vocab8404 解码
│   ├── vad.py                # Silero VAD ONNX
│   ├── audio_segment.py      # 固定桶边界切分（2s/4s/8s）
│   ├── asr_engine.py         # ORT 推理引擎（v1）
│   ├── trt_engine.py         # TensorRT 推理引擎（v2）
│   └── scheduler.py          # GPU Scheduler（bucket + dynamic batch）
├── scripts/                  # 工具脚本
│   ├── export_onnx.py                # v1 整体 ONNX 导出
│   ├── export_onnx_split.py          # v2 分段 ONNX 导出（含 --clamp-value）
│   ├── export_encoder_truncated.py   # encoder 截断实验（保留供后续优化）
│   ├── export_decoder_truncated.py   # decoder 截断实验（保留供后续优化）
│   ├── convert_trt.py                # ONNX → TRT engine 转换
│   ├── convert_int8.py               # fp32 → int8 动态量化
│   ├── convert_fp16.py               # fp32 → fp16 转换（v1 备选）
│   ├── verify_onnx.py                # ONNX vs PT 精度验证
│   ├── inspect_onnx.py               # ONNX 模型结构检查
│   ├── benchmark.py                  # 性能基准测试
│   ├── download_vad.py               # VAD 模型下载
│   └── entrypoint_trt.sh             # TRT 镜像启动脚本
├── tests/                    # 测试脚本
│   ├── test_pt_inference_v2.py       # PT baseline（独立包推理）
│   ├── test_split_onnx_pipeline.py   # ORT 分段串联推理
│   ├── test_trt_pipeline.py          # TRT 分段推理（各部分独立精度）
│   ├── test_model.py                 # v1 整体 ONNX 推理
│   ├── test_service.py               # v1 服务压测
│   ├── test_single.py                # 单次请求测试
│   ├── test_asr_api.py               # v1 HTTP API 测试
│   └── test_vad.py                   # VAD 单独测试
├── models/                   # 模型文件（不纳入版本控制）
│   ├── asr/
│   │   ├── am.mvn / tokens.json / config.yaml
│   │   ├── fp32/             # v1 整体 ONNX fp32
│   │   ├── int8/             # v1 整体 ONNX int8
│   │   ├── split/            # v2 分段 ONNX（encoder/cif/decoder/bias_encoder）
│   │   └── trt/              # v2 TRT engine（GPU 绑定）
│   └── vad/silero_vad.onnx
├── docs/
│   ├── README.md             # 本文件
│   ├── API.md                # API schema
│   ├── DEPLOY.md             # 部署文档
│   └── CONTRIBUTING.md       # 贡献指南
├── logs/                     # 服务日志（按天轮转，保留 7 天）
├── Dockerfile                # v1 推理镜像（ORT + CUDA 12.1）
├── Dockerfile.trt            # v2 推理镜像（TRT 10.6 + CUDA 12.6）
├── docker-compose.yml        # v1 服务编排
├── docker-compose.trt.yml    # v2 服务编排（engine 缓存 volume）
├── step_run.sh               # 完整转换流程示例
├── requirements-infer.txt    # v1 推理依赖
├── requirements-infer-trt.txt # v2 TRT 推理依赖
└── requirements-convert.txt  # 模型转换依赖
```

---

## 服务架构

三级流水线并行架构，多请求间各级独立并行：

```
请求 → [Stage 1: CPU 线程池]    → [Stage 2: CPU 线程池]   → [Stage 3: GPU Scheduler] → 返回
        音频解码 + VAD + 切段       特征提取（torchaudio）   batch 推理（ORT/TRT）
        （多请求并行）              （多请求并行）           （跨请求合并 batch）
```

设计原则：
- CPU 与 GPU 同时满载（VAD/特征提取占 CPU，ASR 占 GPU）
- 请求间互不阻塞（流水线各级独立）
- 单请求延迟 ≈ max(Stage1, Stage2, Stage3)
- 吞吐量随并发线性增长直至 GPU 饱和

### GPU Scheduler 调度策略

- VAD 后段经合并/切分处理，强制归入固定桶：2s / 4s / 8s（LFR 帧数 34 / 67 / 134）
- `audio_segment.py` 按桶边界合并和切分 VAD 段（最小段 ≥ 2s，最大段 ≤ 8s）
- Scheduler 将 chunk 特征 pad 到桶边界
- 在 `BATCH_TIMEOUT` 窗口内持续收集同桶 chunk
- 达到合法 batch size（1, 2, 4, 8, 12）立即触发推理
- 超时后按实际数量 pad 到最近合法 batch size 推理
- OOM Fallback：减半 batch 重试 → 逐条推理 → 返回 `ASR_INFER_FAILED` 错误
