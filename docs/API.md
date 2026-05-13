# API 接口文档

## 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /asr | 语音识别 |
| GET | /health | 健康检查 |
| GET | /metrics | Prometheus 指标 |

---

## POST /asr — 语音识别

### 请求

**Content-Type:** `application/json`

```json
{
  "b64": "UklGRi4AAABXQVZFZm10IBAAAA...",
  "hotwords": ["张三", "李四"]
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| b64 | string | 是 | WAV 16kHz 单声道音频的 Base64 编码 |
| hotwords | string[] | 否 | 热词列表，提高特定词汇识别率 |

### 成功响应（HTTP 200）

```json
{
  "code": 0,
  "text": "今天天气真好适合出去走走",
  "detail": {
    "0": {
      "text": "今天天气真好",
      "start_ms": 0,
      "end_ms": 5200
    },
    "1": {
      "text": "适合出去走走",
      "start_ms": 5200,
      "end_ms": 9800
    }
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int | 业务状态码，0 表示成功 |
| text | string | 全文拼接结果 |
| detail | object | 分段识别结果，key 为段序号 |
| detail.N.text | string | 该段识别文本 |
| detail.N.start_ms | int | 原始音频中的起始时间（毫秒） |
| detail.N.end_ms | int | 原始音频中的结束时间（毫秒） |

### 失败响应

```json
{
  "code": 1001,
  "error": "DECODE_FAILED",
  "message": "音频解码失败，请确认为16kHz单声道WAV格式"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int | 业务错误码 |
| error | string | 错误码名称 |
| message | string | 错误描述 |

---

## 错误码字典

| 错误码 | 名称 | HTTP Status | 说明 |
|--------|------|-------------|------|
| 0 | SUCCESS | 200 | 成功 |
| 1000 | INPUT_PARAM_FAILED | 400 | 输入参数错误（缺少 b64 字段、格式不合法等） |
| 1001 | DECODE_FAILED | 400 | 音频解码失败（非 WAV、文件损坏等） |
| 1002 | VAD_SEGMENT_ERROR | 500 | VAD 模型推理异常 |
| 1003 | AUDIO_SEGMENT_ERROR | 500 | 切段合并逻辑异常 |
| 1004 | ASR_INFER_FAILED | 500 | ASR 模型推理失败 |
| 1005 | AUDIO_TOO_LONG | 400 | 音频超出最大时长限制 |
| 1006 | MODEL_LOAD_FAILED | 500 | 模型加载失败 |
| 1007 | SERVICE_BUSY | 503 | 服务繁忙，队列满/超负载 |

---

## 热词格式说明

- 热词通过 `hotwords` 字段传入，类型为字符串数组
- 每个热词为一个独立字符串
- 热词用于提高特定词汇（人名、品牌名、术语等）的识别准确率
- 热词数量建议不超过 50 个

**示例：**

```json
{
  "hotwords": ["SeACo", "Paraformer", "达摩院", "张三"]
}
```

---

## GET /health — 健康检查

### 响应（HTTP 200）

```json
{
  "status": "ok",
  "device": "cuda",
  "models_loaded": true
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | 服务状态 |
| device | string | 当前推理设备（cuda/cpu） |
| models_loaded | bool | 模型是否已加载 |

---

## GET /metrics — Prometheus 指标

返回 Prometheus 格式的指标数据，包含：

- `asr_request_total{status="success|error"}` — 请求总数
- `asr_inference_duration_seconds` — 推理耗时直方图
- `gpu_memory_usage_bytes` — GPU 显存使用量
