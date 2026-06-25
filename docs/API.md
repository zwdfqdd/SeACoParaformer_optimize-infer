# API 接口文档

## 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /asr | 语音识别 |
| GET | /health | 健康检查 |
| GET | /metrics | Prometheus 指标 |
| POST | /hotwords/reload | 重载默认词表（运行时热更新） |
| GET | /hotwords/status | 查看当前词表版本状态 |
| POST | /hotwords/rollback | 回滚到上一版词表 |

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
| 1008 | HOTWORD_VERSION_CONFLICT | 409 | 词表热更新版本冲突（乐观并发） |

---

## 热词格式说明

热词通过 `hotwords` 字段传入（可选）。服务按**生效词表大小**三路分流，切换点 = 256：

| 条件 | 路径 | 处理方式 |
|---|---|---|
| 客户端传 `hotwords` | 路径 A：SeACo 在线热词 | 截断 Top256 → bias_encoder 编码 → ASF top-50 注入 decoder → 热词增强识别 |
| 不传，默认词表 ≤256 | 路径 A：SeACo（默认表） | 用启动预编码缓存的 bias_embed，零额外成本 |
| 不传，默认词表 >256 | 路径 B：Faiss 后处理纠错 | 普通 ASR → 拼音滑窗检索 → 三重联合判定 → 纠错 |

### 路径 A：SeACo 在线热词

- 每个热词为一个独立字符串（人名、品牌名、术语等）
- **数量上限 256**：超过 256 个时截断保留前 256 个（Top256），并输出告警日志
- 显存上界恒定：热词维度固定 ≤256，TRT engine profile 无需重建

**示例：**

```json
{
  "hotwords": ["SeACo", "Paraformer", "达摩院", "张三"]
}
```

### 路径 B：默认大词库纠错（未传热词且默认词表 >256）

- 服务端维护大词库（`models/asr/hotwords.txt`，规模 1 万~20 万）
- 普通 ASR 输出后，按拼音相似度 + 编辑距离做后处理纠错
- 与 GPU 显存解耦（Faiss CPU 索引），词库规模扩大不增加显存
- 三重联合判定才替换，最大限度防止误纠

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

---

## 词表热更新接口

服务端默认词表（`models/asr/hotwords.txt`）支持运行时热更新，不中断推理。
单容器多 worker 部署下，更新经原子写 + 版本号，其他 worker 后台轮询收敛（最终一致，延迟 ≤ `HOTWORD_POLL_INTERVAL`，默认 5s）。

### POST /hotwords/reload — 重载词表

**请求（方式一：直接传新词表内容）：**

```json
{
  "words": ["阿里巴巴", "达摩院", "通义千问"],
  "expected_version": 3
}
```

**请求（方式二：重读磁盘文件）：**

```json
{
  "reload_from_file": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| words | string[] | 否 | 新词表内容（与 reload_from_file 二选一） |
| reload_from_file | bool | 否 | true 表示重读 `models/asr/hotwords.txt` |
| expected_version | int | 否 | 乐观并发版本号，与当前不符则拒绝（防并发覆盖） |

**成功响应（HTTP 200）：**

```json
{
  "code": 0,
  "version": 4,
  "md5": "a1b2c3...",
  "count": 3,
  "route": "A",
  "dropped_oov": ["xxx"],
  "message": "词表更新成功，已切换至 version 4"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| version | int | 更新后的词表版本号 |
| md5 | string | 词表内容哈希 |
| count | int | 有效词条数（去重 + 剔除 OOV 后） |
| route | string | 生效路径，A（SeACo，≤256）或 B（Faiss，>256） |
| dropped_oov | string[] | 被剔除的无法编码词 |

**校验失败响应（HTTP 400）：**

```json
{
  "code": 1000,
  "error": "INPUT_PARAM_FAILED",
  "message": "词表校验失败：有效词条为 0"
}
```

**版本冲突响应（HTTP 409）：**

```json
{
  "code": 1008,
  "error": "HOTWORD_VERSION_CONFLICT",
  "message": "词表已被其他实例更新至 version 5，请基于最新版本重试"
}
```

### GET /hotwords/status — 词表状态

```json
{
  "version": 4,
  "md5": "a1b2c3...",
  "count": 3,
  "route": "A",
  "loaded_at": "2026-06-25T10:30:00",
  "cache_ready": true
}
```

### POST /hotwords/rollback — 回滚

回滚到上一版本词表内容。注意：回滚是**以上一版内容发布一个新版本**（version 递增），
而非把版本号退回。这样保证 version 单调递增，其他 worker 才能通过轮询感知并收敛。

例如当前 version=4，回滚后 version=5（内容 = version=3 的内容）。

```json
{
  "code": 0,
  "version": 5,
  "message": "已回滚至上一版内容（发布为 version 5）"
}
```
