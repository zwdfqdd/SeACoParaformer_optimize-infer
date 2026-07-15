# API 接口文档

## 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /chinese_asr | 中文语音识别 |
| GET | /health | 健康检查 |
| GET | /metrics | Prometheus 指标 |
| POST | /hotwords/reload | 重载默认词表（运行时热更新） |
| GET | /hotwords/status | 查看当前词表版本状态 |
| POST | /hotwords/rollback | 回滚到上一版词表 |

---

## POST /chinese_asr — 中文语音识别

### 请求

**Content-Type:** `application/json`

```json
{
  "base64": "UklGRi4AAABXQVZFZm10IBAAAA...",
  "article_url": "https://cdn.example.com/audio/xxx.wav",
  "hotwords": ["张三", "李四"]
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| base64 | string | **是** | WAV 16kHz 单声道音频的 Base64 编码 |
| article_url | string | 否 | 原始音频文件的 URL；服务端原样透传到响应，用于业务侧追踪对账 |
| hotwords | string[] | 否 | 热词列表，提高特定词汇识别率 |

### 成功响应（HTTP 200）

```json
{
  "code": 0,
  "article_url": "https://cdn.example.com/audio/xxx.wav",
  "istar_asr": "今天天气真好适合出去走走",
  "asr": [
    {
      "idx": 0,
      "slid": "",
      "text": "今天天气真好",
      "speaker": "",
      "timestamp": [0.0, 5.2],
      "words": [
        {"text": "今", "timestamp": [0.12, 0.24]},
        {"text": "天", "timestamp": [0.24, 0.42]},
        {"text": "天", "timestamp": [0.42, 0.6]},
        {"text": "气", "timestamp": [0.6, 0.78]},
        {"text": "真", "timestamp": [0.78, 0.96]},
        {"text": "好", "timestamp": [0.96, 1.2]}
      ]
    },
    {
      "idx": 1,
      "slid": "",
      "text": "适合出去走走",
      "speaker": "",
      "timestamp": [5.2, 9.8],
      "words": [
        {"text": "适", "timestamp": [5.32, 5.5]}
      ]
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int | 业务状态码，0 表示成功 |
| article_url | string \| null | 原样透传请求中的 `article_url`；请求未传时为 `null` |
| istar_asr | string | 全文拼接结果（各段 text 顺序拼接） |
| asr | array | 分段识别结果，按时间顺序排列 |
| asr[].idx | int | 段序号（从 0 起） |
| asr[].slid | string | 语种识别结果（**当前未实现**，固定空字符串） |
| asr[].text | string | 该段识别文本 |
| asr[].speaker | string | 说话人识别结果（**当前未实现**，固定空字符串） |
| asr[].timestamp | [float, float] | [起始秒, 结束秒]，段级时间戳；VAD 开启时源自 VAD 时间轴，关闭时为固定 4s 均匀切段边界 |
| asr[].words | array | 字级时间戳数组（CIF alphas 反推得到，需 CIF engine 输出 alphas） |
| asr[].words[].text | string | 字符（中文单字或英文 BPE subword） |
| asr[].words[].timestamp | [float, float] | [起始秒, 结束秒]，字级时间戳，粒度约 60ms |
| message | string | 提示信息；正常识别为空，VAD 后无有效语音时为 `"音频内容为空"` |

**空音频 / 短音频行为**：
- **VAD 后无有效语音**（纯静音、极短或无人声）：返回 HTTP 200 成功，
  `code=0, istar_asr="", asr=[], message="音频内容为空"`（不再报 500 错误）
- **VAD 有效但整体时长 < 2040ms**（最小桶 34 帧 × 60ms，约 2s）：尾部自动 pad 到 2040ms
  后正常识别（保证 encoder 输入不低于 TRT profile 下界）

**VAD 开关（`ENABLE_VAD`，默认 true）对切段的影响**：
- **开启**（默认）：Silero VAD 检测语音段，按 VAD 时间跨度均匀切段（跳过首尾/段间静音）
- **关闭**（`ENABLE_VAD=false`）：不做 VAD，对整段音频按固定 4s 均匀切段：
  - 整段 < 2s → 单段并 pad 到 2s；
  - 尾段 < 2s → 并入前一段；尾段 ≥ 2s → 独立成段。
  - 段级 `timestamp` 为切段边界（非语音边界），空音频仍返回 `message="音频内容为空"`。

**字级时间戳说明**：
- 由独立 timestamp engine（第 5 段，upsample CIF timestamp head）计算，对齐 FunASR
  官方 `ts_prediction_lfr6_standard`：相邻 fire 中点划界（不重叠）+ 超长截断 + 静音扣除
- 精度约 20ms（upsample 3x），单字时长稳定，无重叠、无超长
- **需要开启 `ENABLE_WORD_TIMESTAMP=true`**（默认 false）：
  - 启用后加载第 5 段 timestamp engine，吞吐下降约 30%（含 blstm 计算）
  - 关闭时 `words: []` 空数组，主链路吞吐不受影响
- ORT 整体模型（onnx_fp32/onnx_int8）不支持字级时间戳，`words: []`
- **与 Faiss 纠错一致**（I3）：路径 B 命中替换时，替换同步映射到 `words`（替换段时间
  区间按纠错词字数等分重切），保证 `words` 拼接 == 段 `text`。若字级与段文本口径无法
  可靠对齐（中英混合含空格等边界），保守保留原 `words`（段 `text` 仍已纠错）。
- **中英混合口径统一**（I5）：`words[].text` 已清理 BPE 连接标记（`@@`）与 sentencepiece
  前缀（`▁`），英文按 subword 各自带时间戳，拼接字面与段 `text` 一致；中文逐字一一对应。

**句子级时间戳说明（asr[] 粒度变为子句）**：
- 开启 `ENABLE_SENTENCE_TIMESTAMP=true`（默认 false）后，`asr[]` 每项粒度由 VAD 切段
  变为**一子句**（CT-Transformer 逐 token 恢复标点，任何标点 `，。？、` 都切成独立子句）：
  - `asr[].text`：带标点的子句
  - `asr[].timestamp`：子句起止秒（由句内字级时间戳定位首字 start / 末字 end）
  - `asr[].words`：该子句字级时间戳
  - `istar_asr`：各子句带标点顺序拼接
- **强依赖 `ENABLE_WORD_TIMESTAMP=true`**：子句时间边界由字级时间戳定位；若未开字级
  时间戳，句子级自动降级回段级输出（启动日志告警）。
- 标点由 CT-Transformer 标点模型（纯 onnxruntime，逐 token 分类）恢复，模型缓存于
  `PUNC_MODEL_DIR`（缺失自动下载）。加载失败时自动降级回段级输出，不影响主链路。

### 失败响应

失败响应包含与成功响应一致的 `istar_asr`/`article_url`/`asr` 字段（均为空值），便于客户端用统一结构解析。

```json
{
  "code": 1001,
  "article_url": null,
  "istar_asr": "",
  "asr": [],
  "error": "DECODE_FAILED",
  "message": "音频解码失败，请确认为16kHz单声道WAV格式"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int | 业务错误码 |
| article_url | null | 失败时固定为 `null`（错误分支无法回读已解析请求中的 article_url） |
| istar_asr | string | 失败时固定为空字符串（与成功响应结构对齐） |
| asr | array | 失败时固定为空数组（与成功响应结构对齐） |
| error | string | 错误码名称 |
| message | string | 错误描述 |

---

## 错误码字典

| 错误码 | 名称 | HTTP Status | 说明 |
|--------|------|-------------|------|
| 0 | SUCCESS | 200 | 成功 |
| 1000 | INPUT_PARAM_FAILED | 400 | 输入参数错误（缺少 base64 字段、格式不合法等） |
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

热词通过 `hotwords` 字段传入（可选）。服务按**是否客户端主动传热词**两路分流（防通用识别误触发）：

| 条件 | 路径 | 处理方式 |
|---|---|---|
| 客户端传 `hotwords` | 路径 A：SeACo 在线热词 | 截断 Top256 → bias_encoder 编码 → ASF top-50 注入 decoder → 热词增强识别 |
| 客户端不传（默认词表） | 路径 B：Faiss 后处理纠错 | 普通 ASR → 拼音滑窗检索 → 三重联合判定 → 纠错 |

> 默认词表恒走 Faiss：通用识别多数音频不含默认热词，SeACo 会把声学相似普通词误纠成
> 热词（如"神棚"→"沈鹏"），Faiss 三重判定仅在高度吻合时替换，大幅降低误触发。

> **模块开关**：两条路径分别由环境变量 `ENABLE_HOTWORD`（路径 A）与
> `ENABLE_FAISS_CORRECTION`（路径 B）控制，默认均 `true`。纯通用识别、追求极限吞吐时
> 可将两者置 `false` 跳过全部热词处理；也可单独关闭其中一路按需组合。

### 路径 A：SeACo 在线热词

- 每个热词为一个独立字符串（人名、品牌名、术语等）
- **数量上限 256**：超过 256 个时截断保留前 256 个（Top256），并输出告警日志
- 显存上界恒定：热词维度固定 ≤256，TRT engine profile 无需重建
- **中英文混合**：中文逐字编码（精准）；英文单词经 seg_dict BPE 切分编码（如 `android`→正确 subword）。
  单热词最大长度 16 token（`MAX_HOTWORD_LEN`），超长英文词偏置可能略弱。以中文热词为主时效果最佳。

**示例：**

```json
{
  "hotwords": ["SeACo", "Paraformer", "达摩院", "张三"]
}
```

### 路径 B：默认词表纠错（客户端未传热词时恒走此路径）

- 服务端维护大词库（`models/asr/hotwords.txt`，规模 1 万~20 万）
- 普通 ASR 输出后，按拼音相似度 + 编辑距离做后处理纠错
- 与 GPU 显存解耦（Faiss CPU 索引），词库规模扩大不增加显存
- 三重联合判定才替换，最大限度防止误纠

---

## GET /health — 健康检查

反映**加载态 + 运行时健康**（不仅是模型加载成功与否）。健康返回 **HTTP 200**，
不健康（未加载 / 运行时卡死）返回 **HTTP 503**，供 K8s readiness/liveness 探针
（按状态码判定）真正摘除实例。

### 响应（健康：HTTP 200）

```json
{
  "status": "ok",
  "device": "cuda",
  "models_loaded": true,
  "runtime": {
    "backend": "trt",
    "consecutive_failures": 0,
    "total_infer_ok": 12345,
    "total_infer_fail": 0,
    "degraded_reason": null,
    "runtime_ok": true
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | `ok` / `degraded` |
| device | string | 当前推理设备（cuda/cpu） |
| models_loaded | bool | 模型是否已加载 |
| runtime.backend | string | 实际生效后端（trt/ort/ort_split/pt） |
| runtime.consecutive_failures | int | 当前连续推理失败次数（含超时），一次成功清零 |
| runtime.total_infer_ok/fail | int | 累计推理成功/失败次数 |
| runtime.degraded_reason | string\|null | 加载阶段静默降级原因（后端回退，如 TRT→ORT）；无降级为 null |
| runtime.runtime_ok | bool | 运行时是否健康（连续失败 < `HEALTH_MAX_CONSECUTIVE_FAILURES`） |

**status 判定规则：**

- 模型未加载 → `degraded`（探针识别加载失败/半死状态）
- 运行时连续推理失败超 `HEALTH_MAX_CONSECUTIVE_FAILURES`（默认 20，GPU 卡死典型症状）→ `degraded`，供 K8s/LB 摘除卡死实例
- 加载阶段发生静默降级（后端回退）→ 仍 `ok`（服务可用），但 `runtime.degraded_reason` 暴露原因供运维察觉低性能/降功能模式
- 否则 `ok`

> 开启 `HEALTH_ACTIVE_PROBE=true` 后，每次 `/health` 额外真跑一次极小 dummy 推理（带 `INFER_TIMEOUT` 超时），直接验证 GPU 链路存活（覆盖无流量期间的卡死）。默认 false（被动统计已足够，主动探针给每次健康检查加推理开销）。

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
  "route": "B",
  "message": "词表更新成功，已切换至 version 4"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| version | int | 更新后的词表版本号 |
| md5 | string | 词表内容哈希 |
| count | int | 有效词条数（去重 + 剔除 OOV 后） |
| route | string | 默认词表生效路径，恒为 B（Faiss）；路径 A 仅客户端传热词时使用 |

> 被剔除的 OOV 词不单独返回，可通过 count 变化 + 服务日志（含剔除详情）核对。

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
  "route": "B",
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
