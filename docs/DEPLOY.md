# 部署指南

## Docker 构建

### 构建推理镜像

```bash
docker build -t seaco-asr:latest .
```

> 默认构建最后一个 stage（inference），无需指定 `--target`。

### 构建转换镜像（仅模型导出时使用）

```bash
docker build --target converter -t seaco-asr-converter:latest .
```

### 镜像分层策略

Dockerfile 采用多阶段构建：
- **Stage 1 (converter)**：包含 PyTorch + FunASR，用于模型 ONNX 导出
- **Stage 2 (inference)**：仅包含 ONNX Runtime + FastAPI，轻量化推理

推理镜像不包含 PyTorch/FunASR，体积更小、启动更快。

---

## Docker Compose 部署

```bash
# 启动服务（宿主机 8099 → 容器 8080）
docker-compose up -d

# 停止服务
docker-compose down

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f seaco-asr
```

### 环境变量配置

通过 `.env` 文件或环境变量覆盖默认配置：

```bash
# .env 示例
HOST_PORT=8099
WORKS=1
BATCH=12
BATCH_TIMEOUT=10
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=2000
MODEL_PRECISION=auto
VERBOSE=0
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| HOST_PORT | 8099 | 宿主机映射端口 |
| WORKS | 1 | uvicorn workers（GPU 服务必须为 1） |
| BATCH | 12 | 最大 batch size（合法值：1,2,4,8,12） |
| BATCH_TIMEOUT | 10 | batch 等待超时（毫秒） |
| LOG_LEVEL | INFO | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| MAX_CONCURRENT_REQUESTS | 2000 | 最大并发请求数 |
| MODEL_PRECISION | auto | 模型精度选择（auto/fp32/int8） |
| VERBOSE | 0 | 详细日志输出（1=开启，输出各阶段耗时） |

> **重要**：GPU 推理服务 WORKS 必须为 1（单进程模式），靠 asyncio + CPU 线程池实现并发。
> 多 worker 会导致多进程 fork，空闲时 CPU 占用异常。

### MODEL_PRECISION 说明

| 值 | 行为 |
|------|------|
| auto | GPU 环境自动选 fp32，CPU 环境优先选 int8（若存在） |
| fp32 | 强制使用 fp32 模型（GPU/CPU 均可） |
| int8 | 强制使用 int8 量化模型（仅 CPU） |

---

## K8s 部署示例

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: seaco-asr
  labels:
    app: seaco-asr
spec:
  replicas: 1
  selector:
    matchLabels:
      app: seaco-asr
  template:
    metadata:
      labels:
        app: seaco-asr
    spec:
      containers:
        - name: seaco-asr
          image: seaco-asr:latest
          ports:
            - containerPort: 8080
          env:
            - name: WORKS
              value: "1"
            - name: BATCH
              value: "12"
            - name: BATCH_TIMEOUT
              value: "10"
            - name: LOG_LEVEL
              value: "INFO"
            - name: MAX_CONCURRENT_REQUESTS
              value: "2000"
          resources:
            requests:
              memory: "2Gi"
              cpu: "2"
              nvidia.com/gpu: "1"
            limits:
              memory: "8Gi"
              cpu: "8"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: logs
              mountPath: /app/logs
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 420
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 420
            periodSeconds: 10
      volumes:
        - name: logs
          emptyDir: {}
```

> **注意**：`initialDelaySeconds` 设为 420s（7分钟），因为服务启动时需要预热所有 bucket × batch 组合（约 6 分钟）。

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: seaco-asr
spec:
  selector:
    app: seaco-asr
  ports:
    - port: 8080
      targetPort: 8080
  type: ClusterIP
```

### HPA（自动扩缩容）

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: seaco-asr-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: seaco-asr
  minReplicas: 1
  maxReplicas: 8
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: asr_inference_duration_seconds
        target:
          type: AverageValue
          averageValue: "2"
```

---

## 扩缩容建议

| 场景 | WORKS | BATCH | 副本数 | GPU |
|------|-------|-------|--------|-----|
| 开发测试 | 1 | 1 | 1 | 0-1 |
| 小规模（QPS<10） | 1 | 8 | 1 | 1 |
| 中规模（QPS 10-50） | 1 | 12 | 2 | 2 |
| 大规模（QPS>50） | 1 | 12 | 4+ | 4+ |

> GPU 推理服务每个 Pod 固定 WORKS=1，通过增加副本数（Pod 数量）水平扩展。
> 每个 Pod 绑定一张 GPU，不共享。

### 关键调优参数

| 参数 | 调优建议 |
|------|----------|
| BATCH | 增大可提高 GPU 利用率，但增加单请求延迟。合法值：1,2,4,8,12 |
| BATCH_TIMEOUT | 减小可降低延迟，但降低 batch 填充率。默认 10ms |
| MAX_CONCURRENT_REQUESTS | 控制最大并发，防止内存溢出。默认 2000 |
| VERBOSE | 开启后输出各阶段详细耗时，便于性能分析 |

---

## 配置热更新

支持通过 SIGHUP 信号动态调整部分参数，无需重启服务：

```bash
# 修改环境变量后发送信号
export BATCH_TIMEOUT=20
export LOG_LEVEL=DEBUG
kill -HUP $(pgrep -f "uvicorn src.main:app")
```

可热更新参数：
- `BATCH_TIMEOUT`：batch 等待超时
- `LOG_LEVEL`：日志级别

> 仅在单进程模式（WORKS=1）下有效。

---

## 启动时间说明

服务启动时会执行模型预热（bucket × batch 全组合），确保线上首次请求不会因 CUDA kernel 编译而变慢：

| 阶段 | 耗时 |
|------|------|
| 模型加载 | ~10s |
| ASR 预热（3 bucket × 5 batch = 15 组合） | ~5-6min |
| 特征提取预热 | <1s |
| 端到端预热 | <1s |
| **总计** | **~6min** |

K8s 部署时需确保 `initialDelaySeconds` 大于预热时间，避免 Pod 被误判为不健康而重启。

---

## v2 TensorRT 部署

### 概述

v2 使用 TensorRT 10.6 替代 ORT 进行 GPU 推理，分段模型架构：
- **encoder**（fp32）+ **cif**（fp16）+ **decoder**（fp16）
- 相比 v1 ORT fp32，推理速度提升约 2-3x，显存减半

### 构建 TRT 推理镜像

```bash
docker build -f Dockerfile.trt -t seaco-asr:trt .
```

基础镜像：`nvcr.io/nvidia/tensorrt:24.11-py3`（TRT 10.6 + CUDA 12.6 + PyTorch 2.5）

### 启动服务

```bash
# 使用 TRT 专用 docker-compose
docker-compose -f docker-compose.trt.yml up -d

# 查看日志
docker-compose -f docker-compose.trt.yml logs -f seaco-asr-trt
```

### 首次启动流程

1. `entrypoint_trt.sh` 检测 TRT engine 是否存在
2. 不存在则自动构建（约 5-10 分钟）
3. Engine 缓存到 Docker volume（`trt_engine_cache`），重启不重新构建
4. 启动 uvicorn 服务 + 模型预热

> 首次启动总耗时约 15-20 分钟（engine 构建 + 预热）。
> K8s 部署时 `start_period` 需设为 900s。

### Engine 缓存策略

- 镜像内只打包 ONNX fp32 分段模型
- 首次启动时自动检测 GPU 并构建对应 engine
- Engine 缓存到 Docker volume，持久化
- 不同 GPU 自动生成不同文件名（`{gpu}_{model}_{precision}.engine`）

### 回退机制

- TRT engine 构建失败 → 服务仍可启动（回退 ORT fp32）
- TRT 推理异常 → 日志告警，返回错误码

### v1 vs v2 对比

| 维度 | v1（ORT） | v2（TRT） |
|------|-----------|-----------|
| 推理引擎 | ONNX Runtime | TensorRT 10.6 |
| 模型精度 | fp32 | encoder fp32 + cif/decoder fp16 |
| 推理速度 | 基线 | ~2-3x 提升 |
| 显存占用 | 基线 | ~50% 减少 |
| 部署复杂度 | 低 | 中（需 engine 构建） |
| Dockerfile | Dockerfile | Dockerfile.trt |
| docker-compose | docker-compose.yml | docker-compose.trt.yml |

---

## 日志管理

### 日志输出

- **stdout**：JSON 格式，供 Docker/ELK 采集
- **本地文件**：`logs/asr_{date}.log`，按天轮转，保留 7 天

### 日志字段

```json
{
  "timestamp": "2026-05-15 10:30:42,310",
  "level": "INFO",
  "logger": "seaco_asr",
  "message": "服务启动完成",
  "request_id": "abc123"
}
```

### VERBOSE 模式

设置 `VERBOSE=1` 后，输出各阶段详细耗时：

```
[Stage1] 解码=5ms, VAD=120ms(6段), 切段=1ms(8chunks)
[Stage2] 特征提取=45ms, chunks=8, shapes=[(34,560),(67,560),...]
[Stage3] bucket=67, batch=4, 推理=85ms
```
