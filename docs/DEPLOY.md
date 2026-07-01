# 部署指南

## Docker 构建

### 构建推理镜像

```bash
docker build -t seaco-asr:latest .
```

> 转换推理合一镜像，无需 `--target`。

### 镜像方案：转换推理合一（单镜像）

基础镜像 `nvcr.io/nvidia/tensorrt:24.11-py3` 已内置 TensorRT 10.6 + CUDA 12.6 +
cuDNN 9 + Python 3.10 + PyTorch 2.5。镜像在此基础上：

- 安装业务 + 转换依赖（`requirements-infer.txt`：onnxruntime-gpu / fastapi /
  nvidia-modelopt 等）
- 打包 `src/` `scripts/` `seaco_paraformer/` `models/`（含 PT 权重 + 配置）+ 校准数据
- 启动时由 `entrypoint.sh → prepare_model.py` 按 `MODEL_PRECISION` 从本地 PT 权重
  **现场逐级转换**出所需产物（PT → ONNX → TRT engine），无需独立转换镜像

> 设计取舍：转换与推理用同一镜像，省去跨镜像传递模型产物的复杂度；代价是镜像含
> torch/modelopt（体积较大）。如需精简纯推理镜像，可在镜像构建期预生成 engine 后
> 用 `.dockerignore` 排除 `seaco_paraformer/`、`calib_data/`，并改用不含转换依赖的
> requirements（当前未提供，需自行裁剪）。

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
WORKERS=1
BATCH=12
BATCH_TIMEOUT=10
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=2000
MODEL_PRECISION=trt_int8_enc
VERBOSE=0
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| HOST_PORT | 8099 | 宿主机映射端口 |
| WORKERS | 1 | uvicorn workers（默认 1，最小启动成本；可按显存调大） |
| BATCH | 12 | 最大 batch size（合法值：1,2,4,8,12） |
| BATCH_TIMEOUT | 10 | batch 等待超时（毫秒） |
| LOG_LEVEL | INFO | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| MAX_CONCURRENT_REQUESTS | 2000 | 最大并发请求数 |
| MODEL_PRECISION | auto | 模型精度选择（见 README MODEL_PRECISION 取值表） |
| VERBOSE | 0 | 详细日志输出（1=开启，输出各阶段耗时） |

> **WORKERS 默认 1**：单进程靠 asyncio + CPU 线程池实现并发，启动成本与显存占用最小。
> 代码已按多 worker 安全设计——词表热更新通过容器本地文件 + 版本轮询在各 worker 间收敛
> （见 API.md 词表热更新）。运维可按 GPU 显存调大 WORKERS，但每个 worker 进程独立加载
> 一份 engine + CUDA context，显存随 WORKERS 线性增长，需确认显存充足。

### MODEL_PRECISION 说明

完整精度矩阵见 README「MODEL_PRECISION 取值」表。常用值：

| 值 | 各段精度(enc/cif/dec/bias) | 适用 |
|------|------|------|
| auto | 自动探测 | GPU: trt_int8_enc→trt_fp16→trt_fp32→onnx_fp32；CPU: onnx_int8→onnx_fp32 |
| **trt_int8_enc** | int8/fp16/fp16/fp16 | **线上推荐**（encoder 显存减半，CER≈0，热词精度保留） |
| trt_fp16 | fp16×4 | GPU 通用 |
| trt_fp32 | fp32×4 | GPU 无损基线 |
| onnx_fp32 | ORT 整体 fp32 | CPU / 兜底 |
| onnx_int8 | ORT 整体 int8 动态量化 | CPU |

单段精度可用 `ENCODER_PRECISION`/`CIF_PRECISION`/`DECODER_PRECISION`/`BIAS_PRECISION` 覆盖。

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
            - name: WORKERS
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

| 场景 | WORKERS | BATCH | 副本数 | GPU |
|------|-------|-------|--------|-----|
| 开发测试 | 1 | 1 | 1 | 0-1 |
| 小规模（QPS<10） | 1 | 8 | 1 | 1 |
| 中规模（QPS 10-50） | 1 | 12 | 2 | 2 |
| 大规模（QPS>50） | 1 | 12 | 4+ | 4+ |

> GPU 推理服务每个 Pod 固定 WORKERS=1，通过增加副本数（Pod 数量）水平扩展。
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

> 仅在单进程模式（WORKERS=1）下有效。

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

线上推理使用 TensorRT 10.6 进行 GPU 推理，分段模型架构（encoder/cif/decoder/bias_encoder），
转换与推理合一镜像：启动时按 `MODEL_PRECISION` 从本地 PT 权重逐级转换出所需产物。

线上推荐精度 `trt_int8_enc`：encoder int8(QDQ) + cif/decoder/bias fp16
（encoder 显存减半且 CER≈0，热词精度保留）。

### 构建推理镜像

```bash
docker build -t seaco-asr:latest .
```

基础镜像：`nvcr.io/nvidia/tensorrt:24.11-py3`（TRT 10.6 + CUDA 12.6 + PyTorch 2.5）

### 启动服务

```bash
docker-compose up -d

# 查看日志
docker-compose logs -f seaco-asr
```

### 首次启动流程

1. `entrypoint.sh` 调用 `prepare_model.py` 按 `MODEL_PRECISION` 检查产物
2. 缺失则从本地 PT 权重（`PT_MODEL_DIR`，默认 `models/asr/pt`）逐级转换：
   PT → 分段 ONNX →（QDQ ONNX）→ TRT engine
3. Engine 写入镜像内 `models/asr/trt/`（容器层，无 volume 持久化）
4. 启动 uvicorn 服务 + 模型预热

> 首次启动总耗时约 15-20 分钟（engine 构建 + 预热）。
> K8s 部署时 `start_period` 需设为 900s。
> 注意：未挂载 engine 缓存 volume，容器**重建**会重新构建 engine（重启不会）。
> 如需避免重建开销，可在镜像构建期预生成 engine 一并打包。

### 词表热更新的持久化

默认词表 `models/asr/hotwords.txt` 打包在镜像内，`POST /hotwords/reload` 写入的是
**容器本地文件**：

- 运行期间：写入立即生效，多 worker 经版本轮询收敛（见 API.md）
- 容器**重启/重建**：容器层改动丢失，词表回到镜像打包时的版本

如需热更新结果跨容器生命周期持久化，二选一：
1. 把 `models/asr/hotwords.txt` 所在目录挂载为 volume（reload 写入持久化到宿主机）
2. 词表纳入镜像构建源，变更走重新构建镜像 + 滚动更新（与 engine 同策略）

> 临时/在线热词建议走客户端 `hotwords` 参数（路径 A），无需改默认词表。
> 默认大词库变更频率低时，推荐方式 2（镜像即词表，版本可追溯）。

### Engine 生成策略

- 镜像打包本地 PT 权重 + 配置；ONNX/engine 现场生成或预打包
- 首次启动自动检测 GPU 并构建对应 engine
- Engine 直接写入容器内 `models/asr/trt/`（不挂载 volume）
- 不同 GPU 自动生成不同文件名（`{gpu}_{module}_{precision}[_qdq].engine`）

### 回退机制

- 目标精度产物准备失败 → 回退 ORT onnx_fp32（现场生成兜底）
- TRT 推理异常 → 日志告警，返回错误码

### 精度方案对比

| MODEL_PRECISION | 各段精度(enc/cif/dec/bias) | 显存 | 适用 |
|---|---|---|---|
| onnx_fp32 | ORT 整体 fp32 | 基线 | CPU/兜底 |
| onnx_int8 | ORT 整体 int8 | 减小 | CPU |
| trt_fp16 | fp16×4 | ~50% | GPU 通用 |
| **trt_int8_enc** | int8/fp16/fp16/fp16 | 更省 | **线上推荐** |
| trt_int8 | int8×4（QDQ） | 最省 | 实测可跑但精度损失较大，不推荐线上 |

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
