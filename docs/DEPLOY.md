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
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f seaco-asr
```

### 自定义配置

通过 `.env` 文件或环境变量覆盖默认配置：

```bash
# .env
WORKS=4
BATCH=16
DEVICE=cuda
MAX_BATCH_DURATION=30
```

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
  replicas: 2
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
            - containerPort: 30960
          env:
            - name: WORKS
              value: "4"
            - name: BATCH
              value: "16"
            - name: DEVICE
              value: "cuda"
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
              port: 30960
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 30960
            initialDelaySeconds: 30
            periodSeconds: 10
      volumes:
        - name: logs
          emptyDir: {}
```

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
    - port: 30960
      targetPort: 30960
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
| 小规模（QPS<10） | 2 | 8 | 1 | 1 |
| 中规模（QPS 10-50） | 4 | 16 | 2 | 2 |
| 大规模（QPS>50） | 4 | 32 | 4+ | 4+ |

### 关键调优参数

- **BATCH**：增大可提高 GPU 利用率，但增加单请求延迟
- **BATCH_TIMEOUT**：减小可降低延迟，但降低 batch 填充率
- **WORKS**：CPU 核数的 1-2 倍，过多会增加 GPU 竞争
- **MAX_BATCH_DURATION**：限制单次推理的总音频时长，防止 OOM
