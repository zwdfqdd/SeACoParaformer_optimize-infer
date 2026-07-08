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
# .env 示例（默认 = 模式 A 单进程，任何硬件都能跑）
HOST_PORT=8099
WORKERS=1
BATCH=12
BATCH_TIMEOUT=10
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=2000
MODEL_PRECISION=trt_int8_enc
VERBOSE=0

# 稳定性/性能相关（run.sh 已固化默认值）
VAD_SESSION_POOL_SIZE=4
GPU_STREAM_POOL_SIZE=4
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1

# 大 GPU 生产环境（模式 B，需要 24GB+ 显存）
# WORKERS=11
# MODEL_PRECISION=trt_fp16
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| HOST_PORT | 8099 | 宿主机映射端口 |
| WORKERS | 1 | uvicorn workers；小 GPU 保持 1，大 GPU 可设 11 见「高并发调优 模式 B」 |
| BATCH | 12 | 最大 batch size（合法值：1,2,4,8,12） |
| BATCH_TIMEOUT | 10 | batch 等待超时（毫秒），工业标准 dynamic batching 的 max_queue_delay（实测 10 吞吐最优） |
| LOG_LEVEL | INFO | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| MAX_CONCURRENT_REQUESTS | 2000 | 最大并发请求数 |
| MODEL_PRECISION | auto | 模型精度选择（见 README MODEL_PRECISION 取值表） |
| VAD_SESSION_POOL_SIZE | 4 | Silero VAD ORT session 池大小，round-robin 分配 |
| GPU_STREAM_POOL_SIZE | 4 | TRT engine 多 stream 多 execution_context 池 |
| OMP_NUM_THREADS | 1 | ★必须 1，防 libgomp 崩溃（run.sh 已固化） |
| MKL_NUM_THREADS | 1 | 同上 |
| OPENBLAS_NUM_THREADS | 1 | 同上 |
| VERBOSE | 0 | 详细日志输出（1=开启，输出各阶段耗时） |

> **两种部署形态**（详见「高并发性能调优」章节）：
> - **模式 A（默认）**：`WORKERS=1`，任何硬件都能跑，QPS ~13（conc=20）
> - **模式 B 高并发**：`WORKERS=11`，需 24GB+ GPU，QPS 93.92（conc=120）
>
> 代码已按多 worker 安全设计——词表热更新通过容器本地文件 + 版本轮询在各 worker 间收敛
> （见 API.md 词表热更新）。每个 worker 独立加载一份 engine + CUDA context，
> 显存随 WORKERS 线性增长，需确认显存充足。

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

## K8s 部署（多节点多卡）

完整清单见 [`deploy/k8s.yaml`](../deploy/k8s.yaml)，包含 Deployment + Service（NodePort 30960）。

### 部署形态

- **节点**：多台 GPU 服务器，每台 3 张 GPU 卡，打标签 `asr=asr-gpu-static`
- **Pod**：每张 GPU 卡部署 1 个 Pod（每 Pod 独占 1 卡）
- **对外端口**：容器 8080 → NodePort 30960（每节点宿主机暴露此端口，负载均衡到本节点 Pod）

### 关键约束

- `nodeSelector: asr: asr-gpu-static` 只调度到指定节点
- `resources.requests.nvidia.com/gpu: 1` Pod 独占 1 卡 → 每节点最多 3 Pod（受 GPU 数天然约束）
- `topologySpreadConstraints` 均匀分布，避免 Pod 堆积到单个节点
- `Service.externalTrafficPolicy: Local` 只转发到本节点的 Pod，保留 client IP + 减少跨节点跳数

### 应用命令

```bash
# 1. 给 GPU 节点打标签（每个 GPU 服务器执行）
kubectl label node <node-name> asr=asr-gpu-static

# 2. 按实际节点数调整 replicas（例：3 节点 × 3 卡 = 9）
# 修改 deploy/k8s.yaml 中 spec.replicas: 3 → 9

# 3. 部署
kubectl apply -f deploy/k8s.yaml

# 4. 查看 Pod 分布（应每节点 3 个）
kubectl get pods -l app=seaco-asr -o wide

# 5. 测试访问（任意节点 IP + 30960）
curl http://<node-ip>:30960/health
```

### 启动等待时间

服务启动需构建 TRT engine（5-10 min）+ 预热（约 6 min），总计约 15 分钟。

清单中 `startupProbe` 允许 `30 × 30s = 15 分钟` 内完成健康检查，超过则重启 Pod。
如果实际环境预热更慢（大 batch/多热词维度），调整 `startupProbe.failureThreshold`。

### 扩缩容策略

- **纵向**：改 `MODEL_PRECISION`（int8_enc 省显存）或 `WORKERS`（大 GPU 走模式 B）
- **横向**：新增节点后重打标签 + 调大 `replicas`（`replicas = 节点数 × 3`）
- 本方案**不用 HPA**（GPU 资源固定，副本数由节点数决定，不适合根据 CPU/QPS 自动扩缩）

---

## 扩缩容建议

按 GPU 显存和目标 QPS 选择两种部署模式（详见「高并发性能调优」章节）。

| 场景 | 目标 QPS | WORKERS | BATCH | 副本数 | 单卡显存 | 单卡 GPU |
|---|---|---|---|---|---|---|
| 开发测试 | <5 | 1 | 1 | 1 | 2GB | 任意 |
| 小规模（模式 A） | <15 | 1 | 12 | 1-2 | ~1.5GB | 8-12GB 类（2080 Ti/T4） |
| 中规模（模式 A 多副本） | 15-50 | 1 | 12 | 2-4 | ~1.5GB × 副本 | 每 Pod 一张 8-12GB 卡 |
| 大规模（模式 B） | 50-100 | **11** | 12 | 1-2 | ~15-20GB | A10 24GB / A100 |
| 超大规模（模式 B 多副本） | >100 | 11 | 12 | 2+ | 20GB × 副本 | A10/A100 集群 |

**两种模式的选择依据**：
- **模式 A（WORKERS=1）**：显存少、启动快、任何硬件都能跑，多 Pod 水平扩展
- **模式 B（WORKERS=11）**：大 GPU 单卡榨到极限，单 Pod QPS 93.92，减少 Pod 数量

### 关键调优参数

| 参数 | 调优建议 |
|------|----------|
| WORKERS | **1** 或 **11**（详见「WORKERS 参数说明」），中间值不推荐 |
| BATCH | 增大可提高 GPU 利用率，但增加单请求延迟。合法值：1,2,4,8,12 |
| BATCH_TIMEOUT | 减小可降低延迟，但降低 batch 填充率。默认 10ms（实测吞吐最优） |
| MAX_CONCURRENT_REQUESTS | 控制最大并发，防止内存溢出。默认 2000 |
| VAD_SESSION_POOL_SIZE | 默认 4；单进程 20+ 并发可调 8 |
| GPU_STREAM_POOL_SIZE | 默认 4；显存充足可扩到 8 |
| OMP_NUM_THREADS | ★必须 1，防 libgomp 崩溃 |
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

服务启动时会执行模型预热（bucket × batch 全组合），确保线上首次请求不会因 CUDA kernel
编译而变慢。Uniform Chunking 后主要工作点是 opt=67 帧，最大桶 134 帧仍保留兜底：

| 阶段 | 耗时 |
|------|------|
| 模型加载 | ~10s |
| ASR 预热（3 bucket × 5 batch × 热词维度组合） | ~5-6min |
| 特征提取预热 | <1s |
| 端到端预热 | <1s |
| **总计** | **~6min** |

K8s 部署时需确保 `initialDelaySeconds` 大于预热时间，避免 Pod 被误判为不健康而重启。

> 若显式设 `WORKERS>1`（如模式 B），预热在**每个 worker 进程各执行一次**（并行），
> 不影响墙钟时间（多进程同时预热），但每个进程仍需消耗独立显存。

---

## v2 TensorRT 部署

### 概述

线上推理使用 TensorRT 10.6 进行 GPU 推理，分段模型架构（encoder/cif/decoder/bias_encoder
+ 可选 timestamp 第 5 段），
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
[Stage1] 解码=5ms, VAD=120ms(6段), 切段=1ms(7chunks)
[Stage2] 特征提取=45ms, chunks=7, shapes=[(67,560),(67,560),...]  # uniform 4020ms 均匀切段
[Stage3] target_seq_len=67, batch=4, 推理=85ms  # 按 batch 内 max(lengths) 动态 pad
```

---

## 高并发性能调优（2026-07 阶段沉淀）

本节记录从初始基线到当前最优配置的完整性能演进路径与关键参数含义，供后续运维/迭代参考。

### 性能演进路径

| 阶段 | 关键改动 | QPS(conc10,30s) | QPS(conc20,30s) | avg_batch | 说明 |
|---|---|---|---|---|---|
| 基线 | 桶分组 + EAGER=4 触发 | 4.79 | 4.21 | 2.04 | 起点 |
| Scheduler 工业标准 | max_batch_size + max_queue_delay | 4.21 | — | 2.10 | 桶分组阻碍合批，触发方式改无效 |
| Uniform Chunking | audio_segment 均匀 4020ms 切段 + 尾块<1s 并入前段 | 4.21 | 崩溃 | **8.14** | 合批目标达成，但 CPU 侧被打满 |
| VAD Session 池 | pool=8 round-robin + arena off | 10.13 | 崩溃 150/400 | 8+ | 多 session 并发，但 OMP 线程爆炸 |
| **OMP_NUM_THREADS=1** | 固化 OMP/MKL/BLAS 单线程 | 12.32 | 12.32 | 8+ | 消除 libgomp 崩溃 |
| **Pool Size 扫描收敛** | pool=2 最快, pool=4 平衡 | ~13 | **12.88（pool=2）/ 12.67（pool=4）** | 8+ | 单进程最优 |
| **多 stream 多 context** | GPU_STREAM_POOL_SIZE=4，engine 共享 weights + 4 context/stream | ~13 | 13.15 | 9.09 | Stage3 GPU -47%，P99 -22% |
| **★多进程 WORKERS=11** | 大 GPU 上多进程隔离 CPU 竞争（模式 B） | — | — | 9+ | **conc=120 QPS 93.92** ★峰值 |

**累计 QPS 提升**：
- 单进程模式：4.79 → 13.15（**+174%**，conc=20）
- 多进程模式：4.79 → **93.92**（**+1861%**，conc=120，大 GPU）

**P99 延迟**：3454ms → 1640ms（单进程，**-53%**）

### VAD Session Pool 扫描结果（20 并发 × 30s 音频 × 400 请求）

固定 `OMP_NUM_THREADS=1`，扫描 `VAD_SESSION_POOL_SIZE`：

| Pool Size | QPS | 平均延迟 | P99 | 稳定性 | 备注 |
|---|---|---|---|---|---|
| 1 | 12.76 | 1542ms | 2141ms | ✓ | 单 session 反而不差 |
| **2** | **12.88** | **1529ms** | **1951ms** | ✓ | **★峰值** |
| 4 | 12.67 | 1556ms | 1941ms | ✓ | 推荐默认（留余量） |
| 8 | 12.48 | 1579ms | 2004ms | ✓ | — |
| 16 | 12.32 | 1598ms | 2168ms | ✓ | — |
| 32 | 12.08 | 1631ms | 2052ms | ✓ | — |

**反直觉现象**：Pool 越大 QPS 反而略降。
- 原因：OMP=1 后单 session 已高效，多 session 增加内存分配 / 缓存 miss / 上下文切换开销
- 结论：Pool=2 QPS 最高但太紧；**Pool=4 留 2 个余量应对突发，性能仅差 1.6%**

### OMP_NUM_THREADS 扫描（20 并发，Pool=32 固定）

| OMP | QPS | 平均延迟 | 稳定性 |
|---|---|---|---|
| **1** | **12.08** | **1631ms** | ✓ |
| 2 | 11.51 | 1711ms | ✓ |
| 4 | 11.51 | 1712ms | ✓ |
| 8 | 11.06 | 1781ms | ✓ |

**OMP=1 最优**，因为：
- VAD（Silero）是串行 LSTM，OMP 内部并行无收益
- 高并发下多 session × 多 OMP 线程 = 线程爆炸，触发 `libgomp thread creation failed`
- 关闭 OMP 内部并行反而降低系统开销

### 关键参数说明

#### `WORKERS`（部署形态的核心决策）
- 作用：uvicorn worker 进程数，每 worker 独立加载模型 + 独立 CUDA context
- 建议值：
  - **小 GPU（<12GB）或低并发（<20）**：**1**（模式 A）
  - **大 GPU（≥24GB）+ 高并发（>50）**：**11**（模式 B，实测最优）
- 显存约束：`WORKERS × 1.5GB` ≤ GPU 显存
- 调参路径：
  - 从 WORKERS=1 起步验证单进程性能
  - 显存和并发都充足时逐步扫描 4/6/8/11 找 QPS 拐点
- **反直觉**：大 GPU 上 WORKERS>1 反而更快（进程隔离消除 CPU 竞争）

#### `OMP_NUM_THREADS`（强烈建议保持 1）
- 作用：控制 libgomp 每进程 OpenMP 线程数
- 影响范围：ORT / numpy / torch / MKL / OpenBLAS 内部算子并行
- 建议值：**1**（默认）
- 什么时候可以尝试 >1：非 VAD 场景、CPU 未跑满、有大矩阵密集计算
- 相关：`MKL_NUM_THREADS` `OPENBLAS_NUM_THREADS` 一并设为 1（若上层库走不同 BLAS）

#### `ENABLE_WORD_TIMESTAMP`（默认 false）
- 作用：是否返回字级时间戳（响应 `asr[].words`，每字带 [start_s, end_s]）
- 实现：独立第 5 段 timestamp engine（upsample CIF timestamp head + blstm），
  对齐 FunASR 官方 `ts_prediction_lfr6_standard`，精度约 20ms
- **性能影响**：启用后吞吐下降约 30%（实测 2800 → 2000 req/s），因 upsample+blstm
  对每个请求额外一次 GPU 推理
- 建议：
  - **需要字幕/对齐/精确定位** → true（接受吞吐下降）
  - **纯转写、追求吞吐** → false（默认，words 为空）
- 启用前提：`ENABLE_WORD_TIMESTAMP=true` 时 prepare_model 会额外构建 timestamp engine
  （首次启动多几分钟），engine 命名 `{gpu}_timestamp_fp16.engine`

#### `CPU_THREAD_POOL_SIZE` / `ORT_INTRA_OP_THREADS` / `ORT_INTER_OP_THREADS`（CPU 侧线程）
三个参数职责与**生效后端**不同，务必区分：

| 参数 | 作用对象 | 生效后端 | 说明 |
|------|---------|---------|------|
| `CPU_THREAD_POOL_SIZE` | Stage1 VAD + Stage2 特征提取的线程池 | ★所有后端（含 TRT/GPU） | 0=自动取全核；多 worker 务必设小，否则每 worker 各开满核线程超额订阅 |
| `ORT_INTRA_OP_THREADS` | 主 ASR 模型单 session 算子并行 | **仅 CPU 后端**（onnx_fp32/onnx_int8） | TRT/GPU 部署完全不生效（主 ASR 在 GPU 推理） |
| `ORT_INTER_OP_THREADS` | 主 ASR 模型 session 间并行 | **仅 CPU 后端** | 同上 |

- **重要**：`ORT_INTRA/INTER_OP_THREADS` 对 **VAD 无效**——Silero VAD 是串行 LSTM，
  `vad.py` 中 session 线程数硬编码为 `intra=inter=1`（单次 run 仅处理 (1,576) 极小张量，
  多线程无收益且徒增调度开销），VAD 的并行靠 `VAD_SESSION_POOL_SIZE` 多 session round-robin。
- **GPU/TRT 网格压测结论**：能影响 TRT 吞吐的 CPU 侧参数只有 `CPU_THREAD_POOL_SIZE` 与
  `VAD_SESSION_POOL_SIZE`；把 `ORT_INTRA/INTER_OP_THREADS` 列为网格变量属无效项（结果全是噪声）。
- **`CPU_THREAD_POOL_SIZE` 取值建议**（256 核机器，WORKERS=10）：总线程 ≈ WORKERS × 值，
  不超订分界 ≈ 256/10 ≈ 25/worker；推荐扫描 `{8, 16, 24, 32}`，勿到 64（10×64=640 严重超订）。

#### `VAD_SESSION_POOL_SIZE`（推荐 4，可调 2-8）
- 作用：Silero VAD ORT session 池大小，多请求 round-robin 分配
- 影响范围：仅 Stage1 VAD
- **单 session 线程数硬编码 intra=inter=1**（见 vad.py），并行度完全由本 pool 提供
- 建议值：**4**（默认）
- 调优场景：
  - 低并发（<10）：可降到 2 省内存
  - 高并发（>30）：可升到 8，但收益递减
  - 观测 `stage1_vad_sum/count` 均值：>800ms 时考虑增大 pool
- **注意**：与 OMP 相关：`OMP × Pool ≈ 有效线程数`，OMP=1 时 Pool 数直接对应真实并发

#### `BATCH_TIMEOUT`（推荐 10ms）
- 作用：GPU Scheduler 单个 chunk 最长排队时间（工业标准 max_queue_delay）
- 影响：单请求 GPU 侧延迟严格上限 ≤ BATCH_TIMEOUT + 1ms tick
- 建议值：**10ms**（默认，性能网格实测吞吐最优）
- 调优场景：
  - 高并发下 chunk 到达密集，10ms 已能合到大 batch，无需更大超时
  - 追求更强合批（低并发大音频）：可上调 20-30ms（延迟略升）
  - 追求最高吞吐：50-100ms（合批更强，延迟略升）
- 与 `VALID_BATCH_SIZES[-1]=12` 配合：满 12 立即触发，未满按 timeout 兜底

#### `MODEL_PRECISION`（推荐 trt_int8_enc）
- 精度矩阵见 README，简要：
  - `trt_int8_enc`：encoder int8 + 其余 fp16，最快，CER≈0 ★线上推荐
  - `trt_fp16`：全 fp16，稳定，CER 与 baseline 一致
  - `trt_fp32`：全 fp32，无损但最慢
  - `onnx_int8` / `onnx_fp32`：CPU 兜底

### 生产环境推荐配置

按流量规模分为两种部署模式，**多进程模式在大 GPU + 高并发下 QPS 提升 7 倍**。

#### 模式 A：单进程（低-中并发 <20，2080 Ti 类小 GPU）

```bash
MODEL_PRECISION=trt_int8_enc     # 或 trt_fp16
WORKERS=1                        # 单进程独占 GPU
BATCH_TIMEOUT=10
VAD_SESSION_POOL_SIZE=4
GPU_STREAM_POOL_SIZE=4
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

实测性能（2080 Ti + trt_fp16，30s 音频）：
- QPS：13.15（conc=20）
- 平均延迟：1500ms
- P99：1640ms
- 显存：~1.5GB
- 适用：小规模服务、开发调试、显存受限

#### 模式 B：★多进程高并发（>50 并发，大 GPU / A10 24GB+）

```bash
MODEL_PRECISION=trt_fp16
WORKERS=11                       # ★关键：多进程隔离 CPU 竞争
BATCH_TIMEOUT=10
VAD_SESSION_POOL_SIZE=4
GPU_STREAM_POOL_SIZE=4
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

实测性能（大 GPU + trt_fp16，30s 音频 concurrency=120，total=2500）：
- **QPS：93.92**（较模式 A 提升 614%）
- **吞吐量：2823 audio_s/s**（每秒处理 47 分钟音频）
- 平均延迟：1226ms
- P99：2785ms
- 成功率：100%
- 显存：~15-20GB
- 适用：大流量生产环境

**为什么 WORKERS=11 反直觉地更快**：
- 之前 plan.md 记录"GPU 服务必须 WORKERS=1"是基于 2080 Ti 小 GPU 场景
- 大 GPU 显存充足时，多进程隔离 CPU 竞争的收益 > CUDA context 切换开销
- 每 worker 分摊约 10-11 并发，正好命中单进程最优点（模式 A）
- 11 进程并行 → 总 QPS ≈ 单进程 QPS × 7-8 倍

**WORKERS 调参经验**：
- 显存充足：WORKERS = 目标并发数 / 10 附近
- 显存紧张：以显存除以单 engine 大小算上限（如 24GB / 1.5GB ≈ 16）
- 未验证过 WORKERS>11，可根据实际显存扫描 4/6/8/11/16 找拐点

#### 重要发现：VAD 保持 CPU 是最优（不要上 GPU）

`VAD_PROVIDER=cuda` 曾被尝试作为优化选项，实测**反而慢 4x**（WORKERS=1 场景）
或**无收益**（WORKERS=11 场景，2823 vs 2700 audio_s/s）。

根因：Silero VAD 单次调用只处理 (1,576) 小 tensor，30s 音频 937 次 session.run。
PCIe H2D+D2H 传输开销（~60μs/次）远大于 GPU kernel 计算（~5μs/次），
GPU sm-util 假高（大部分是 memory 等待）。因此代码固定使用 CPUExecutionProvider。

### 崩溃故障排查

**症状 1：`libgomp: Thread creation failed: Resource temporarily unavailable`**
- 原因：OMP 线程池爆炸
- 解决：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`

**症状 2：`free(): corrupted unsorted chunks` / `Segmentation fault`**
- 原因：ORT session 并发调用时 mem arena 竞态
- 解决：`src/vad.py` 已 `enable_cpu_mem_arena=False` + `enable_mem_pattern=False`

**症状 3：`avg_actual_batch < 3`（合批目标未达成）**
- 原因：桶分组把 chunk 摊薄
- 解决：确认 `src/audio_segment.py` 已启用 Uniform Chunking

**症状 4：`stage1_vad` 均值 >1s（VAD 阻塞）**
- 原因：session pool 太小 or OMP 争用
- 解决：`VAD_SESSION_POOL_SIZE=8` 或检查 OMP_NUM_THREADS=1

