# SeACo-Paraformer ASR 服务

基于 SeACo-Paraformer 的工业级中文语音识别服务，支持热词定制、动态 Batch 推理、GPU 加速。

## 环境准备

### 系统要求

- 基础镜像：`nvcr.io/nvidia/tensorrt:24.11-py3`（TensorRT 10.6 + CUDA 12.6 + cuDNN 9 + Python 3.10 + PyTorch 2.5）
- Docker + Docker Compose + NVIDIA Container Toolkit

### 安装依赖

```bash
# 推理 + 现场转换统一依赖（转换推理合一镜像）
pip install -r requirements-infer.txt
```

> TensorRT/PyTorch 由基础镜像内置，requirements 仅补充业务依赖 + 转换工具（onnx / nvidia-modelopt）。

---

## v2 推理路径（推荐：纯 fp16）

### 总体技术路径

通过 3 个独立技术点叠加，实现纯 fp16 推理（无任何 fp32 fallback）：

| # | 技术点 | 实现位置 | 作用 |
|---|---|---|---|
| 1 | **opset 17 LayerNormalization 单节点** | `scripts/export_onnx_split.py` 默认 `--opset 17` | PyTorch 2.5 trace 自动识别 `nn.LayerNorm`，导出为单节点；TRT 10.6 内部对该节点自动 fp32 累加，避免 fp16 LayerNorm 内 `(x-mean)²` 溢出 |
| 2 | **encoder 残差 Add 后 clamp 60000** | `seaco_paraformer/encoder.py` 的 `EncoderLayerSANM(clamp_value=60000)` | encoder 后段层残差激活峰值高达 ~48万（远超 fp16 上限 65504）；60000 贴近 fp16 上限最大化保留信息，防溢出 inf。clamp=30000 裁剪过狠致截断输入解码错乱，已弃用 |
| 3 | **纯 trtexec --fp16 转换** | `scripts/convert_trt.py --precision fp16` | 不需要 Python TRT API、不需要 OBEY_PRECISION_CONSTRAINTS、不需要任何手动 fp32 fallback |

### 完整执行流程

```bash
# Step 1：PT baseline 验证（转换容器内）
python tests/test_pt_inference_v2.py \
    --audio test_data/audio_16000_10s.wav \
    --hotwords 埃文 账号

# Step 2：导出分段 ONNX（注入 clamp=60000）
python scripts/export_onnx_split.py \
    --output-dir ./models/asr/split \
    --clamp-value 60000

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
| token_num | 与 PT baseline 一致 | CIF predictor 数值稳定 |
| encoder max | ~0.4（与 PT baseline 同量级） | 数值量级一致 |
| nan/inf | False | 无溢出 |
| 识别文本 | 与 PT baseline 一致（fp16 仅极少数边缘 token 微抖） | 推理等价 |
| RTX | ~80-100x | 性能符合预期（2080 Ti） |

> **fp16 本质局限**：encoder 后段激活天然 ~48万，fp16(65504) 无法无损表示，
> clamp=60000 仍裁极少数峰值点（单条样本偶见叠字微抖，CER 影响极小）。
> 追求完全无损用 `trt_fp32` / `onnx_fp32`（clamp=0）。

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

## v2 阶段 2：INT8 量化（QDQ Explicit）

### 核心结论

| 方案 | 在 SeACo 架构上的效果 |
|---|---|
| Calibrator Implicit（`IInt8EntropyCalibrator2`） | ❌ 无效。encoder MatMul 被 TRT myelin 融合进 LayerNorm 大 kernel，融合 kernel 不支持部分 INT8，全部 fall back fp16（engine 体积不降，INT8 层=0） |
| **QDQ Explicit（`nvidia-modelopt`）** | ✅ 有效。Q/DQ 节点显式标记量化边界，TRT 不融合掉，INT8 真正生效 |

### 环境依赖（仅 INT8 导出需要）

```bash
# 必须钉版本！0.44+ 会把 torch 顶到 2.12+cu130 破坏 torchaudio/TRT 环境
pip install nvidia-modelopt==0.21.0 torchprofile pulp regex --extra-index-url https://pypi.nvidia.com

# pulp/regex 为 modelopt 0.21 隐性依赖，缺失会报误导性的
#   "Please install optional [torch] dependencies"（实为缺这两个纯工具库）
# 验证安装：
python -c "import modelopt.torch.quantization as mtq; print('modelopt OK')"
```

### 量化范围

| 模块 | 精度 | 体积 | 说明 |
|---|---|---|---|
| encoder | INT8 (QDQ) | 337MB → 187MB | 全量化，CER≈0 |
| decoder | INT8 (QDQ) | 159MB → 112MB | 主 decoder 量化，**SeACo 热词路径保持 fp16** |
| cif | fp16（默认）/ INT8 (QDQ) | — | cumsum 数值敏感；trt_int8 时可 QDQ（cumsum 路径天然不量化） |
| bias_encoder | fp16（默认）/ INT8 (QDQ) | — | LSTM；trt_int8 时可 QDQ（精度需实测） |

> **线上推荐 `trt_int8_enc`**：仅 encoder int8，cif/decoder/bias fp16，CER≈0、热词精度保留。
> **`trt_int8`（4 段全 int8）**：cif/bias 也走 QDQ，显存最省，4 段 engine 已实测可正常运行，
> 但**精度损失较大**（cif cumsum 数值敏感 + bias LSTM 量化），**不推荐线上**，
> 仅在显存极度紧张且可接受精度下降时使用；追求精度请用 `trt_int8_enc`。
>
> decoder 全量化会破坏热词修正（"埃文"→"艾文"）。
> `export_decoder_qdq.py` 默认 `--exclude-patterns seaco_decoder hotword_output_layer`，
> 将 SeACo 路径排除在 INT8 外保持 fp16。
> cif QDQ（`export_cif_qdq.py`）需 fp16 encoder engine 生成校准输入；
> bias QDQ（`export_bias_qdq.py`）自包含，用词表编码 token 校准。

### 完整执行流程

```bash
# 校准数据：calib_data/audio_data 下放 16kHz 单声道 WAV（300 条）
# 注：QDQ 导出脚本统一加 --model-id ./models/asr/pt（本地 PT，避免联网下载）
#     涉及特征的脚本加 --cmvn-path ./models/asr/pt/am.mvn；bias 加 --tokens-path ./models/asr/pt/tokens.json

# 1. encoder QDQ 量化导出 + 转 engine
python scripts/export_encoder_qdq.py \
    --calib-data ./calib_data/audio_data \
    --model-id ./models/asr/pt \
    --cmvn-path ./models/asr/pt/am.mvn \
    --output ./models/asr/split/encoder_qdq.onnx
python scripts/convert_trt.py --input ./models/asr/split/encoder_qdq.onnx \
    --precision int8 --profile encoder \
    --output ./models/asr/trt/2080_ti_encoder_int8_qdq.engine

# 2. decoder QDQ 量化导出 + 转 engine（用 fp16 encoder+cif 生成校准输入）
python scripts/export_decoder_qdq.py \
    --encoder-engine ./models/asr/trt/2080_ti_encoder_fp16.engine \
    --cif-engine ./models/asr/trt/2080_ti_cif_fp16.engine \
    --model-id ./models/asr/pt \
    --cmvn-path ./models/asr/pt/am.mvn \
    --output ./models/asr/split/decoder_qdq.onnx
python scripts/convert_trt.py --input ./models/asr/split/decoder_qdq.onnx \
    --precision int8 --profile decoder \
    --output ./models/asr/trt/2080_ti_decoder_int8_qdq.engine

# 3.（可选，trt_int8 全 int8）cif + bias QDQ
python scripts/export_cif_qdq.py \
    --calib-data ./calib_data/audio_data \
    --encoder-engine ./models/asr/trt/2080_ti_encoder_fp16.engine \
    --model-id ./models/asr/pt \
    --cmvn-path ./models/asr/pt/am.mvn \
    --output ./models/asr/split/cif_qdq.onnx
python scripts/convert_trt.py --input ./models/asr/split/cif_qdq.onnx \
    --precision int8 --profile cif \
    --output ./models/asr/trt/2080_ti_cif_int8_qdq.engine
python scripts/export_bias_qdq.py \
    --hotword-file ./models/asr/hotwords.txt \
    --model-id ./models/asr/pt \
    --tokens-path ./models/asr/pt/tokens.json \
    --output ./models/asr/split/bias_encoder_qdq.onnx
python scripts/convert_trt.py --input ./models/asr/split/bias_encoder_qdq.onnx \
    --precision int8 --profile bias \
    --output ./models/asr/trt/2080_ti_bias_encoder_int8_qdq.engine

# 4. 数据集级 CER 评测（基准 fp16 vs 待测 int8，阈值 3%）
python scripts/evaluate_cer.py --audio-dir calib_data/audio_data \
    --config-dir ./models/asr/pt --csv report_cer.csv
```

### 诊断工具

```bash
# 查看 engine 各层精度分布（判断 INT8 是否真正生效）
python scripts/inspect_engine_precision.py --engine models/asr/trt/2080_ti_encoder_int8_qdq.engine
```

### 性能说明

- INT8 体积 encoder+decoder 合计 496MB → 299MB，显存占用大幅下降
- **2080 Ti（Turing）小 batch 下 INT8 因 Q/DQ 开销速度无明显提升**
- INT8 的速度价值在大 batch 吞吐 + Ampere/Hopper 架构（A10/T4/Orin），部署卡上预期有收益

### 待完成（TODO）

- 真实标注测试集复核 CER（当前以 fp16 输出为参考基准，偏乐观）
- cif/bias int8 QDQ 精度实测（cif cumsum 敏感、bias LSTM 量化支持有限，未达标回退 fp16）
- CER 超标时：decoder 额外排除 `src_attn`（`--exclude-patterns seaco_decoder hotword_output_layer src_attn`）
- 多 GPU engine 构建（各目标卡分别 build）

---

## 热词管理（两路分流 + 运行时热更新）

路由按**是否客户端主动传热词**分流（防通用识别误触发）：

```
请求到达
├─ 客户端传了 hotwords？
│   ├─ 是 → 截断 Top256 → 路径 A：SeACo 在线热词（每请求实时编码 bias_embed）
│   │        （客户端主动传 = 明确知道音频含这些词，激进增强合理）
│   └─ 否 → 默认词表恒走路径 B：普通 ASR + Faiss 后处理纠错
│            （通用识别多数音频不含默认热词，SeACo 会误纠相似音，Faiss 三重判定更稳）
```

| 路径 | 触发 | 处理 | bias_embed 来源 |
|---|---|---|---|
| A-客户端 | 传 hotwords | SeACo 模型内增强 | 每请求实时编码（含 Top256 截断） |
| B-默认 | 不传（默认词表） | 普通 ASR + Faiss 纠错 | 不用（bias=全零） |

> 设计依据：客户端主动传热词 = 明确该音频含这些词（垂直场景），用 SeACo 激进增强；
> 默认词表面向通用识别，绝大多数音频不含热词，SeACo 会把声学相似的普通词误纠成热词
> （如"神棚"→"沈鹏"），故默认词表恒走 Faiss——「检索命中 + 三重阈值才替换」的保守策略。

### 路径 A：SeACo 在线热词

| 项 | 选择 |
|---|---|
| 数量上限 / 切换点 MAX_HOTWORD_NUM | 256（客户端超限截断 Top256 + 告警；engine profile=256+1 含哨兵） |
| profile opt OPT_HOTWORD_NUM | 64 |
| ASF 过滤 NFILTER | 50 |
| 编码 | tokenizer（中文逐字 / 英文查 seg_dict BPE）+ `[sos]` 哨兵 → bias_encoder |
| 显存 | bias 维度 ≤256，engine profile max 固定不重建 |
| 触发 | 仅客户端主动传 `hotwords` 时（默认词表不再走此路径） |
| 默认词表优化 | 静态，启动预编码 bias_embed 缓存，命中默认路径直接复用 |

### 路径 B：Faiss 大词库后处理纠错

| 项 | 选择 |
|---|---|
| 触发 | 客户端不传热词（默认词表恒走此路径，不论大小） |
| ASR | 普通识别（SeACo bias=全零，无热词增强） |
| 热词表示 | 拼音向量 + 编辑距离辅助 |
| Faiss 索引 | `IndexFlatIP`（未来百万级可换 `IVFFlat`） |
| 检索粒度 | 滑窗片段，窗口 2/3/4 字 |
| TopK 召回 | 30 |
| 重排打分 | 拼音分数×0.75 + 编辑距离分×0.25 |
| 词库规模 | 1 万~20 万（`models/asr/hotwords.txt`） |

**三重联合判定（全满足才替换）：**

```python
if (top1.faiss_score > 0.85
    and (top1.faiss_score - top2.faiss_score) > 0.05
    and final_score > 0.88):
    replace()
```

### 词表热更新（运行时不中断，多 worker 安全）

部署形态：单机单容器、指定 GPU、`WORKERS=N`（默认 1，运维按显存调大）。所有 worker 进程共享容器本地文件 `models/asr/hotwords.txt`，无需挂载/NFS/K8s。

```
models/asr/hotwords.txt          词表内容（原子写）
models/asr/hotwords.txt.version  版本标记 {version, md5, count, route, updated_at}
models/asr/.hotwords.lock        跨进程互斥锁文件
```

更新流程：
1. `POST /hotwords/reload` 落到任一 worker → 取 flock 锁 → 校验链全过
2. `expected_version` CAS 防覆盖 → 原子写（temp → fsync → rename，commit point = version 文件 +1）
3. 本 worker 立即重建缓存 + 原子切换引用
4. 其他 worker 后台轮询 version（`HOTWORD_POLL_INTERVAL` 默认 5s）→ 发现变更各自重建 → 数秒内全局收敛（最终一致）

校验链（任一失败 → 丢弃新表，保留旧表）：UTF-8/去空白/去重/非空 → tokenizer 可编码(剔 OOV) → Faiss 索引构建（默认词表恒走路径 B）。

零中断原理：缓存是只读内存对象，后台构建新对象后用一行引用赋值切换（GIL 原子），在途请求用旧引用跑完，旧对象引用归零自动 GC。

### 热更新接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /hotwords/reload | body：新词表内容 或 `{"reload_from_file": true}`；返回校验结果 + 新 version |
| GET | /hotwords/status | 当前 version/md5/count/route/loaded_at（巡检各 worker 收敛） |
| POST | /hotwords/rollback | 回滚到上一版内容（发布为新 version） |

### 关键参数

| 参数 | 默认 | 路径 | 说明 |
|---|---|---|---|
| MAX_HOTWORD_NUM | 256 | A | SeACo 热词硬上限 / 路径切换点 |
| OPT_HOTWORD_NUM | 64 | A | TRT profile opt point |
| NFILTER | 50 | A | ASF 过滤注入数 |
| DEFAULT_HOTWORD_PATH | models/asr/hotwords.txt | A/B | 服务端默认词表 |
| HOTWORD_RELOAD_ENABLED | true | — | 是否开启热更新接口 |
| HOTWORD_POLL_INTERVAL | 5 | — | 各 worker 轮询 version 间隔（秒） |
| FAISS_WINDOW_SIZES | 2,3,4 | B | 滑窗大小 |
| FAISS_TOPK | 30 | B | 召回数 |
| FAISS_PINYIN_WEIGHT | 0.75 | B | 拼音权重 |
| FAISS_EDIT_WEIGHT | 0.25 | B | 编辑距离权重 |
| FAISS_SCORE_THRESHOLD | 0.85 | B | Faiss 检索分门槛 |
| GAP_THRESHOLD | 0.05 | B | top1-top2 区分度门槛 |
| FINAL_SCORE_THRESHOLD | 0.88 | B | 融合分门槛 |

### 核心优势

1. 显存上界恒定：路径 A（客户端热词）≤256，路径 B（默认词表）=0，与词库规模解耦
2. engine 永不重建：profile max=256 固定
3. 通用识别防误触发：默认词表恒走 Faiss 三重判定，避免 SeACo 把相似音误纠成热词
4. 大词库平滑扩展：`IndexFlatIP` → 未来百万级换 `IVFFlat`
5. 运行时热更新：多 worker 文件轮询收敛，零中断、可校验、可回滚
6. 职责分离：在线/小词表走模型，离线大词库走检索纠错

---

## v1 推理路径（ORT）

### 模型准备

```bash
# 整体导出 fp32 ONNX
python scripts/export_onnx_whole.py --output-dir ./models/asr

# fp32 → int8 动态量化（CPU 部署）
python scripts/convert_onnx_int8_dynamic.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/int8

# 下载 VAD 模型
python scripts/download_vad.py --output-dir ./models/vad
```

### 启动服务

```bash
# 本地启动
python -m uvicorn src.main:app --host 0.0.0.0 --port 8080

# Docker 启动（转换推理合一镜像）
docker-compose up -d
```

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| HOST_PORT | 8099 | 宿主机映射端口 |
| WORKERS | 1 | uvicorn workers；小 GPU 保持 1，大 GPU（≥24GB）可设 11 见 DEPLOY.md 模式 B |
| BATCH | 12 | 最大 batch size（合法值：1,2,4,8,12） |
| BATCH_TIMEOUT | 30 | batch 等待超时（毫秒），工业标准 dynamic batching 的 max_queue_delay |
| LOG_LEVEL | INFO | 日志级别 |
| MAX_CONCURRENT_REQUESTS | 2000 | 最大并发请求数 |
| MAX_AUDIO_DURATION_MS | 7200000 | 音频最大时长（ms），超出返回 1005；默认 2 小时，0=不限 |
| ACQUIRE_TIMEOUT | 5 | 过载并发等待超时（秒），超时返回 1007；0=无限等待 |
| MODEL_PRECISION | auto | 模型精度（见下表） |
| CPU_THREAD_POOL_SIZE | 0(自动全核) | CPU 流水线线程池（Stage1 VAD + Stage2 特征提取）；★对所有后端生效；多 worker 务必设小 |
| ORT_INTRA_OP_THREADS | 0(自动全核) | 主 ASR 单 session 算子并行线程数；**仅 CPU 后端（onnx_fp32/onnx_int8）生效**，TRT/GPU 与 VAD 均不生效 |
| ORT_INTER_OP_THREADS | 1 | 主 ASR session 间并行线程数；**仅 CPU 后端生效**，同上 |
| VAD_SESSION_POOL_SIZE | 4 | Silero VAD ORT session 池大小，round-robin 分配；VAD 单 session 线程数硬编码为 1，靠多 session 实现并行 |
| GPU_STREAM_POOL_SIZE | 4 | TRT 多 stream 多 execution_context 池；作用于 encoder/cif/decoder（启用时含 timestamp）；bias_encoder 固定单 context（低频调用不池化） |
| ENABLE_WORD_TIMESTAMP | false | 字级时间戳（asr[].words）；true 启用第 5 段 timestamp engine，按 GPU_STREAM_POOL_SIZE 池化，吞吐降约 30% |
| ENABLE_HOTWORD | true | 路径 A（SeACo 在线热词）总开关；false 时忽略客户端传入的 hotwords，不做 SeACo 增强 |
| ENABLE_FAISS_CORRECTION | true | 路径 B（默认词表 Faiss 纠错）总开关；false 时不构建/不运行 Faiss 后处理，通用识别零后处理开销 |
| OMP_NUM_THREADS | 1 | ★必须保持 1，防 libgomp 崩溃（run.sh 已固化） |
| MKL_NUM_THREADS | 1 | 同上 |
| OPENBLAS_NUM_THREADS | 1 | 同上 |

> 容器内部固定端口 8080，通过 HOST_PORT 映射到宿主机。
> **两种部署形态**（详见 DEPLOY.md 高并发性能调优）：
> - **模式 A 单进程**（默认）：WORKERS=1，任何硬件都能跑，QPS ~13
> - **模式 B 多进程**（大 GPU 生产）：WORKERS=11，QPS 93.92，需 24GB+ 显存
> 代码已按多 worker 安全设计（词表热更新经文件轮询跨 worker 收敛）。
> 每个 worker 独立加载 engine + CUDA context，显存占用随 WORKERS 线性增长。

### MODEL_PRECISION 取值

| 取值 | 后端 | 各段精度(enc/cif/dec/bias) | 说明 |
|---|---|---|---|
| auto | 自动 | — | GPU: trt_int8_enc→trt_fp16→trt_fp32→onnx_fp32；CPU: onnx_int8→onnx_fp32 |
| pt | PT | — | 原始 PyTorch 模型（转换环境用，服务回退 onnx_fp32） |
| onnx_fp32 | ORT | — | ONNX Runtime fp32（v1 整体模型） |
| onnx_int8 | ORT | — | ONNX Runtime int8 动态量化（CPU） |
| trt_fp32 | TRT | fp32/fp32/fp32/fp32 | 4 段全 fp32 |
| trt_fp16 | TRT | fp16/fp16/fp16/fp16 | 4 段全 fp16 |
| trt_int8 | TRT | int8/int8/int8/int8 | 4 段全 int8（QDQ）。实测可运行但**精度损失较大**，不推荐线上，仅显存极紧张时用 |
| **trt_int8_enc** | TRT | **int8/fp16/fp16/fp16** | **线上推荐**：encoder 显存减半，热词精度保留 |

单段精度可用环境变量覆盖（优先级最高）：`ENCODER_PRECISION` / `CIF_PRECISION` / `DECODER_PRECISION` / `BIAS_PRECISION`，取值 `fp32`/`fp16`/`int8`。

---

## API 示例

### curl

```bash
curl -X POST http://localhost:8099/chinese_asr \
    -H "Content-Type: application/json" \
    -d '{
        "base64": "'"$(base64 -w0 test.wav)"'",
        "article_url": "https://cdn.example.com/audio/test.wav",
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
    "http://localhost:8099/chinese_asr",
    json={
        "base64": b64_audio,
        "article_url": "https://cdn.example.com/audio/test.wav",  # 可选，透传到响应
        "hotwords": ["张三", "李四"],  # 可选
    },
)
print(response.json())
# {
#   "code": 0,
#   "article_url": "https://cdn.example.com/audio/test.wav",
#   "istar_asr": "...",
#   "asr": [
#     {"idx": 0, "slid": "", "text": "...", "speaker": "",
#      "timestamp": [0.0, 5.2],
#      "words": [{"text": "今", "timestamp": [0.12, 0.24]}, ...]}
#   ]
# }
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
│   └── load_model.py         # 加载本地 PT 权重（PT_MODEL_DIR，不依赖 FunASR）
├── src/                      # 推理服务源代码
│   ├── main.py               # FastAPI 入口（三级流水线 + 热词路由 + 热更新接口）
│   ├── config.py             # 精度矩阵 + batch/timeout + VAD/GPU 池 + 热词/Faiss 参数（单一数据源）
│   ├── errors.py             # 业务错误码
│   ├── schemas.py            # 请求/响应 schema
│   ├── logger.py             # 结构化日志（多 worker 按 PID 分文件）
│   ├── feature_extractor.py  # torchaudio kaldi fbank + LFR + CMVN
│   ├── tokenizer.py          # vocab8404 解码
│   ├── vad.py                # Silero VAD ONNX
│   ├── audio_segment.py      # 固定桶边界切分（桶边界从 config 派生）
│   ├── asr_engine.py         # ORT/TRT 双后端路由 + 热词编码
│   ├── trt_engine.py         # TensorRT 4 段串联推理引擎
│   ├── scheduler.py          # GPU Scheduler（bias-aware 分桶 + dynamic batch）
│   ├── hotword_manager.py    # 默认词表加载 + 预编码缓存 + 热更新
│   └── hotword_faiss.py      # 路径 B：拼音检索纠错
├── scripts/                  # 工具脚本
│   ├── prepare_model.py              # 启动编排：按精度检查/构建产物（核心）
│   ├── export_onnx_whole.py          # PT → 整体 ONNX（onnx_fp32）
│   ├── export_onnx_split.py          # PT → 分段 ONNX（trt 系列源，含 --clamp-value）
│   ├── export_encoder_qdq.py         # encoder QDQ INT8 量化导出
│   ├── export_cif_qdq.py             # cif QDQ INT8 量化导出（trt_int8 用）
│   ├── export_decoder_qdq.py         # decoder QDQ INT8 量化导出
│   ├── export_bias_qdq.py            # bias_encoder QDQ INT8 量化导出（trt_int8 用）
│   ├── export_encoder_truncated.py   # encoder 截断实验（保留供后续优化）
│   ├── export_decoder_truncated.py   # decoder 截断实验（保留供后续优化）
│   ├── convert_trt.py                # ONNX → TRT engine 转换（fp32/fp16/int8）
│   ├── convert_onnx_int8_dynamic.py  # ONNX → int8 动态量化（onnx_int8，CPU）
│   ├── verify_onnx.py                # ONNX vs PT 精度验证
│   ├── inspect_onnx_structure.py     # ONNX 模型结构检查
│   ├── inspect_engine_precision.py   # TRT engine 层精度诊断
│   ├── evaluate_cer.py               # 数据集级 CER 批量评测
│   ├── download_vad.py               # VAD 模型下载
│   └── entrypoint.sh                 # 镜像启动脚本（prepare_model → 启动服务）
├── tests/                    # 测试脚本
│   ├── test_pt_inference_v2.py       # PT baseline（独立包推理）
│   ├── test_split_onnx_pipeline.py   # ORT 分段串联推理
│   ├── test_trt_pipeline.py          # TRT 分段推理（各部分独立精度）
│   ├── test_model.py                 # 整体 ONNX 推理
│   ├── test_service.py               # 服务压测
│   ├── test_single.py                # 单次请求测试
│   ├── test_asr_api.py               # HTTP ASR API 测试
│   ├── test_hotword_api.py           # 热词热更新接口测试
│   ├── test_error_api.py             # 错误码路径测试
│   └── test_vad.py                   # VAD 单独测试
├── models/                   # 模型文件（不纳入版本控制）
│   ├── asr/
│   │   ├── am.mvn / tokens.json / config.yaml / hotwords.txt
│   │   ├── pt/               # PT 权重（提前打包，PT_MODEL_DIR）
│   │   ├── fp32/             # 整体 ONNX fp32（onnx_fp32）
│   │   ├── int8/             # 整体 ONNX int8（onnx_int8）
│   │   ├── split/            # 分段 ONNX（encoder/cif/decoder/bias_encoder + *_qdq）
│   │   └── trt/              # TRT engine（GPU 绑定，直接打包进镜像）
│   └── vad/silero_vad.onnx
├── docs/
│   ├── README.md             # 本文件
│   ├── API.md                # API schema
│   ├── DEPLOY.md             # 部署文档
│   └── CONTRIBUTING.md       # 贡献指南
├── logs/                     # 服务日志（按天轮转，保留 7 天，多 worker 按 PID 分文件）
├── Dockerfile                # 推理镜像（TRT 10.6 + CUDA 12.6，转换推理合一）
├── docker-compose.yml        # 服务编排
├── step_run.sh               # 完整转换流程示例
└── requirements-infer.txt    # 推理 + 转换统一依赖
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

### GPU Scheduler 调度策略（工业标准 dynamic batching）

**Uniform Chunking 均匀切段**（`audio_segment.py`）：
- VAD 后取整体时间跨度 [first.start_ms, last.end_ms]，按 UNIFORM_CHUNK_MS
  （=TRT_OPT_SEQ × 60 = 4020ms）均匀切分，所有 chunk 长度统一到 opt=67 帧
- 尾块 <MIN_TAIL_MS（1000ms）合并到前段，避免过短尾块识别精度损失
- 消除桶分组把 chunk 摊薄到 3 个独立队列的合批瓶颈

**Scheduler 触发逻辑**（`scheduler.py`）：
- 按 bias 身份分组（避免跨请求热词串扰），去掉桶维度分组
- **满触发**：group 累计 ≥ MAX_BATCH_SIZE (=12) 立即推理
- **超时触发**：按最早入队 chunk 的 enqueue_time 计时，超过 BATCH_TIMEOUT 兜底
- 1ms 高精度 tick 扫描，保证严格延迟上限
- 触发后剩余 chunk 保留 group，enqueue_time 不变，下轮继续判定

**GPU 多 stream 多 execution_context**（`trt_engine.py`）：
- 每段 engine 一份 weights + GPU_STREAM_POOL_SIZE (=4) 个 context + stream
- 无锁 round-robin 分配 (context, stream) 槽位
- `_gpu_executor` max_workers=GPU_STREAM_POOL_SIZE，多 batch 真正并行 GPU

**Batch 组装**：
- submit 时不 pad，_execute_batch 按 batch 内 max(lengths) 动态 pad
- pad 到最近合法 batch size (1/2/4/8/12)，dummy pad 用 batch[-1] 复制
- OOM Fallback：减半 batch 重试 → 逐条推理 → 返回 `ASR_INFER_FAILED` 错误
