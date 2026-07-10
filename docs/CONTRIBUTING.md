# 贡献指南

## 模型更新流程

当需要更新 ASR 模型版本或重新导出 ONNX/TRT engine 时，参考本指南。

### 1. 模型代码框架

项目内置完整的 SeACo-Paraformer 模型定义（`seaco_paraformer/` 目录），**不依赖 FunASR 运行时**：

```
seaco_paraformer/
├── __init__.py          # 包入口
├── model.py             # SeacoParaformer 主模型
├── encoder.py           # SANMEncoder + EncoderLayerSANM（含 clamp_value 参数）
├── decoder.py           # ParaformerSANMDecoder + DecoderLayerSANM
├── predictor.py         # CifPredictorV3 + cif / cif_v1_export（向量化，TRT 兼容）
├── attention.py         # SANM Self-Attention / Cross-Attention（FSMN）
├── layers.py            # LayerNorm / FFN / SinusoidalPositionEncoder
├── utils.py             # MultiSequential / repeat / make_pad_mask
└── load_model.py        # 加载本地 PT 权重（PT_MODEL_DIR）
```

PT 权重存放于 `models/asr/pt/`（默认 `PT_MODEL_DIR`，扁平结构：model.pt / config.yaml /
am.mvn / tokens.json / seg_dict）。推荐提前打包进镜像；若缺失，`prepare_model.py`
（ensure_pt）会自动调 `scripts/download_asr.py`（ModelScope HTTP 直链，扁平下载）拉取，
也可手动运行：

```bash
python scripts/download_asr.py --output-dir models/asr/pt   # 已存在则跳过
```

### 2. 启动编排（推荐方式）

线上不手动逐条执行转换命令，统一由 `prepare_model.py` 按 `MODEL_PRECISION`
检查产物，缺失则从本地 PT 权重按依赖链逐级转换：

```bash
# 检查 + 按需构建（容器 entrypoint.sh 自动调用）
python scripts/prepare_model.py --precision trt_int8_enc

# 仅检查不构建
python scripts/prepare_model.py --precision trt_fp16 --check-only
```

依赖链：
```
PT 权重 → 分段 ONNX（export_onnx_split.py）→ TRT fp32/fp16 engine（convert_trt.py）
                                          → QDQ ONNX（export_{encoder,cif,decoder,bias}_qdq.py）→ TRT int8 engine
       → 整体 ONNX（export_onnx_whole.py）→ int8 动态量化（convert_onnx_int8_dynamic.py）
```

### 3. 手动导出 + 转换（开发调试）

完整流程见 `step_run.sh`，关键命令：

```bash
# 分段 ONNX 导出（opset 17 + clamp 60000，fp16 关键）
python scripts/export_onnx_split.py --output-dir ./models/asr/split --clamp-value 60000

# TRT fp16（纯 fp16，无 fp32 fallback）
python scripts/convert_trt.py --input ./models/asr/split/encoder.onnx --precision fp16 --profile encoder
python scripts/convert_trt.py --input ./models/asr/split/cif.onnx --precision fp16 --profile cif
python scripts/convert_trt.py --input ./models/asr/split/decoder.onnx --precision fp16 --profile decoder
python scripts/convert_trt.py --input ./models/asr/split/bias_encoder.onnx --precision fp16 --profile bias

# 整体 ONNX（v1 ORT 路径）+ int8 动态量化（CPU）
python scripts/export_onnx_whole.py --output-dir ./models/asr --skip-fp16
python scripts/convert_onnx_int8_dynamic.py --input-dir ./models/asr/fp32 --output-dir ./models/asr/int8

# VAD 模型
python scripts/download_vad.py --output-dir ./models/vad
```

### 4. 精度验证

```bash
# PT baseline（独立包推理）
python tests/test_pt_inference_v2.py --audio test_data/audio_16000_10s.wav --hotwords 埃文 账号

# ORT 分段串联验证
python tests/test_split_onnx_pipeline.py --audio test_data/audio_16000_10s.wav --device cuda --hotwords 埃文 账号

# TRT 分段验证（各段独立精度）
python tests/test_trt_pipeline.py --audio test_data/audio_16000_10s.wav \
    --encoder-precision fp16 --cif-precision fp16 \
    --decoder-precision fp16 --bias-precision fp16 --hotwords 埃文 账号

# 数据集级 CER 评测（基准 fp16 vs 待测 int8）
python scripts/evaluate_cer.py --audio-dir calib_data/audio_data --csv report_cer.csv
```

---

## v2 TRT 分段模型架构

模型拆分为 4 段主模型 + 1 段可选（含热词与字级时间戳支持）：

| 子模型 | 功能 | 推荐精度 | 说明 |
|--------|------|----------|------|
| encoder.onnx | 语音编码 | fp16 | opset 17 + clamp 60000，纯 fp16 不崩溃 |
| cif.onnx | CIF 预测器 | fp16 | 向量化实现（cumsum+bmm），TRT 兼容 |
| decoder.onnx | 解码器+SeACo | fp16 | 含 ASF + SeACo decoder + 热词合并 |
| bias_encoder.onnx | 热词编码器 | fp16 | LSTM 编码热词 token IDs |
| timestamp.onnx | 字级时间戳 | fp16 | upsample CIF head + blstm，ENABLE_WORD_TIMESTAMP 开关，默认不加载 |

> timestamp 段独立设计原因：upsample+blstm 计算量大，并入 CIF 会拖累吞吐
> （实测 2800→2000 req/s）。拆为独立 engine + 环境开关，不启用时零成本。
> 字级时间戳算法对齐 FunASR ts_prediction_lfr6_standard（见 src/timestamp.py）。

## 三后端功能支持矩阵

三个推理后端（TRT / ORT / PT）均支持字级时间戳、热词（路径 A SeACo）、Faiss（路径 B），
且均由环境变量参数控制开关，功能对齐。

| 后端 | MODEL_PRECISION | 字级时间戳 | 热词 A | Faiss B | 说明 |
|------|-----------------|:----------:|:------:|:-------:|------|
| TRT | trt_fp16 / trt_fp32 / trt_int8 / trt_int8_enc | ✅ 第 5 段 engine | ✅ | ✅ | GPU 生产主路径；timestamp 仅 fp16/fp32（BLSTM 不量化） |
| ORT | onnx_fp32 / onnx_int8 | ✅ 分段串联 | ✅ | ✅ | 时间戳**开**→分段 ONNX 串联（encoder→cif→decoder+bias+timestamp）；**关**→整体模型兜底 |
| PT | pt | ✅ predictor 内置 | ✅ | ✅ | 原生 PyTorch，GPU 优先/CPU 兜底；无需转换，适合验证/无 TRT 环境 |

开关（三后端通用）：
- `ENABLE_WORD_TIMESTAMP`（默认 false）：字级时间戳（asr[].words）
- `ENABLE_SENTENCE_TIMESTAMP`（默认 false）：句子级时间戳（asr[] 粒度变为句）
- `ENABLE_HOTWORD`（默认 true）：客户端热词 SeACo 在线增强（路径 A）
- `ENABLE_FAISS_CORRECTION`（默认 true）：默认词表 Faiss 后处理纠错（路径 B）

### 句子级时间戳（后端无关的结果后处理）

字级时间戳是模型（各后端 engine/predictor）输出；**句子级时间戳是纯文本后处理**，
与推理后端解耦，三后端行为一致：
- 对全文跑 ngram 标点模型（KenLM n-gram + Qwen2.5 BPE，纯 CPU，`src/sentence_segmenter.py`）
  恢复标点并按句末标点断句，再用**已有字级时间戳**定位每句 [start, end]，asr[] 每项变为一句话。
- **强依赖 `ENABLE_WORD_TIMESTAMP=true`**：句子边界靠字级时间戳定位；未开字级时间戳时
  服务启动告警并自动降级回段级输出（asr[] 保持 VAD 切段形态）。
- 模型扁平存放于 `PUNC_MODEL_DIR`（默认 `models/punc`：prune*.bin / vocab.json / merges.txt），
  缺失时 `prepare_model.py`（ensure_punc）或加载阶段自动调 `scripts/download_punc.py`（HTTP 直链）下载。
- 生产参数固化实测最优：`PUNC_NGRAM_ORDER=3` + `PUNC_CANDIDATES=，。？` + `PUNC_PPL_DROP_RATIO=0.12`
  （网格实测见 `scripts/benchmark_punctuator.py`：中文候选子集较全集快 3-4x，中英混合断句
  位置不受影响、标点风格统一为中文）。
- 内存：每 worker 独立加载一份标点模型（prune*.bin ~253MB），多进程（WORKERS）随 worker 数线性增长。
- 分句为 CPU 密集（KenLM 打分），在结果合并环节走 CPU 线程池执行，不阻塞事件循环。

要点：
- **ORT 时间戳依赖分段 ONNX**：`onnx_fp32 + ENABLE_WORD_TIMESTAMP=true` 时 prepare_model
  会自动确保 `models/asr/split/*.onnx`（含 timestamp）存在；整体模型 model.onnx 无法输出时间戳。
- **onnx_int8 + 时间戳的降级**：分段 ONNX 只有 fp32 产物（无 int8 量化分段），故
  `onnx_int8 + ENABLE_WORD_TIMESTAMP=true` 会静默切到 **fp32 分段串联**，失去 int8 的
  体积（-75%）/速度特性（启动日志有告警）。需 int8 体积优势请关闭字级时间戳。
- **TRT timestamp 精度**：跟随 profile（trt_fp32→fp32，其余→fp16），可用 `TIMESTAMP_PRECISION` 覆盖；int8 非法自动回退 fp16。
- **PT timestamp 无额外产物**：直接调用 `predictor.get_upsample_timestamp`，只需 PT 权重。

> 历史问题：早期 encoder/decoder 全 fp16 会精度崩溃（残差 Add 溢出 inf）。
> 现已通过 **opset 17 原生 LayerNormalization + encoder 残差 Add clamp 60000**
> 实现纯 fp16，无需任何 fp32 fallback（详见 docs/README.md）。

### 纯 fp16 三大关键技术

1. **opset 17 LayerNormalization 单节点**：TRT 10.6 内部对该算子自动 fp32 累加
2. **encoder 残差 Add 后 clamp 60000**：后段层激活峰值 ~48万 >> fp16 上限 65504，60000 贴近上限最大化保留
3. **纯 trtexec --fp16**：无需 Python TRT API / OBEY_PRECISION_CONSTRAINTS / 手动 fallback

### 已知问题与排查经验（重要）

> 以下是分段 ONNX/TRT 导出中踩过的坑，重新导出或改模型代码时务必注意。

**1. encoder 残差激活峰值高达 ~48万，clamp 值直接影响精度**

- encoder 后段层（约 29-49 层）残差累积激活峰值达 **~48万**（用
  `tests/test_pt_inference_v2.py --dump-act` 实测），远超 fp16 上限 65504。
- fp16/int8 路径**必须 clamp**（否则溢出 inf）；fp32/ORT 路径**必须不 clamp**
  （`--clamp-value 0`，否则 Clip 算子引入误差）。
- clamp 值取舍：**60000**（贴近 fp16 上限，仅极少数峰值点被裁，CER 影响极小）。
  早期 30000 把后段 ~48万 狠裁到 3万，导致 **VAD 切段产生的非满桶输入**（如 125 帧）
  解码中段重复乱码——整段 167 帧因边界帧占比小不易暴露，极具迷惑性。

**2. SinusoidalPositionEncoder 不能用 `torch.arange(timesteps)` 构造 position**

- `torch.arange(1, x.size(1)+1)` 在 ONNX trace 时会把 timesteps 固化成 dummy 长度常量，
  非 dummy 长度输入时位置编码错位 → encoder_out 系统性偏差。
- 已改用 `cumsum(ones_like(x))` 生成动态长度的 position 序列（`layers.py`）。

**3. cif_v1_export 必须与 PT `cif`（for 循环软分配）数学等价**

- 旧实现用 `floor(cum/thr)` one-hot 硬分配，把跨 token 边界的帧整帧分给单个 token，
  丢失边界拆分；短/截断输入下 acoustic 偏差累积致解码乱码。
- 已改用**区间重叠软分配**（帧累积区间 [cum-α, cum) 与 token 区间 [j·thr,(j+1)·thr)
  的重叠长度，min/max/clamp+bmm，无 Loop/scatter，TRT 兼容）。
- `python -m seaco_paraformer.predictor` 自检：新版 vs PT 误差 1e-5，ORT 导出保真。

**4. token_num 取整用 round 不是 int（floor）**

- PT 用 `torch.round(alphas.sum())`，下游脚本/服务取 token_num 也必须 round，
  用 `int()`（截断）会在小数 >0.5 时少算 1 个 token，导致 decoder 对齐错位。

**5. 逐层定位工具**

- `tests/test_split_onnx_pipeline.py --compare-pt`：逐段对比 PT vs ONNX
  （encoder_out / token_num / acoustic / 最终 argmax 分叉位置 + 交叉验证）。
- `scripts/export_encoder_truncated.py --num-layers N --compare`：导出前 N 层 encoder
  并对比 PT vs ORT，二分定位误差出现的层。
- `tests/test_pt_inference_v2.py --dump-act`：dump 每层残差激活峰值（定 clamp 下限）。

### INT8 量化（QDQ Explicit）

- 量化库：`nvidia-modelopt==0.21.0`（必须钉版本，0.44+ 破坏 torch 环境）
  - 隐性依赖：`pulp`、`regex`（缺失报误导性的 "Please install optional [torch] dependencies"）
  - 安装：`pip install nvidia-modelopt==0.21.0 torchprofile pulp regex --extra-index-url https://pypi.nvidia.com`
  - 验证：`python -c "import modelopt.torch.quantization as mtq; print('modelopt OK')"`
- 方案：QDQ Explicit（插入 Q/DQ 节点显式标记量化边界），Calibrator Implicit 在 SeACo 架构上无效
- encoder QDQ：`export_encoder_qdq.py`；decoder QDQ：`export_decoder_qdq.py`（默认排除 SeACo 路径保 fp16）
- cif QDQ：`export_cif_qdq.py`（trt_int8 用）；bias QDQ：`export_bias_qdq.py`（trt_int8 用）
- 所有 QDQ 导出脚本需传 `--model-id ./models/asr/pt`（本地 PT 目录，避免联网下载）；
  涉及特征的脚本（encoder/cif/decoder）还需 `--cmvn-path ./models/asr/pt/am.mvn`，
  bias 脚本需 `--tokens-path ./models/asr/pt/tokens.json`（配置文件实际在 `models/asr/pt`）

### 热词推理流程

```
hotword_ids → bias_encoder → hw_embed → 按长度取最后时间步 → bias_embed (1, H, 512)
                                                                    ↓
speech → encoder → cif → acoustic_embeds + encoder_out + bias_embed → decoder+SeACo → logits
```

SeACo 内部：
1. 主 decoder → logits + hidden
2. ASF（注意力分数过滤）→ top-NFILTER(50) 热词
3. SeACo decoder × 2（query=acoustic_embeds / hidden，memory=filtered_hotwords）
4. merged → hotword_output_layer → dha_logits
5. NO_BIAS mask 合并：`logits * mask + dha_logits * (1-mask)`

### 热词维度规格（bias_encoder 输出 ↔ decoder 输入对齐）

数据流维度（H=热词数含哨兵，L=hw_len token 长度，D=512）：

```
bias_encoder 输入 hotword:  (num_hotwords=H, hw_len=L)
bias_encoder 输出 hw_embed: (hw_len=L, num_hotwords=H, D)   ← L 在前
  → 中间处理：transpose→(H,L,D)→按热词长度取最后时间步→ bias_embed (1, H, D)
decoder 输入 bias_embed:    (batch=1, num_hotwords=H, D)    ← decoder 用 H 作热词数
```

两个维度的单一数据源（`src/config.py`）：

| 维度 | 参数 | profile opt / max | 说明 |
|---|---|---|---|
| 热词数量 num_hotwords | MAX_HOTWORD_NUM=256, OPT_HOTWORD_NUM=64 | opt=64 / max=257 | +1 为 `[sos]` 哨兵行 |
| 热词长度 hw_len | MAX_HOTWORD_LEN=16 | opt=4 / max=16 | 中文逐字 / 英文 seg_dict BPE 后可达 ~12 |

- bias profile（`hotword`）与 decoder profile（`bias_embed`）的数量维度都用 opt=64/max=257，**全程一致**。
- 各导出脚本 dummy 统一 num_hotwords=4、轴名统一 `num_hotwords`/`hw_len`（均动态轴示例值，不限制 engine 范围）。
- bias QDQ 校准长度 `--calib-hw-len` = 16（与 MAX_HOTWORD_LEN 对齐，覆盖长英文热词）。

### 英文热词支持（seg_dict BPE）

- `tokenizer.encode` 中英混合切分：中文逐字最长匹配；英文单词查 `models/asr/pt/seg_dict`
  （FunASR 官方英文 BPE 表，31 万词）得正确 subword 序列，未命中 fallback 贪心匹配。
- `tokenizer.load` 自动探测同目录 `seg_dict`，缺失时降级为纯贪心（向后兼容）。
- 中文热词逐字命中、encode↔decode 字符级一致，路径不变。

Engine 产物（按 GPU 命名）：
```
models/asr/trt/
├── 2080_ti_encoder_fp16.engine
├── 2080_ti_cif_fp16.engine
├── 2080_ti_decoder_fp16.engine
├── 2080_ti_bias_encoder_fp16.engine
└── {gpu}_{module}_int8_qdq.engine   # int8 QDQ 产物
```

> **注意**：TRT engine 与 GPU 硬件绑定，不同 GPU 需分别构建。

### engine 层精度诊断

```bash
# 查看 engine 各层精度分布（判断 INT8 是否真正生效）
python scripts/inspect_engine_precision.py --engine models/asr/trt/2080_ti_encoder_int8_qdq.engine
```

---

## 代码规范

- Python 文件名：英文小写，下划线分隔
- 文档和日志：中文
- 不自动生成测试文件
- 未明确要求创建新文件时，在原文件上修改
- 涉及中文内容的文件编辑只用代码编辑工具（避免 shell 文本替换导致编码损坏）

## 目录结构

| 目录 | 用途 |
|------|------|
| seaco_paraformer/ | 模型代码框架（独立，不依赖 FunASR 运行时） |
| src/ | 服务源代码 |
| scripts/ | 工具脚本（导出、转换、量化、评测、编排） |
| models/ | 模型文件（不纳入 Git） |
| configs/ | 配置文件 |
| docs/ | 文档 |
| logs/ | 运行日志（按天轮转，多 worker 按 PID 分文件） |
| tests/ | 测试代码 |
