######################### 任务 ########################################

模型地址：
    ASR: https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
    VAD: https://modelscope.cn/models/pengzhendong/silero-vad

核心目标：构建工业级 SeACo-Paraformer 推理服务，具备：
    1. 高吞吐（30s 音频单卡 QPS 90+）
    2. 热词定制（在线 SeACo + 大词库 Faiss 后处理双路）
    3. 精度稳定（trt_fp16/trt_int8_enc 与 PT baseline 字符级一致）
    4. Docker 单镜像交付（转换 + 推理合一）

######################### 技术路径 ####################################

# 一、模型转换（一次性完成，产物打包进镜像/volume）

架构：4 段主 engine + 1 段可选
    - encoder.engine       — speech → encoder_out
    - cif.engine           — encoder_out + mask → acoustic_embeds, token_num
    - decoder.engine       — 主 decoder + SeACo + ASF 热词合并 → logits
    - bias_encoder.engine  — hotword_ids → hw_embed（供 decoder bias 注入）
    - timestamp.engine     — encoder_out + mask + token_num → us_alphas/us_cif_peak
                             （字级时间戳，ENABLE_WORD_TIMESTAMP 开关，默认关闭）

关键技术（详见 docs/CONTRIBUTING.md v2 TRT 分段模型架构）：
    - opset 17 LayerNormalization 单节点（TRT 10.6 内部 fp32 累加）
    - encoder 残差 Add clamp=60000（贴近 fp16 上限，避免溢出）
    - CIF 向量化实现 cif_v1_export（cumsum + one-hot + bmm，无 Loop 算子）
    - 纯 trtexec --fp16 转换，无需 fp32 fallback
    - INT8 走 QDQ Explicit（nvidia-modelopt 0.21.0），decoder 排除 SeACo 路径

# 二、推理架构（三级流水线 + 工业标准 dynamic batching）

Stage 1 (CPU 线程池)  音频解码 + VAD + 均匀切段
Stage 2 (CPU 线程池)  fbank 特征提取（torchaudio.compliance.kaldi）
Stage 3 (GPU Scheduler) TRT 4 段串联推理

设计原则：
    - CPU 与 GPU 同时满载（Stage 1/2 独立线程池，Stage 3 GPU 池化）
    - 请求间流水线互不阻塞
    - 单请求延迟 ≈ max(Stage1, Stage2, Stage3)

## 音频切段：Uniform Chunking（去桶分组）

audio_segment.py 策略：
    - VAD 后取整体时间跨度 [first.start_ms, last.end_ms]
    - 按 UNIFORM_CHUNK_MS (=TRT_OPT_SEQ × 60 = 4020ms) 均匀切分
    - 尾块 <MIN_TAIL_MS (1000ms) 合并到前段，避免过短尾块识别精度损失
    - 所有 chunk 长度统一到 opt=67 帧附近，最长 ~84 帧
    - engine profile 上下界仍保留 [1, 134] 兜底

## GPU Scheduler：工业标准 dynamic batching

- 分组键 = (0, bias_key)：只按 bias 身份分组，避免热词串扰
- **满触发**：group 累计 >= MAX_BATCH_SIZE(=12) 立即推理
- **超时触发**：按最早入队 chunk 的 enqueue_time 计时，>= BATCH_TIMEOUT 兜底
- 1ms 高精度 tick 扫描，保证严格延迟上限
- Batch 组装：submit 不 pad，_execute_batch 按 batch 内 max(lengths) 动态 pad
- 合法 batch size (1/2/4/8/12)，dummy pad 用 batch[-1] 复制
- OOM Fallback：减半 batch 重试 → 逐条推理 → 返回错误

## GPU 池化：多 stream 多 execution_context

trt_engine._TRTInferencer 池化设计：
    - 一份 engine（weights 共享）+ N 个 execution_context + N 个 CUDA stream
    - N = GPU_STREAM_POOL_SIZE（默认 4）
    - queue.Queue 连接池借还 (context, stream)（保证同一 context 不被两线程并发 execute，
      避免 CUDA illegal memory access；v2.3.1 从 itertools.count 改造）
    - _gpu_executor 扩到 max_workers=N，多 batch 并发提交到不同 stream

## VAD Session 池

vad.py Silero VAD ONNX 推理：
    - Session 池 round-robin 分配（VAD_SESSION_POOL_SIZE，默认 2，实测最优）
    - session options 禁用 arena/mem_pattern（避免 ORT 并发竞态）
    - 固定 CPUExecutionProvider（GPU VAD 实测慢 4x，PCIe 传输开销压垒 GPU 计算）

# 三、精度矩阵

    | MODEL_PRECISION | 各段(enc/cif/dec/bias) | 显存 | 说明 |
    | onnx_fp32       | fp32/fp32/fp32/fp32   | 大   | CPU/GPU 兜底 |
    | onnx_int8       | int8 动态量化          | 小   | CPU 部署首选 |
    | trt_fp32        | fp32/fp32/fp32/fp32   | 大   | GPU 无损基线 |
    | trt_fp16        | fp16/fp16/fp16/fp16   | 中   | GPU 生产推荐（含 clamp 60000）|
    | trt_int8_enc    | int8/fp16/fp16/fp16   | 中偏小 | ★线上推荐（CER≈0，显存减半）|
    | trt_int8        | int8/int8/int8/int8   | 小   | 全 QDQ（精度损失较大，仅备选）|

# 四、热词管理（两路分流）

按"是否客户端主动传热词"路由（防通用识别误触发）：
    - 客户端传 hotwords → 路径 A（SeACo 在线）：截断 Top256 → 实时编码 bias_embed
      （客户端主动传 = 明确知道音频含这些词，激进增强合理）
    - 客户端不传 → 默认词表恒走路径 B（Faiss 后处理纠错）
      （通用识别多数音频不含默认热词，SeACo 会把声学相似普通词误纠成热词，
        Faiss 三重判定仅在拼音+编辑距离高度吻合时替换，大幅降低误触发）

设计变更（防误触发）：默认词表不再按大小走 A，恒走 B。
    早期"默认词表 ≤256 走 A"在通用识别场景导致相似音被误纠
    （如"神棚"→"沈鹏"），故默认词表统一走保守的 Faiss。

模块开关（按需裁剪推理路径）：
    - ENABLE_HOTWORD（默认 true）：路径 A 总开关，false 忽略客户端 hotwords
    - ENABLE_FAISS_CORRECTION（默认 true）：路径 B 总开关，false 不构建/不运行 Faiss
    - 纯通用识别追求极限吞吐可两者全关；也可单独关一路按需组合

路径 A（SeACo）：
    - MAX_HOTWORD_NUM=256（硬上限，engine bias profile max=257 含哨兵）
    - OPT_HOTWORD_NUM=64（TRT profile opt point）
    - NFILTER=50（ASF 过滤 top-K）
    - MAX_HOTWORD_LEN=16（英文 BPE 后可达 ~12）
    - tokenizer 中英混合：中文逐字，英文查 seg_dict BPE
    - decoder 内部：主 logits + ASF top51 热词 + SeACo × 2 + hotword_output_layer
    - NO_BIAS mask 合并：logits * mask + dha_logits * (1 - mask)

路径 B（Faiss）：
    - 拼音向量（multi-hot 音节 + L2 归一化）+ IndexFlatIP
    - 滑窗 2/3/4 字，TopK=30
    - 三重判定：faiss>0.85 且 top1-top2>0.05 且 final>0.88

词表热更新（多 worker 安全）：
    - POST /hotwords/reload：flock + expected_version CAS + 原子写 + 版本轮询收敛
    - GET  /hotwords/status
    - POST /hotwords/rollback
    - 校验链：UTF-8 → 去重 → tokenizer 可编码 → 试跑 bias_encoder 验 nan/inf

# 五、部署形态（两种模式）

模式 A：单进程（默认，兼容任何硬件）
    WORKERS=1，实测 QPS 13.15（conc=20，2080 Ti），显存 ~1.5GB

模式 B：多进程高并发（大 GPU 生产）
    WORKERS=11，实测 QPS 93.92（conc=120，A10+），显存 ~15-20GB
    大 GPU 上多进程隔离 CPU 竞争 > CUDA context 切换开销

必须固化的环境变量（run.sh/compose/config 默认，A10 24GB 最优）：
    OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1
      （防 libgomp 崩溃，Silero VAD 是串行 LSTM，OMP 内并行无收益）
    WORKERS=11（模式 B；★小显存改回 1）
    CPU_THREAD_POOL_SIZE=32（256 核最优，per-worker）
    VAD_SESSION_POOL_SIZE=2（实测最优）
    GPU_STREAM_POOL_SIZE=4
    BATCH_TIMEOUT=10（实测吞吐最优）
    MODEL_PRECISION=trt_fp16（部署默认；config 兜底 auto）

# 六、API 规约（POST /chinese_asr）

输入：{"base64": "wav_16k_1channel_base64", "article_url": "https://...", "hotwords": ["张三", "李四"]}
    - base64 必填；article_url、hotwords 可选

成功：{"code": 0, "article_url": null|str, "istar_asr": "全文",
       "asr": [{"idx": 0, "slid": "", "text": "...", "speaker": "",
                "timestamp": [start_s, end_s],
                "words": [{"text": "字", "timestamp": [start_s, end_s]}, ...]}]}
    - slid（语种）、speaker（说话人）当前未实现，固定空字符串
    - 段级 timestamp 源自 VAD 时间轴
    - 字级 words 由 CIF alphas 反推得到（需 CIF engine 输出 alphas），粒度约 60ms
    - 旧版 engine 或 ORT 整体模型不支持字级时间戳，words:[] 空数组

失败：{"code": 1001, "article_url": null, "istar_asr": "", "asr": [],
       "error": "DECODE_FAILED", "message": "..."}

字级时间戳（asr[].words）：
    - ENABLE_WORD_TIMESTAMP 开关控制（默认 false，吞吐优先）
    - 独立第 5 段 timestamp engine（upsample CIF head + blstm），对齐 FunASR
      ts_prediction_lfr6_standard：相邻 fire 中点划界（不重叠）+ 超长截断 + 静音扣除
    - 精度约 20ms（upsample 3x），启用后吞吐降约 30%（2800→2000 req/s）
    - 关闭或 ORT 模型：words 为空数组

istar_asr 段间用逗号分隔（各段为 VAD 切段单位，非完整句），便于阅读。

错误码：
    SUCCESS=0 / INPUT_PARAM_FAILED=1000 / DECODE_FAILED=1001 /
    VAD_SEGMENT_ERROR=1002 / AUDIO_SEGMENT_ERROR=1003 / ASR_INFER_FAILED=1004 /
    AUDIO_TOO_LONG=1005 / MODEL_LOAD_FAILED=1006 / SERVICE_BUSY=1007 /
    HOTWORD_VERSION_CONFLICT=1008

HTTP status 映射：
    1000/1001/1005 → 400   1002/1003/1004/1006 → 500   1007 → 503   1008 → 409

######################### 版本进展 ####################################

v1（已完成）：
    - ONNX fp32 + int8 动态量化
    - FastAPI 三级流水线（桶分组 scheduler）
    - Docker 双镜像方案 → 统一为转换推理合一

v2 阶段 1（已完成，见 CONTRIBUTING.md v2 TRT 分段模型架构）：
    - 4 段独立 engine 分段架构
    - opset 17 + clamp 60000 + 纯 fp16
    - 热词维度对齐（bias 输出 ↔ decoder 输入）

v2 阶段 2（已完成）：
    - INT8 QDQ 量化（encoder 全量化，decoder 排除 SeACo 路径）
    - 精度对比工具链（evaluate_cer.py / compare_accuracy.py / inspect_engine_precision.py）

v2.1 工程性能优化（已完成，见 docs/DEPLOY.md 高并发性能调优）：
    - Uniform Chunking 消除桶分组合批瓶颈
    - 工业标准 dynamic batching（满触发 + 按最早 chunk 超时兜底）
    - VAD Session Pool + OMP=1（消除 libgomp 崩溃）
    - TRT 多 stream 多 execution_context
    - 两种部署模式：单进程 QPS 13.15 / 多进程 QPS 93.92
    - 已 tag v2.1.0

v2.2 API 重构（已完成）：
    - /asr → /chinese_asr；b64 → base64；article_url 透传
    - 响应对齐 istar_asr + asr[]（idx/slid/text/speaker/timestamp/words）

v2.3 字级时间戳（已完成，方案 C）：
    - 独立第 5 段 timestamp engine（upsample CIF head + blstm），对齐 FunASR
      ts_prediction_lfr6_standard（相邻 fire 中点划界 + 超长截断 + 静音扣除）
    - 响应 asr[].words 字级时间戳；跨 chunk 单调钳制消除段边界重叠
    - ENABLE_WORD_TIMESTAMP 开关（默认 false）；实测开销 -32%（BLSTM 硬瓶颈）
    - v2.3.1 修复 TRT 多 context 竞态（queue.Queue 连接池借还）

v2.4 模块开关 + 防误触发（已完成）：
    - 默认词表恒走路径 B（Faiss），防通用识别相似音误纠
    - ENABLE_HOTWORD / ENABLE_FAISS_CORRECTION 开关（默认 true）
    - 短音频 pad 到 2040ms、空音频返回成功 + message="音频内容为空"

v2.5 三后端统一 + 最优默认（已完成，当前）：
    - 三后端（TRT/ORT/PT）字级时间戳/热词/Faiss 功能统一，均参数可选
      * ORT 分段串联（ort_engine.py，启用时间戳时替代整体模型）
      * PT 原生推理（pt_engine.py，GPU 优先/CPU 兜底）
    - timestamp 纳入精度体系（fp16/fp32，int8 回退 fp16）
    - 默认参数固化 A10 最优（WORKERS=11/CPU_POOL=32/VAD_POOL=2/BATCH_TIMEOUT=10）
    - ORT CUDA arena kSameAsRequested 降显存；启动 dump 生效配置
    - 各精度/后端横向对比（trt_fp16>trt_fp32>onnx_fp32>pt>onnx_int8）

v2.5.x 健壮性加固（已完成，三轮终审修复，tag v2.5.1）：
    - 请求级端到端超时：INFER_TIMEOUT（默认 120s，asyncio.wait_for），防调度卡死请求永久挂起
    - 通用异常兜底 handler：非 ASRException（切段/特征/解码）统一转结构化 500
    - OOM fallback 形状修复（动态 target_seq_len + 切片赋值）+ 多标记 OOM 判定
    - Faiss 索引 _FaissState 原子快照切换（消除 reload 与 correct 并发中间态）
    - 关闭时线程池 shutdown；/health 未加载返回 degraded（供探针识别）
    - .env WORKERS 拼写修复（原 WORKS 静默失效）

v2.6 风险点闭环（已完成，M3/I3/I5/R12/R14）：
    - M3 超长音频 chunk 分片限流：per-request 信号量 MAX_INFLIGHT_CHUNKS_PER_REQUEST
      （默认 64，0=不限），约束同时在途 chunk 数，完成一个放行一个（非分批 barrier）。
      超长音频（上千 chunk）不再一次性灌满调度器、饿死其他请求、抬高内存峰值。
    - I3 字级时间戳与 Faiss 纠错一致：Faiss 新增 correct_with_spans 返回替换区间，
      _apply_faiss_to_words 把替换同步映射到 asr[].words（等分替换段时间区间），
      保证 words 拼接 == 纠错后 text。口径不一致（空格/BPE 边界）时保守放弃 words 改写。
    - I5 中英混合字级时间戳口径统一：_decode_char_list 清理 BPE 连接标记 @@ 与
      sentencepiece 前缀 ▁、过滤 <...> 特殊标记，与 tokenizer.decode 显示口径一致
      （英文 subword 各自带时间戳，拼接字面与段 text 一致；中文逐字不变）。
    - R12/R14 运行时健康探针：asr_engine 记录连续失败数/累计成功失败/静默降级原因；
      scheduler 在推理成功/失败/超时打点；/health 纳入运行时判定（连续失败超
      HEALTH_MAX_CONSECUTIVE_FAILURES=20 → degraded），后端静默回退（TRT→ORT 等）
      通过 runtime.degraded_reason 暴露；HEALTH_ACTIVE_PROBE=true 可开主动 dummy 探针。

v2.7 句子级时间戳（已完成）：
    - 标点分句器 src/sentence_segmenter.py：CT-Transformer 标点模型（iic/punc_ct-transformer_
      zh-cn-common-vad_realtime-vocab272727-onnx）纯 onnxruntime 手写推理，懒加载，
      依赖/模型缺失自动禁用降级回段级，不阻塞主流程。
      * 早期用 ngram-punctuator（KenLM），因对长文本/重复口语困惑度阈值失效（整段不断句）
        且慢，已弃用改 CT-Transformer：逐 token 分类无阈值问题，实测 20000+ 字/秒，标点完整。
    - asr[] 粒度变为子句（ENABLE_SENTENCE_TIMESTAMP=true）：逐 token 恢复标点，任何标点
      （，。？、）都切成独立子句，子句字符区间映射全局字级 words → timestamp=[首字start,末字end]，
      asr[] 每项=一子句（text 带标点/timestamp 子句时间/words 该句字级），istar_asr 子句拼接。
    - ★强依赖 ENABLE_WORD_TIMESTAMP=true（子句时间边界靠字级定位）；未开则启动告警 +
      自动降级回段级输出，段级形态完全不变。
    - ONNX 契约：输入 inputs/text_lengths/vad_masks/sub_masks，输出 logits[1,L,6]，
      punc_list=[<unk>,_,，,。,？,、]；CharTokenizer 逐字查 tokens.json（272727）；长文本按
      PUNC_MAX_LEN 滑窗。参数 PUNC_ONNX_NAME（默认 model_quant.onnx）/ PUNC_MAX_LEN（默认 200）。
    - 模型扁平存于 models/punc（model_quant.onnx + tokens.json + config.yaml）；缺失时
      自动调 scripts/download_punc.py（HTTP 直链）下载；ASR PT 权重缺失同样自动调
      scripts/download_asr.py（prepare_model.ensure_pt）。
    - 纯 onnxruntime（项目已有），无新增依赖（弃用 ngram-punctuator/kenlm/modelscope）。

暂不实施（明确搁置）：
    - trt_int8_enc GPU 吞吐/显存实测、热词路径 A（带 hotwords）吞吐实测
    - 真实标注测试集 CER 复核（当前以 fp16 输出为参考基准）

######################### 交付物 ######################################

代码：
    src/                  推理服务
    seaco_paraformer/     模型定义（不依赖 funasr 运行时）
    scripts/              导出/转换/校准/评测脚本
    tests/                单元测试与压测

文档：
    docs/README.md        产品说明 + 环境变量 + 架构概览
    docs/API.md           接口 schema + 错误码
    docs/DEPLOY.md        Docker + K8s + 性能调优（两种部署模式）
    docs/CONTRIBUTING.md  模型更新流程 + 排查经验 + 热词开发细节

开发规范：
    - Python 文件名英文
    - 文档/日志中文
    - 每次改动明确同意后执行
    - 未要求创建新文件时在原文件上修改
    - 涉及中文的文件用代码编辑工具（避免 shell 编码损坏）
