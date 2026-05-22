######################### 任务 ########################################
模型地址：
    ASR:https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
    VAD:https://modelscope.cn/models/pengzhendong/silero-vad
任务：
    1.构建运行环境，分为模型转化环境和模型推理环境。有gpu使用gpu,没有则使用cpu
        1).环境与部署
            Python 版本>=3.12
            GPU 推理时 CUDA/cuDNN 版本: CUDA 12.1 + cuDNN 9（通过 NVIDIA apt 源安装 libcudnn9-cuda-12）
            onnxruntime-gpu==1.19.2（要求 cuDNN 9）
            Docker 内部署,Dockerfile 编写、镜像分层策略、多阶段构建分离转换环境和推理环境
            pip 源：清华镜像加速（pypi.tuna.tsinghua.edu.cn）
            模型文件存放路径：models/{asr,vad}
        2).requirements.txt
            提供 requirements-convert.txt 与 requirements-infer.txt
        3).构建docker-compose.yml 配置相关参数及启动服务。
    2.对模型进行onnx转换
        导出产物（双模型架构）：
            model.onnx — ASR 主模型（encoder + predictor + decoder）
            model_eb.onnx — SeACo bias encoder（热词编码器）
        1).FunASR AutoModel.export() 导出 fp32 ONNX，opset_version=16。
        CIF 向量化导出（消除 Loop 算子）：
            问题：FunASR bicif_paraformer 中 CifPredictorV3Export 调用的 cif_export/cif_wo_hidden_export 是带 for 循环的 TorchScript 函数，导出为 ONNX Loop 算子。
            方案：在转换容器内修改 FunASR 源码，将 Loop 版本替换为 paraformer 模块中的向量化版本（cumsum 实现）：
                sed 修改 /usr/local/lib/python3.12/dist-packages/funasr/models/bicif_paraformer/cif_predictor.py：
                    1. 顶部添加 import：from funasr.models.paraformer.cif_predictor import cif_v1_export as _cif_v1_export, cif_wo_hidden_v1 as _cif_wo_hidden_v1
                    2. 替换 cif_export → _cif_v1_export
                    3. 替换 cif_wo_hidden_export → _cif_wo_hidden_v1
            效果：导出的 ONNX 无 Loop 算子。
        模型精度方案：
            方案一（GPU 线上）：直接使用 fp32 模型 + CUDAExecutionProvider
                原因：fp16 模型在 GPU 原生 fp16 kernel 下 CIF cumsum 精度崩溃，输出乱码
                fp16 仅在 CPU 推理时可用（ORT 自动 cast 回 fp32 计算）
            方案二（CPU 线上）：fp32 → int8 动态量化（onnxruntime.quantization.quantize_dynamic）
                量化 MatMul/Gemm 权重为 int8，模型缩小约 75%
                无需校准数据集，适用于 CPUExecutionProvider
            转换脚本：
                scripts/convert_int8.py — fp32 → int8 动态量化
                scripts/convert_fp16.py — fp32 → fp16（备用，仅 CPU 可用）
        2).转换后精度验证（scripts/verify_onnx.py）：
            验证对象：
                PT 模型：FunASR AutoModel.generate() 加载原始 PyTorch 模型推理
                ONNX 模型：onnxruntime.InferenceSession + src/feature_extractor.py + src/tokenizer.py 推理
            验证方式：输入音频 → 分别在 PT 和 ONNX 上推理 → 对比识别文本
            对比指标：
                文本输出：字符错误率 CER ≤ 1% 通过，≤ 5% 警告，> 5% 失败
            运行环境：在转换容器内执行（需要 torch + funasr + onnxruntime）
            依赖说明：
                PT 推理依赖 funasr（仅转换环境）
                ONNX 推理使用内联 torchaudio 特征提取 + 内联 tokenizer（不引用 src/，脚本自包含）
        3).opset_version=16
    3.构建fastapi服务
        1).环境变量：
            变量（运行时可调）：
                WORKS：uvicorn workers 数量（GPU 服务必须 1）
                BATCH：最大 batch size（合法值：1,2,4,8,12）
                PORT：容器内部端口
                BATCH_TIMEOUT：batch 等待超时（毫秒）
                LOG_LEVEL：日志级别
                MAX_CONCURRENT_REQUESTS：最大并发请求数
                MODEL_PRECISION：模型精度选择（auto/fp32/int8）
            固定参数（模型已打包进镜像）：
                MODEL_DIR=./models
            MODEL_PRECISION 策略：
                auto：GPU 环境自动选 fp32，CPU 环境优先选 int8（若存在）
                fp32：强制 fp32（GPU/CPU 均可）
                int8：强制 int8 动态量化模型（仅 CPU）
            默认配置：
                WORKS=1
                BATCH=12
                PORT=8080
                BATCH_TIMEOUT=10 # 毫秒
                LOG_LEVEL=INFO
                MAX_CONCURRENT_REQUESTS=2000
                MODEL_PRECISION=auto
        2).多请求合并推理，具体 batch 组装与调度逻辑见 3.16 GPU Scheduler。
        3).热词注入方式：API 支持传入 hotwords 参数，设定为可选参数。
        4).音频切段的 VAD 策略：静音检测切段，直接用官方 ONNX 文件。VAD 只输出时间戳列表 [(start_ms, end_ms), ...]，不修改音频数据。
        5).VAD 后音频段处理（强制归入固定桶边界）：
            桶边界：2s, 4s, 8s（对应 LFR 帧数 34, 67, 134）
            处理步骤：
                Step 1 - 合并相邻短段：
                    遍历 VAD 段，将相邻段合并直到满足最小桶（2s）
                    合并后不超过最大桶（8s）
                    合并范围：第一个段 start_ms 到最后一个段 end_ms（包含中间静音）
                    时间戳保留原始位置
                Step 2 - 超长段切分：
                    合并后仍超过 8s 的段，按 8s 固定切分
                Step 3 - 最后一段处理：
                    切分后最后一段 < 2s 时合并到前一段（合并后 ≤ 8s）
                Step 4 - 就近桶归类 + pad（由 Scheduler 执行）：
                    ≤ 2s → pad 到 34 帧
                    2s < x ≤ 4s → pad 到 67 帧
                    4s < x ≤ 8s → pad 到 134 帧
                    speech_lengths 传桶长度（保证 attention mask 匹配）
                    输出按实际有效帧数截断
            Chunk Metadata：
                ChunkMeta = { chunk_id, segment_index, raw_start_ms, raw_end_ms }
                时间戳对齐：保留原始 VAD 时间戳，响应中直接使用
            切分只操作时间戳区间，最后按最终区间从原始 PCM 数组中 slice 提取音频片段。
        6).输入输出定义：
            输入{"b64":"wav_16k——1channel 的b64"， "hotwords": ["张三", "李四"]};
            输出：{"code": 0, "text":"全文拼接结果","detail":{"0":{"text":"part0_txt","start_ms":0,"end_ms":5200},"1":{"text":"part1_txt","start_ms":5200,"end_ms":12400},...}}
            失败：{"code": 1001, "error": "DECODE_FAILED", "message": "错误描述"}
            备注：成功时不含 error 字段，失败时返回 code/error/message 字段。
        7).错误处理：音频格式不合法、采样率不匹配、空音频等异常返回格式
        8).健康检查接口（/health）
        9).日志规范：
            日志记录字段：请求 ID、耗时、音频时长、识别结果摘要
            双输出策略：
                stdout：JSON 格式输出，供 Docker/ELK 采集
                本地文件：按天轮转，保留 7 天，路径 logs/asr_{date}.log
            轮转实现：Python logging.handlers.TimedRotatingFileHandler
                when='midnight', backupCount=7, encoding='utf-8'
                超过 7 天自动删除最远历史日志文件
        10).HTTP status code 约定：正常 200，参数错误 400，服务内部错误 500
        11).推理优化：
            ONNX Runtime 开启 graph_optimization_level=ORT_ENABLE_ALL
            ONNX Runtime 禁用内存模式（enable_mem_pattern=False, enable_cpu_mem_arena=False）：
                原因：CIF predictor 输出动态 token 数量，ORT 内存缓存会因 shape 变化导致
                      第二次推理时 decoder self_attn Mul 节点广播失败（180 is invalid）
                影响：每次推理重新分配内存，性能略降（约 5-10%），但保证多次推理稳定性
            GPU 推理禁用 IO Binding（动态 shape 下不稳定，回退到普通 session.run）
            batch 内按音频段长度分桶（bucket 队列划分为 2s/4s/8s），减少 padding 开销
            VAD 与 ASR 流水线并行处理；vad采用cpu，ASR根据实际选择设备。
            batch按照，1，2，4，8，12 进行合并。
            服务启动时模型预热（dummy inference），在不同音频段，不同batch下都预热一次。
        12).VAD 切段处理（对齐官方 Silero VAD OnnxWrapper）
                推理参数：
                    window_size=512 samples, context_size=64 samples
                    state shape=(2, batch, 128)
                    输入：拼接 context + chunk → (batch, 576)
                    sr：标量 int64
                后处理参数：
                    threshold=0.5, neg_threshold=0.35
                    min_speech=250ms, min_silence=100ms, speech_pad=30ms
                合并阶段维护 offset mapping（segment_index → raw_start_ms/raw_end_ms），不物理拼接音频数据，只操作时间戳区间列表。
                合并后交给后续切段(5)处理识别。
        13).可观测性与运维
            结构化日志
                采用 JSON 格式输出，包含 request_id, audio_duration_ms, vad_segments, asr_latency_ms, result_length，便于 ELK 采集
            指标监控
                集成 Prometheus：① asr_request_total（带 status 标签）；② asr_inference_duration_seconds（histogram）；③ gpu_memory_usage_bytes
            链路追踪
                可选集成 OpenTelemetry，标记 VAD/ASR/PostProcess 各阶段 span，便于性能瓶颈定位
            配置热更新
                支持通过 SIGHUP 信号动态调整 BATCH_TIMEOUT、LOG_LEVEL，避免重启服务
                实现：asyncio 事件循环注册 SIGHUP handler（单进程模式下有效）
                用法：kill -HUP <pid>（修改环境变量后发送信号）
        14).业务错误码定义（便于客户端精准处理）：
            错误码表：
                SUCCESS=0                 # 成功
                INPUT_PARAM_FAILED=1000   # 输入参数错误（缺少b64字段、格式不合法等）
                DECODE_FAILED=1001        # 音频解码失败（非WAV、损坏文件等）
                VAD_SEGMENT_ERROR=1002    # VAD 模型推理异常
                AUDIO_SEGMENT_ERROR=1003  # 切段合并逻辑异常（如音频过短无法切段）
                ASR_INFER_FAILED=1004     # ASR 模型推理失败
                AUDIO_TOO_LONG=1005       # 音频超出最大时长限制
                MODEL_LOAD_FAILED=1006    # 模型加载失败
                SERVICE_BUSY=1007         # 队列满/超负载拒绝请求
            错误码与 HTTP Status 对应：
                1000/1001/1005 → 400（客户端可修复）
                1002/1003/1004/1006 → 500（服务端内部异常）
                1007 → 503（服务不可用）
            响应格式统一（与 3.6 一致）：
                成功：{"code": 0, "text": "全文拼接结果", "detail": {"0": {"text": "part0_txt", "start_ms": 0, "end_ms": 5200}, ...}}
                失败：{"code": 1001, "error": "DECODE_FAILED", "message": "音频解码失败，请确认为16kHz单声道WAV格式"}
        15).PCM 生命周期
            全流程仅维护一份原始 PCM 数据
        16).GPU Scheduler（GPU 统一调度器）
            Bucket 管理
                根据 chunk 时长归入固定桶：2s/4s/8s（LFR 帧数 34/67/134）
                同桶内 chunk pad 到桶边界，保证 shape 固定
            Dynamic Batch 组装
                GPU Scheduler 在 BATCH_TIMEOUT 窗口内持续收集同桶 chunk
                达到合法 batch size（1,2,4,8,12）立即触发推理
                超时后按实际数量 pad 到最近合法 batch size 推理
            统一 GPU 提交
                所有 GPU inference 统一由 scheduler 在 GPU 专用单线程池中提交
                避免多线程直接调用 CUDA 造成 stream 冲突
            OOM Fallback
                当 GPU 推理出现 CUDA OOM 时：
                    1. 减半 batch size 重试
                    2. 仍失败则逐条推理
                    3. 仍失败则返回 ASR_INFER_FAILED 错误
        17).特征提取（使用 torchaudio.compliance.kaldi.fbank，确保与训练时特征完全一致）
            文件：src/feature_extractor.py
            依赖：torch + torchaudio（仅用于 fbank 计算，不用于模型推理）
            流程：PCM * 32768 → torchaudio.kaldi.fbank(hamming, 80-dim, 25ms/10ms, dither=0) → LFR(左填充3帧, 堆叠7帧跳6帧) → 560维 → CMVN
            参数：
                SAMPLE_RATE=16000
                NUM_MEL_BINS=80, window=hamming, dither=0, snip_edges=True
                LFR_M=7, LFR_N=6（输出 feat_dim=560）
            CMVN：
                加载模型目录下 am.mvn 文件（支持 .json、.npy、Kaldi文本格式）
                FunASR am.mvn 格式：第一行 AddShift，第二行 Rescale
                公式：output = (input + shift) * scale
        18).Tokenizer 解码（自实现，无第三方 ASR 框架依赖，仅依赖 numpy + json）
            文件：src/tokenizer.py
            词表：vocab8404（8404 个 token，含中文字、英文 subword、标点）
            特殊 token：<blank>=0, <sos>=1, <eos>=2（解码时过滤）
            词表文件格式：
                tokens.json: JSON 数组 ["<blank>", "<sos>", ...]
                tokens.txt: 每行 "token id" 或仅 "token"（行号为 ID）
            decode：token_ids → 过滤特殊 token → 拼接（▁ 替换为空格）
            encode：文本 → 最长匹配 → token_id 列表（用于 hotwords 编码）
        19).推理服务依赖
            推理环境 requirements-infer.txt 包含：
                onnxruntime-gpu — ONNX 模型推理
                torch + torchaudio — 特征提取（kaldi fbank）
                numpy, soundfile — 音频处理
                fastapi, uvicorn, pydantic — HTTP 服务
                prometheus-client, opentelemetry — 可观测性
            不包含 funasr/modelscope（模型转换环境专用）
        20).性能优化实现方案
            特征提取（src/feature_extractor.py）：
                使用 torchaudio.compliance.kaldi.fbank 确保与训练时特征完全一致
                LFR 堆叠：torch as_strided 向量化
            请求处理流程优化（src/main.py）：
                三级流水线：Stage1(VAD) + Stage2(特征提取) 在独立 CPU 线程池，Stage3(GPU) 在 GPU 专用线程池
                多请求间各 Stage 独立并行，CPU/GPU 同时满载
            GPU 调度优化（src/scheduler.py）：
                固定 shape bucket + 合法 batch size 推理
                达到合法 batch 立即触发，超时按实际数量触发
                GPU 专用单线程池，避免 stream 冲突
                OOM Fallback：减半 batch 重试 → 逐条推理 → 返回错误
任务管理：
    任务依赖关系：任务 1 → 任务 2 → 任务 3，顺序执行
版本管理：
    v1（当前版本）：
        目标：
            1. 生成基础推理模型（ONNX fp32 + int8 动态量化）
            2. 完成工业级推理应用工程方案，提高推理吞吐率，CPU/GPU 使用率
            3. 构建可用镜像（转换镜像 + 推理镜像）
        模型精度方案：
            GPU 线上：fp32 模型 + CUDAExecutionProvider（精度稳定）
            CPU 线上：int8 动态量化模型 + CPUExecutionProvider（模型缩小 75%）
            fp16 已验证不可用：GPU 原生 fp16 kernel 下 CIF cumsum 精度崩溃
        架构：三级流水线并行
            Stage 1 - VAD（CPU 线程池）：多请求 VAD 并行检测，完成后立即将 segments 送入下一级
            Stage 2 - 特征提取（CPU 线程池）：VAD 完成后立即开始，按 chunk 粒度提交，不等整个请求完成
            Stage 3 - ASR 推理（GPU Scheduler）：持续收集 chunk，按 bucket 分桶组 batch，统一 GPU 推理
            结果路由：按 request_id + chunk_id 合并，所有 chunk 完成后返回响应
        设计原则：
            - CPU 和 GPU 同时满载（VAD/特征提取占 CPU，ASR 占 GPU）
            - 请求间不互相阻塞（流水线各级独立）
            - 单请求延迟 = max(VAD, 特征提取, ASR)，而非三者之和
    v2（TensorRT 量化优化）：
        环境：
            TensorRT 10.6 + CUDA 12.6（镜像 nvcr.io/nvidia/tensorrt:24.11-py3）
            目标 GPU：A10、2080 Ti
            校准数据：现有测试音频集
            TRT 10.x 支持 NonZero 算子，可直接转换完整模型（无需拆分 encoder/decoder）
        目标：
            1. TensorRT fp16 替代 ORT fp32，提升推理速度 2-3x，显存减半
            2. TensorRT INT8 量化，进一步压缩模型，提升吞吐
            3. 多 GPU 适配，按目标硬件生成专用 engine
            4. 完善精度验证和性能对比工具链
        技术要点：
            TRT engine 硬件绑定：不同 GPU 需分别构建 engine（2080 Ti 的 engine 不能在 A10 上跑）
            Dynamic shape profile：min=(1,34,560) opt=(4,67,560) max=(12,134,560)，与 bucket 策略对齐
            TRT fp16 vs ONNX fp16：TRT 自动逐层决定 fp16/fp32，比 ONNX 全局 fp16 精度更好
            INT8 校准：需要代表性音频数据（200-500 条，覆盖 2s/4s/8s 各桶、不同说话人/噪声）
            engine 缓存：首次构建耗时 5-10min，序列化到文件后续直接加载
            回退机制：TRT engine 加载失败 → 自动回退 ORT fp32
        精度验证标准：
            TRT fp16：CER ≤ 1%（相对 ORT fp32 基线）
            TRT INT8：CER ≤ 3%（相对 ORT fp32 基线）
            超过阈值则调整量化策略（逐层 fallback fp16/fp32）
        执行阶段：
            阶段 1 — TRT fp16 基线（最高优先级）：
                交付物：
                    scripts/convert_trt.py — ONNX → TRT engine 转换（指定 GPU、precision、shape profile）
                    src/trt_engine.py — TensorRT 推理引擎（替代 ORT session，支持 dynamic batch）
                    src/config.py 新增 MODEL_PRECISION=trt_fp16 / trt_int8
                    engine 缓存机制：models/asr/trt/{gpu}_{precision}.engine
                    精度对比：ORT fp32 vs TRT fp16 CER 报告
                技术实现：
                    trtexec 或 Python TRT API 构建 engine
                    dynamic shape：3 个 optimization profile 对应 3 个 bucket
                    推理接口对齐 asr_engine.py（infer_batch_raw 兼容）
                    scheduler.py 无需修改（只替换底层推理引擎）
            阶段 2 — INT8 量化：
                交付物：
                    data/calibration/ — 校准数据集（200-500 条音频）
                    scripts/calibrate_int8.py — INT8 校准脚本（生成 calibration cache）
                    scripts/compare_accuracy.py — 批量精度对比工具
                    精度报告：ORT fp32 vs TRT fp16 vs TRT INT8
                技术实现：
                    IInt8EntropyCalibrator2 校准器
                    校准数据覆盖：短音频(2s) + 中音频(4s) + 长音频(8s)
                    逐层精度分析：标记精度敏感层 fallback fp16
            阶段 3 — 多 GPU 适配 + 工程化：
                交付物：
                    scripts/build_engine.py — 多 GPU 构建脚本（自动检测当前 GPU 构建）
                    Dockerfile 更新：构建阶段生成 TRT engine 或首次启动时构建 + 缓存
                    MODEL_PRECISION=auto 策略更新：TRT engine 存在 → 用 TRT，否则回退 ORT
                    部署文档更新
                engine 存放结构：
                    models/asr/trt/
                    ├── a10_fp16.engine
                    ├── a10_int8.engine
                    ├── 2080ti_fp16.engine
                    └── 2080ti_int8.engine
            阶段 4 — 性能调优 + 生产验证：
                交付物：
                    性能基线报告：各方案 RTX/QPS/显存/延迟对比
                    长时间稳定性测试（连续 24h，监控内存泄漏/精度漂移）
                    最终推荐配置文档
                    scripts/benchmark_trt.py — TRT 专项性能测试
        工程架构（双镜像方案）：
            设计原则：
                - 代码统一：同一份 src/，通过 MODEL_PRECISION 环境变量决定走 ORT 还是 TRT
                - 依赖隔离：trt_engine.py 顶部 try import，v1 镜像没装 TRT 不报错
                - 镜像独立：不同 Dockerfile 安装不同依赖
                - 部署独立：不同 docker-compose 文件启动不同镜像
            文件结构：
                Dockerfile              — v1 推理镜像（ORT + CUDA 12.1）
                Dockerfile.trt          — v2 推理镜像（TRT 8.6.1 + CUDA 12.1）
                docker-compose.yml      — v1 部署
                docker-compose.trt.yml  — v2 部署（含 engine 缓存 volume）
                requirements-infer.txt      — v1 依赖
                requirements-infer-trt.txt  — v2 依赖（含 tensorrt + cuda-python）
            TRT engine 缓存策略：
                - 镜像内只打包 ONNX fp32 模型
                - 首次启动时 entrypoint_trt.sh 自动检测并构建 engine
                - engine 缓存到 Docker volume（trt_engine_cache）
                - 重启不重新构建，volume 持久化
                - 不同 GPU 自动生成不同文件名（{gpu}_{model}_{precision}.engine）
            回退机制：
                - TRT engine 构建失败 → 服务仍可启动（回退 ORT fp32）
                - TRT 推理异常 → 日志告警，返回错误码
文档与交付物:
    README.md：环境准备、启动命令、API 示例（curl/Python）
    API.md：请求/响应 schema、错误码字典、热词格式说明
    DEPLOY.md：Docker 构建、K8s 部署 YAML 示例、扩缩容建议
    CONTRIBUTING.md：模型更新、ONNX 重导出流程（便于后续迭代）

######################### 进展记录 ########################################

v1 — 已完成 ✓
    - ONNX fp32 + int8 模型导出
    - FastAPI 三级流水线推理服务
    - GPU Scheduler（bucket 分桶 + dynamic batch）
    - Docker 双镜像方案（转换 + 推理）
    - 可观测性（Prometheus + 结构化日志）
    - 文档完善（README/API/DEPLOY/CONTRIBUTING）

v2 阶段 1 — 进行中
    已完成：
        ✓ 分段 ONNX 导出（scripts/export_onnx_split.py）
            - encoder.onnx（~604MB）
            - cif.onnx（~23MB）
            - decoder.onnx（~254MB）
        ✓ TRT engine 转换脚本（scripts/convert_trt.py）
            - 支持 encoder/cif/decoder/bias 各自的 dynamic shape profile
            - 支持 fp32/fp16/int8 精度选择
        ✓ TRT 推理引擎（src/trt_engine.py）
        ✓ TRT 部署方案
            - Dockerfile.trt（基于 nvcr.io/nvidia/tensorrt:24.11-py3）
            - docker-compose.trt.yml（含 engine 缓存 volume）
            - scripts/entrypoint_trt.sh（首次构建 + 缓存）
            - requirements-infer-trt.txt
        ✓ 测试验证
            - tests/test_split_onnx_pipeline.py（ORT 分段串联验证）
            - tests/test_trt_pipeline.py（TRT 分段串联验证）
        ✓ 推理成功验证（2080 Ti）：
            - encoder_fp32 + cif_fp16 + decoder_fp16 → 识别正确
            - encoder_fp16 + cif_fp16 + decoder_fp16 → 识别失败（encoder fp16 精度崩溃）
        ✓ 精度分析工具（scripts/analyze_encoder_precision.py）
            - 基于 Polygraphy 逐层对比 ONNX fp32 vs TRT fp16
            - 支持选择性标记关键层输出（避免 mark all 导致 TRT 构建失败）
            - 支持迭代 fallback：指定问题层 fp32 后继续分析后续层
            - 分析结果：Add（残差）层出现 inf 溢出，MatMul 最大误差 69.7

    当前精度方案（临时版本）：
        encoder: fp32（TRT）
        cif:     fp16（TRT）
        decoder: fp16（TRT）
        性能：RTF=0.0297, RTX=33.7x（10s 音频，2080 Ti）

    下一阶段任务 — Encoder 混合精度优化：
        目标：将 encoder 也降到 fp16 可用（混合精度：大部分 fp16 + 敏感层 fp32）
        方案：
            1. 用 analyze_encoder_precision.py 定位精度崩溃的源头层
               - 已知：Add（残差）层 inf 溢出是根因
               - 需要确定：是哪些 encoder block 的哪些子层最先溢出
            2. 逐步 fallback：
               - 先按类别 fallback（如所有 LayerNorm/ReduceMean）
               - 再精确到具体 block（如 encoders.0-3 的残差 Add）
            3. 用 TRT Python API 构建混合精度 engine
               - BuilderFlag.OBEY_PRECISION_CONSTRAINTS
               - 逐层设置 layer.precision = trt.float32
            4. 验证混合精度 engine 的最终输出精度（CER ≤ 1%）
            5. 目标：fp32 层数最少化，最大化 fp16 加速收益


