######################### 任务 ########################################
模型地址：
    ASR:https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
    VAD:https://modelscope.cn/models/pengzhendong/silero-vad
任务：
    1.构建运行环境，分为模型转化环境和模型推理环境。有gpu使用gpu,没有则使用cpu
        1).环境与部署
            Python 版本>=3.12
            GPU 推理时 CUDA/cuDNN 版本. cuda:12.1
            Docker 内部署,Dockerfile 编写、镜像分层策略、多阶段构建分离转换环境和推理环境
            模型文件存放路径：models/{asr,vad}
        2).requirements.txt
            提供 requirements-convert.txt 与 requirements-infer.txt
        3).构建docker-compose.yml 配置相关参数及启动服务。
    2.对模型进行onnx转换
        导出产物（双模型架构）：
            model.onnx — ASR 主模型（encoder + predictor + decoder）
            model_eb.onnx — SeACo bias encoder（热词编码器）
        1).FunASR AutoModel.export() 导出 fp32 ONNX → onnxconverter-common 转 fp16，opset_version=16。
        CIF 向量化导出（消除 Loop 算子，使 fp16 转换可行）：
            问题：FunASR bicif_paraformer 中 CifPredictorV3Export 调用的 cif_export/cif_wo_hidden_export 是带 for 循环的 TorchScript 函数，导出为 ONNX Loop 算子，fp16 转换时 Sequence 类型不兼容。
            方案：在转换容器内修改 FunASR 源码，将 Loop 版本替换为 paraformer 模块中的向量化版本（cumsum 实现）：
                sed 修改 /usr/local/lib/python3.12/dist-packages/funasr/models/bicif_paraformer/cif_predictor.py：
                    1. 顶部添加 import：from funasr.models.paraformer.cif_predictor import cif_v1_export as _cif_v1_export, cif_wo_hidden_v1 as _cif_wo_hidden_v1
                    2. 替换 cif_export → _cif_v1_export
                    3. 替换 cif_wo_hidden_export → _cif_wo_hidden_v1
            效果：导出的 ONNX 无 Loop 算子，fp16 转换成功。
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
                ONNX 推理使用自实现模块（与推理服务一致，不依赖 funasr）
        3).opset_version=16
    3.构建fastapi服务
        1).环境变量：
            变量：
                WORKS：works 是 uvicorn workers 数量
                BATCH:每个 worker 独立 batch 
                PORT、MODEL_DIR、DEVICE、BATCH_TIMEOUT、LOG_LEVEL
                MAX_BATCH_DURATION # 单次 GPU 推理 batch 中，所有 chunk 的总音频时长不得超过 30 秒
                MAX_CONCURRENT_REQUESTS:服务同时允许处理的最大请求数
            默认配置：
                WORKS=1
                BATCH=1
                PORT=30960
                MODEL_DIR=./models
                DEVICE=auto
                BATCH_TIMEOUT=10 # 毫秒
                LOG_LEVEL=INFO
                MAX_BATCH_DURATION=30 # 秒
                MAX_CONCURRENT_REQUESTS=2000
        2).多请求合并推理，具体 batch 组装与调度逻辑见 3.16 GPU Scheduler。
        3).热词注入方式：API 支持传入 hotwords 参数，设定为可选参数。
        4).音频切段的 VAD 策略：静音检测切段，直接用官方 ONNX 文件。VAD 只输出时间戳列表 [(start_ms, end_ms), ...]，不修改音频数据。
        5).对长音频切段处理，min=5s, opt=12s， max=15s；
            Chunk 调整规则
                第一段 <5s：
                拼接到下一段前部
                最后一段 <5s：
                拼接到上一段尾部
            Overlap 策略
                chunk 间增加：
                200ms overlap
                减少边界切词
            Chunk Metadata
                系统内部维护：
                ChunkMeta = {
                    "chunk_id": int,
                    "segment_index": int,

                    "raw_start_ms": int,
                    "raw_end_ms": int,

                    "overlap_left_ms": int,
                    "overlap_right_ms": int
                }
                用于：
                    时间戳恢复
                    overlap 去重
                    字幕定位
                    后处理
            Overlap 去重策略
                由于 overlap 会导致：
                chunk 边界重复识别
                后处理阶段需要进行 overlap merge。
                支持：
                    token 时间戳对齐
                    文本编辑距离
                    chunk 尾首 token 对齐
                避免重复文本输出。
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
            GPU 推理使用 IO Binding 减少数据拷贝
            服务启动时模型预热（dummy inference）
            batch 内按音频段长度分桶（bucket 队列划分为 5s/8s/12s/15s），减少 padding 开销
            VAD 与 ASR 流水线并行处理；vad采用cpu，ASR根据实际选择设备。
        12).VAD 切段处理
                使用默认配置文件加载配置。最小语音段（min_speech_duration=0.5s）、段间合并策略，避免切碎有效语音。
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
                支持通过 SIGHUP 或 HTTP 接口动态调整 BATCH_TIMEOUT、LOG_LEVEL，避免重启服务
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
                根据 chunk 时长：5s,8s,12s,15s,进入不同 bucket。
                保证：
                    同 batch 内长度接近
                减少 padding。
            Dynamic Batch 组装
                GPU Scheduler 会在：BATCH_TIMEOUT时间窗口内持续收集请求。
                达到：batch_size或MAX_BATCH_DURATION即立即触发推理
            统一 GPU 提交:
                所有 GPU inference：
                    统一由 scheduler 提交。
                避免：
                    多线程直接调用 CUDA
                造成：
                    stream 冲突
                    context 切换
                    GPU utilization 降低
            OOM Fallback
                当出现CUDA out of memory, scheduler 自动减小 batch,重新切 batch,CPU fallback避免服务崩溃
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
            特征提取优化（src/feature_extractor.py）：
                分帧：使用 np.lib.stride_tricks.as_strided 零拷贝视图替代 Python for 循环
                LFR 堆叠：向量化高级索引 + reshape 替代双层 for 循环
                Mel 滤波器矩阵：模块级预计算缓存（单例），避免每次调用重新创建
                窗函数：模块级常量 _WINDOW 预计算
                功率谱：real**2 + imag**2 替代 np.abs()**2（避免 sqrt 开销）
                log/max：np.maximum(out=) + np.log(out=) 原地操作，减少内存分配
            请求处理流程优化（src/main.py）：
                特征提取并行化：通过 asyncio.run_in_executor 将 CPU 密集的特征提取放入线程池，不阻塞 FastAPI 事件循环
                多 chunk 特征提取批量执行：一次性提取所有 chunk 特征后统一提交 GPU Scheduler
            GPU 调度优化（src/scheduler.py）：
                每轮调度清空 bucket 所有可组装 batch（while 循环），避免高并发时积压
                BATCH_TIMEOUT 每次循环重新读取 settings，支持热更新即时生效
任务管理：
    任务依赖关系：任务 1 → 任务 2 → 任务 3，顺序执行
版本管理：
    v1（当前版本，已完成）：
        fp16 ONNX 推理服务，CER ≤ 5%（实测 2.23%）
        设备：2080 Ti / A10
    v2（量化优化）：
        目标设备：2080 Ti（INT8）、A10（INT8/INT4）
        任务：
            1).ONNX INT8 量化推理
                使用 onnxruntime 动态量化或静态量化（需校准数据集）
                对比 fp16 基线的 CER 损失
            2).ONNX INT4 量化推理（仅 A10）
                使用 onnxruntime GPTQ/AWQ 风格量化（如支持）
                对比 fp16 基线的 CER 损失
            3).TensorRT INT8 推理
                将 fp32 ONNX 转为 TensorRT engine（INT8 校准）
                对比 fp16 基线的 CER 损失和推理速度
            4).TensorRT INT4 推理（仅 A10）
                TensorRT FP8/INT4 量化（Ampere+ 架构）
                对比 fp16 基线的 CER 损失和推理速度
            5).精度对比报告
                统一测试集，对比各方案：
                    fp32 ONNX（基线 CER=0%）
                    fp16 ONNX（当前 CER=2.23%）
                    INT8 ONNX
                    INT8 TensorRT
                    INT4 ONNX（A10）
                    INT4 TensorRT（A10）
                指标：CER、RTF（Real-Time Factor）、显存占用、吞吐量
        硬件约束：
            2080 Ti（Turing）：支持 fp16、INT8，不支持 INT4
            A10（Ampere）：支持 fp16、INT8、INT4/FP8
文档与交付物:
    README.md：环境准备、启动命令、API 示例（curl/Python）
    API.md：请求/响应 schema、错误码字典、热词格式说明
    DEPLOY.md：Docker 构建、K8s 部署 YAML 示例、扩缩容建议
    CONTRIBUTING.md：模型更新、ONNX 重导出流程（便于后续迭代）

