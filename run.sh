#!/bin/bash
# ============================================================
# SeACo-Paraformer 容器内一键启动脚本（手动测试用）
#
# 适用场景：docker run -it ... 镜像 /bin/bash 进入容器后，
#   手动改下方参数值再运行本脚本一键启动服务，测试各精度。
#
# 用法：
#   bash run.sh        # 用下方参数启动（直接改下面的值即可）
#
# 说明：
#   - 直接修改下方等号右边的值更换配置；每个参数后注释列出可选值
#   - 启动前会调 prepare_model.py 按精度检查/构建产物，缺失则现场转换
#   - 容器内部端口固定 8080，对外靠 docker run -p 宿主端口:8080 映射
# ============================================================

# ─── 模型精度（核心，逐个测试时改这里）───
MODEL_PRECISION=${MODEL_PRECISION:-trt_fp16}       # 可选: auto onnx_fp32 onnx_int8 trt_fp32 trt_fp16 trt_int8 trt_int8_enc

# 单段精度覆盖（可选，优先级高于 MODEL_PRECISION；留空 "" 表示不覆盖）
ENCODER_PRECISION=${ENCODER_PRECISION:-}           # 可选: "" fp32 fp16 int8
CIF_PRECISION=${CIF_PRECISION:-}                   # 可选: "" fp32 fp16 int8
DECODER_PRECISION=${DECODER_PRECISION:-}           # 可选: "" fp32 fp16 int8
BIAS_PRECISION=${BIAS_PRECISION:-}                 # 可选: "" fp32 fp16 int8
TIMESTAMP_PRECISION=${TIMESTAMP_PRECISION:-}       # 可选: "" fp32 fp16（含 BLSTM 不量化，int8 会回退 fp16）

# ─── 服务运行参数 ───
WORKERS=${WORKERS:-1}                         # uvicorn worker 进程数；可选: 1 2 4...（GPU 显存够才调大）
BATCH=${BATCH:-12}                            # 最大 batch；合法值: 1 2 4 8 12
BATCH_TIMEOUT=${BATCH_TIMEOUT:-10}            # batch 等待超时(ms)，工业标准 max_queue_delay；可选: 10 20 30 50（实测 10 吞吐最优）
MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-2000}   # 最大并发请求数
ACQUIRE_TIMEOUT=${ACQUIRE_TIMEOUT:-5}         # 过载拒绝等待超时（秒）；0=不拒绝（无限排队）
MAX_AUDIO_DURATION_MS=${MAX_AUDIO_DURATION_MS:-7200000}   # 音频时长上限（ms）；默认 2 小时；0=不限
LOG_LEVEL=${LOG_LEVEL:-INFO}                  # 可选: DEBUG INFO WARNING ERROR
VERBOSE=${VERBOSE:-0}                         # 可选: 0 1（1=输出各阶段耗时，需配合 LOG_LEVEL=DEBUG）

# ─── CPU 推理线程数（★仅主 ASR 走 CPU 后端 onnx_fp32/onnx_int8 时生效）───
# 应用范围：只作用于 asr_engine.py 的 device=="cpu" 分支（主 ASR ORT 推理）。
#   - TRT 后端（trt_*）：主 ASR 在 GPU 推理，这两个参数无效；
#   - Silero VAD：vad.py 硬编码 intra=inter=1（串行 LSTM，多线程无收益），不读这两个参数。
#   → GPU/TRT 部署下调这两个参数无意义，网格压测勿将其列为变量。
# 高并发（CPU 部署）务必按经验法则设小，避免线程超额订阅（越并发越慢）：
#   ORT_INTRA_OP_THREADS × WORKERS × 预期并发 ≈ 物理核数
#   低延迟单请求: WORKERS=1 + ORT_INTRA_OP_THREADS=全核
#   高并发吞吐:  ORT_INTRA_OP_THREADS = 总核数 / 并发数
ORT_INTRA_OP_THREADS=${ORT_INTRA_OP_THREADS:-0}    # 单 session 算子并行线程数；0=自动取全核（仅 CPU 后端）
ORT_INTER_OP_THREADS=${ORT_INTER_OP_THREADS:-1}    # session 间并行线程数；可选: 1 2（仅 CPU 后端）
CPU_THREAD_POOL_SIZE=${CPU_THREAD_POOL_SIZE:-0}    # CPU 流水线线程池(Stage1 VAD+Stage2 特征提取)；0=自动全核；多 worker 务必设小；★对所有后端生效
VAD_SESSION_POOL_SIZE=${VAD_SESSION_POOL_SIZE:-4}  # VAD ORT session 池大小（round-robin，多请求并行 VAD）
GPU_STREAM_POOL_SIZE=${GPU_STREAM_POOL_SIZE:-4}    # TRT 多 stream 多 context 池（榨干 GPU sm）；作用于 encoder/cif/decoder(+timestamp)；bias_encoder 固定 1（低频调用无需池化）
ENABLE_WORD_TIMESTAMP=${ENABLE_WORD_TIMESTAMP:-false}  # 字级时间戳（asr[].words）；true 启用，吞吐降~30%；可选: true false

# ─── 热词模块开关（按需裁剪推理路径，纯通用识别可全关省开销）───
ENABLE_HOTWORD=${ENABLE_HOTWORD:-true}                 # 路径A SeACo 在线热词（客户端传 hotwords 时）；可选: true false
ENABLE_FAISS_CORRECTION=${ENABLE_FAISS_CORRECTION:-true}  # 路径B 默认词表 Faiss 后处理纠错（客户端不传时）；可选: true false

# ─── OMP / BLAS 线程数（★重要，高并发稳定性和性能双收益）───
# libgomp/MKL/OpenBLAS 默认按 CPU 核数预分配线程池；高并发下多 session/多线程叠加
# 会触发 libgomp thread creation failed 崩溃，且 VAD 是串行 LSTM 无 OMP 并行收益。
# 20 并发压测扫描：OMP=1 (QPS 12.32) > OMP=2 (12.09) > OMP=4 (11.53) > OMP=8 (11.06)。
# 默认 1：稳定性最好、性能最优；如果观测到 CPU 未跑满可尝试调 2（略降）。
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}         # OpenMP 线程数；★强烈建议 1
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}         # MKL 线程数（numpy/torch 走 MKL 时生效）
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}  # OpenBLAS 线程数（numpy 走 OpenBLAS 时生效）

# ─── Bucket / Batch（改动后需重新转 engine）───
BUCKET_SEQ_LENS=${BUCKET_SEQ_LENS:-34,67,134}      # 桶边界 LFR 帧数（2s/4s/8s）
VALID_BATCH_SIZES=${VALID_BATCH_SIZES:-1,2,4,8,12} # 合法 batch size 列表
TRT_OPT_SEQ=${TRT_OPT_SEQ:-67}                # TRT profile opt 主力桶；可选: 34 67 134
TRT_OPT_BATCH=${TRT_OPT_BATCH:-4}             # TRT profile opt batch；可选: 1 2 4 8 12

# ─── 热词参数（改 MAX/OPT 后需重新转 bias/decoder engine）───
MAX_HOTWORD_NUM=${MAX_HOTWORD_NUM:-256}       # 热词硬上限 / 路径切换点（≤走 SeACo，>走 Faiss）
OPT_HOTWORD_NUM=${OPT_HOTWORD_NUM:-64}        # TRT profile opt 热词数
NFILTER=${NFILTER:-50}                        # ASF 过滤注入 decoder 的 top-K
MAX_HOTWORD_LEN=${MAX_HOTWORD_LEN:-16}        # 单热词最大 token 数（英文 BPE 后可达 ~12）

# ─── 词表热更新 ───
DEFAULT_HOTWORD_PATH=${DEFAULT_HOTWORD_PATH:-models/asr/hotwords.txt}
HOTWORD_RELOAD_ENABLED=${HOTWORD_RELOAD_ENABLED:-true}   # 可选: true false
HOTWORD_POLL_INTERVAL=${HOTWORD_POLL_INTERVAL:-5}        # 各 worker 轮询 version 间隔（秒）

# ─── 路径 B：Faiss 大词库纠错（默认词表 >MAX_HOTWORD_NUM 时启用）───
FAISS_WINDOW_SIZES=${FAISS_WINDOW_SIZES:-2,3,4}    # 滑窗大小
FAISS_TOPK=${FAISS_TOPK:-30}                       # 召回数
FAISS_PINYIN_WEIGHT=${FAISS_PINYIN_WEIGHT:-0.75}   # 拼音权重
FAISS_EDIT_WEIGHT=${FAISS_EDIT_WEIGHT:-0.25}       # 编辑距离权重
FAISS_SCORE_THRESHOLD=${FAISS_SCORE_THRESHOLD:-0.85}   # Faiss 检索分门槛
GAP_THRESHOLD=${GAP_THRESHOLD:-0.05}               # top1-top2 区分度门槛
FINAL_SCORE_THRESHOLD=${FINAL_SCORE_THRESHOLD:-0.88}   # 融合分门槛

# ─── 本地 PT 权重目录 + 校准数据 ───
PT_MODEL_DIR=${PT_MODEL_DIR:-models/asr/pt}
CALIB_DATA=${CALIB_DATA:-calib_data/audio_data}

# ─── UTF-8 locale（中文日志/文件安全）───
LC_ALL=${LC_ALL:-C.UTF-8}                     # 可选: C.UTF-8 en_US.UTF-8
LANG=${LANG:-C.UTF-8}

# ============================================================
# 以下为执行逻辑，一般无需修改
# ============================================================
export MODEL_PRECISION ENCODER_PRECISION CIF_PRECISION DECODER_PRECISION BIAS_PRECISION TIMESTAMP_PRECISION
export WORKERS BATCH BATCH_TIMEOUT MAX_CONCURRENT_REQUESTS ACQUIRE_TIMEOUT MAX_AUDIO_DURATION_MS
export LOG_LEVEL VERBOSE
export ORT_INTRA_OP_THREADS ORT_INTER_OP_THREADS CPU_THREAD_POOL_SIZE VAD_SESSION_POOL_SIZE GPU_STREAM_POOL_SIZE
export ENABLE_WORD_TIMESTAMP ENABLE_HOTWORD ENABLE_FAISS_CORRECTION
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export BUCKET_SEQ_LENS VALID_BATCH_SIZES TRT_OPT_SEQ TRT_OPT_BATCH
export MAX_HOTWORD_NUM OPT_HOTWORD_NUM NFILTER MAX_HOTWORD_LEN
export DEFAULT_HOTWORD_PATH HOTWORD_RELOAD_ENABLED HOTWORD_POLL_INTERVAL
export FAISS_WINDOW_SIZES FAISS_TOPK FAISS_PINYIN_WEIGHT FAISS_EDIT_WEIGHT
export FAISS_SCORE_THRESHOLD GAP_THRESHOLD FINAL_SCORE_THRESHOLD
export PT_MODEL_DIR CALIB_DATA LC_ALL LANG

echo ORT_INTRA_OP_THREADS=$ORT_INTRA_OP_THREADS ORT_INTER_OP_THREADS=$ORT_INTER_OP_THREADS CPU_THREAD_POOL_SIZE=$CPU_THREAD_POOL_SIZE VAD_SESSION_POOL_SIZE=$VAD_SESSION_POOL_SIZE GPU_STREAM_POOL_SIZE=$GPU_STREAM_POOL_SIZE

# 内部固定端口（对外映射由 docker run -p 决定）
PORT=8080

set -uo pipefail
ORT_FP32_MODEL="./models/asr/fp32/model.onnx"

echo "=========================================="
echo "SeACo-Paraformer 服务启动"
echo "=========================================="
echo "MODEL_PRECISION : ${MODEL_PRECISION}"
if [ -n "${ENCODER_PRECISION}${CIF_PRECISION}${DECODER_PRECISION}${BIAS_PRECISION}" ]; then
echo "单段覆盖        : enc=${ENCODER_PRECISION:-—} cif=${CIF_PRECISION:-—} dec=${DECODER_PRECISION:-—} bias=${BIAS_PRECISION:-—}"
fi
echo "WORKERS / BATCH : ${WORKERS} / ${BATCH}"
echo "MAX_HOTWORD_NUM : ${MAX_HOTWORD_NUM} (opt=${OPT_HOTWORD_NUM}, nfilter=${NFILTER})"
echo "内部端口        : ${PORT}（对外由 docker run -p 映射）"
echo "=========================================="
echo ""

# ─── 1. 检查 + 按需构建模型产物 ───
echo "[准备] 检查模型产物（缺失则按依赖链转换）..."
if python scripts/prepare_model.py --precision "${MODEL_PRECISION}"; then
    echo "[OK] 模型产物就绪"
else
    echo "[警告] 目标精度产物准备失败"
    if [ -f "${ORT_FP32_MODEL}" ]; then
        echo "       检测到 ORT fp32 模型，回退 onnx_fp32 启动"
        export MODEL_PRECISION="onnx_fp32"
    else
        echo "       尝试现场生成 ORT fp32 兜底..."
        if python scripts/prepare_model.py --precision onnx_fp32; then
            export MODEL_PRECISION="onnx_fp32"
            echo "       已生成 ORT fp32 兜底，回退启动"
        else
            echo "[致命] 兜底模型也无法生成，退出"
            exit 1
        fi
    fi
fi

echo ""
echo "[启动] uvicorn (workers=${WORKERS}, MODEL_PRECISION=${MODEL_PRECISION})"
echo "       健康检查: curl http://localhost:${PORT}/health"
echo ""

exec python -m uvicorn src.main:app --host 0.0.0.0 --port "${PORT}" --workers "${WORKERS}"

:<<!
docker run -it -p 8099:8080 --gpus '"device=0"' seaco-asr_infer:latest /bin/bash
curl http://localhost:8099/health

# 默认 trt_int8_enc 启动
bash run.sh

# 测试不同精度（命令行覆盖优先）
MODEL_PRECISION=trt_fp16  bash run.sh
MODEL_PRECISION=trt_fp32  bash run.sh
MODEL_PRECISION=trt_int8  bash run.sh
MODEL_PRECISION=onnx_fp32 bash run.sh
MODEL_PRECISION=onnx_int8 bash run.sh
MODEL_PRECISION=trt_int8_enc bash run.sh
# 单段精度混搭测试
ENCODER_PRECISION=int8 DECODER_PRECISION=fp16 bash run.sh

# 多 worker / 调 batch
MODEL_PRECISION=trt_fp16 WORKERS=2 BATCH=8 bash run.sh

# CPU int8 高并发（16 核机器扛 8 并发：每推理占 2 核，不抢核）
MODEL_PRECISION=onnx_int8 ORT_INTRA_OP_THREADS=2 WORKERS=1 bash run.sh

# 开详细耗时日志
VERBOSE=1 bash run.sh
!