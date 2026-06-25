#!/bin/bash
# ============================================================
# SeACo-Paraformer 容器内一键启动脚本（手动测试用）
#
# 适用场景：以 docker run -it ... 镜像 /bin/bash 进入容器后，
#   手动运行本脚本一键启动服务，按需改下方环境变量测试各精度。
#
# 用法：
#   bash run.sh                          # 用下方默认参数启动
#   MODEL_PRECISION=trt_fp16 bash run.sh # 命令行临时覆盖某参数
#   MODEL_PRECISION=trt_int8 WORKS=2 bash run.sh
#
# 说明：
#   - 命令行传入的环境变量优先级高于本脚本默认值（${VAR:-默认}）
#   - 启动前会调 prepare_model.py 按精度检查/构建产物，缺失则现场转换
#   - 容器内部端口固定 8080，对外靠 docker run -p 宿主端口:8080 映射
# ============================================================

# ─── 模型精度（核心，逐个测试时改这里）───
#   onnx_fp32 / onnx_int8 / trt_fp32 / trt_fp16 / trt_int8 / trt_int8_enc / auto
export MODEL_PRECISION="${MODEL_PRECISION:-trt_int8_enc}"

# 单段精度覆盖（可选，优先级高于 MODEL_PRECISION；留空则不覆盖）
#   取值 fp32 / fp16 / int8
export ENCODER_PRECISION="${ENCODER_PRECISION:-}"
export CIF_PRECISION="${CIF_PRECISION:-}"
export DECODER_PRECISION="${DECODER_PRECISION:-}"
export BIAS_PRECISION="${BIAS_PRECISION:-}"

# ─── 服务运行参数 ───
export WORKS="${WORKS:-1}"                                  # uvicorn worker 进程数（GPU 显存够才调大）
export BATCH="${BATCH:-12}"                                 # 最大 batch（合法值 1,2,4,8,12）
export BATCH_TIMEOUT="${BATCH_TIMEOUT:-10}"                 # batch 等待超时（毫秒）
export MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-2000}"
export ACQUIRE_TIMEOUT="${ACQUIRE_TIMEOUT:-5}"             # 过载拒绝等待超时（秒），0=不拒绝
export MAX_AUDIO_DURATION_MS="${MAX_AUDIO_DURATION_MS:-7200000}"  # 音频时长上限（ms），0=不限
export LOG_LEVEL="${LOG_LEVEL:-INFO}"                       # DEBUG/INFO/WARNING/ERROR
export VERBOSE="${VERBOSE:-0}"                              # 1=输出各阶段耗时

# ─── Bucket / Batch（改动后需重新转 engine）───
export BUCKET_SEQ_LENS="${BUCKET_SEQ_LENS:-34,67,134}"
export VALID_BATCH_SIZES="${VALID_BATCH_SIZES:-1,2,4,8,12}"
export TRT_OPT_SEQ="${TRT_OPT_SEQ:-67}"
export TRT_OPT_BATCH="${TRT_OPT_BATCH:-4}"

# ─── 热词参数（改 MAX/OPT 后需重新转 bias/decoder engine）───
export MAX_HOTWORD_NUM="${MAX_HOTWORD_NUM:-256}"
export OPT_HOTWORD_NUM="${OPT_HOTWORD_NUM:-64}"
export NFILTER="${NFILTER:-50}"
export MAX_HOTWORD_LEN="${MAX_HOTWORD_LEN:-8}"

# ─── 词表热更新 ───
export DEFAULT_HOTWORD_PATH="${DEFAULT_HOTWORD_PATH:-models/asr/hotwords.txt}"
export HOTWORD_RELOAD_ENABLED="${HOTWORD_RELOAD_ENABLED:-true}"
export HOTWORD_POLL_INTERVAL="${HOTWORD_POLL_INTERVAL:-5}"

# ─── 路径 B：Faiss 大词库纠错（默认词表 >MAX_HOTWORD_NUM 时启用）───
export FAISS_WINDOW_SIZES="${FAISS_WINDOW_SIZES:-2,3,4}"
export FAISS_TOPK="${FAISS_TOPK:-30}"
export FAISS_PINYIN_WEIGHT="${FAISS_PINYIN_WEIGHT:-0.75}"
export FAISS_EDIT_WEIGHT="${FAISS_EDIT_WEIGHT:-0.25}"
export FAISS_SCORE_THRESHOLD="${FAISS_SCORE_THRESHOLD:-0.85}"
export GAP_THRESHOLD="${GAP_THRESHOLD:-0.05}"
export FINAL_SCORE_THRESHOLD="${FINAL_SCORE_THRESHOLD:-0.88}"

# ─── 本地 PT 权重目录 + 校准数据 ───
export PT_MODEL_DIR="${PT_MODEL_DIR:-models/asr/pt}"
export CALIB_DATA="${CALIB_DATA:-calib_data/audio_data}"

# ─── UTF-8 locale（中文日志/文件安全）───
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"

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
echo "WORKS / BATCH   : ${WORKS} / ${BATCH}"
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
echo "[启动] uvicorn (workers=${WORKS}, MODEL_PRECISION=${MODEL_PRECISION})"
echo "       健康检查: curl http://localhost:${PORT}/health"
echo ""

exec python -m uvicorn src.main:app --host 0.0.0.0 --port "${PORT}" --workers "${WORKS}"
