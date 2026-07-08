#!/bin/bash
# ============================================================
# 网格压测脚本（CPU 侧线程池参数 × 功能模块开关对照系统）
#
# 固定：BATCH / WORKERS / BATCH_TIMEOUT / GPU_STREAM_POOL_SIZE / MODEL_PRECISION
#       ENABLE_WORD_TIMESTAMP / ENABLE_HOTWORD / ENABLE_FAISS_CORRECTION（均可环境变量覆盖）
# 变量：CPU_THREAD_POOL_SIZE / VAD_SESSION_POOL_SIZE（下方 CONFIGS 笛卡尔积）
#
# 【为何只扫 CPU_POOL / VAD_POOL】TRT 后端下：
#   - ORT_INTRA_OP_THREADS / ORT_INTER_OP_THREADS 只作用于主 ASR 的 CPU 后端
#     （onnx_*），TRT/GPU 推理不经过 ORT CPU session，故对本网格无效；
#   - Silero VAD 的 session 线程数在 vad.py 硬编码为 1，也不读上述两个参数。
#   真正影响 TRT 吞吐的 CPU 侧参数只有 CPU_THREAD_POOL_SIZE（Stage1/2 线程池）
#   与 VAD_SESSION_POOL_SIZE（VAD 并行度）。
#
# 【取值范围】参数 per-worker，实际总线程 ≈ WORKERS × 值：
#   - CPU_POOL 扫 1~32（256 核 / WORKERS=10 不超订分界 ≈ 25/worker，32 已到上界）；
#   - VAD_POOL 扫 1~8（OMP=1 下已知 2~4 最优，8 为上界观察收益拐点）。
#   增删实验组只需编辑下方 CONFIGS 列表（每行 "CPU_POOL VAD_POOL"）。
#
# 每组：启动 run.sh → 等 /health 就绪 → 压测 → 解析结果 → 杀 run.sh → 等显存释放
# 全部跑完输出汇总表 + CSV。
#
# ============================================================
# 【测试对照系统】容器内 /app 下执行。各命令对应报告《性能网格测试报告》章节：
#
#   ── A. CPU 线程池网格（时间戳关，WORKERS=10）───────────── 报告第四章
#      bash tests/benchmark_grid.sh
#
#   ── B. CPU 线程池网格（时间戳开，WORKERS=10）───────────── 报告第六章
#      ENABLE_WORD_TIMESTAMP=true bash tests/benchmark_grid.sh
#      # A vs B = 时间戳+Faiss 组合的开销（同 WORKERS，干净对照）
#
#   ── C. 显存换吞吐（时间戳关，WORKERS=11）───────────────── 报告第七章
#      WORKERS=11 bash tests/benchmark_grid.sh
#      # 关时间戳省显存，可多开 1 个 worker
#
#   ── D. 全模块关闭极限吞吐（WORKERS=11）─────────────────── 报告第八章
#      WORKERS=11 bash tests/benchmark_grid.sh
#      # 默认三模块全关，即纯 ASR 峰值基线
#
#   ── E. 纯时间戳单项开销（时间戳开 / Faiss关 / 热词关）──── 待补（报告 6.4 待精确化）
#      WORKERS=11 ENABLE_WORD_TIMESTAMP=true bash tests/benchmark_grid.sh
#      # E vs D = 纯时间戳开销（同 WORKERS，扣除 Faiss 干扰）
#      # 注意：开时间戳显存吃紧，WORKERS=11 若 OOM 则降到 10 重跑
#
#   ── F. 仅默认词表 Faiss 开销 ──────────────────────────── 报告第八章 8.2
#      WORKERS=11 ENABLE_FAISS_CORRECTION=true bash tests/benchmark_grid.sh
#      # F vs D = Faiss 后处理开销（实测约 -11%）
#
#   ── G. 热词路径 A（SeACo 在线）开销 ───────────────────── 待补（报告 9.2 未测）
#      # 本脚本压测不带 hotwords，路径 A 不会触发；需手动带 --hotwords 压测：
#      WORKERS=11 ENABLE_HOTWORD=true bash run.sh   # 另开终端启动服务
#      python tests/test_service.py --hotwords 词1 词2 ... --concurrency 150 --total 3000
#
# 说明：C/D 配置相同（默认全关），差异仅在“是否显式认知为极限基线”，实测数据可复用。
# ============================================================

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# ─── 固定参数（均可通过环境变量覆盖，便于做模块开关对照实验）───
export BATCH=${BATCH:-12}
export WORKERS=${WORKERS:-10}
export BATCH_TIMEOUT=${BATCH_TIMEOUT:-10}
export GPU_STREAM_POOL_SIZE=${GPU_STREAM_POOL_SIZE:-4}
export MODEL_PRECISION=${MODEL_PRECISION:-trt_fp16}
# 功能模块开关（默认全关，测纯 ASR / 单项开销对照时按需覆盖）
export ENABLE_WORD_TIMESTAMP=${ENABLE_WORD_TIMESTAMP:-false}
export ENABLE_HOTWORD=${ENABLE_HOTWORD:-false}
export ENABLE_FAISS_CORRECTION=${ENABLE_FAISS_CORRECTION:-false}
# 对照实验命令见文件头「测试对照系统」；只跑少数线程组则精简下方 CONFIGS。

# ─── 压测参数 ───
CONCURRENCY=120
TOTAL=2500
AUDIO=test_data/audio_16000_30s.wav
HEALTH_URL=http://localhost:8080/health
SERVICE_URL=http://localhost:8080

# ─── 服务就绪等待上限（秒）：首次可能需构建/预热 ───
READY_TIMEOUT=900
# ─── 杀服务后等显存释放（秒）───
COOLDOWN=20

# ─── 实验组合：每行 "CPU_POOL VAD_POOL" ───
# CPU_POOL 取值 {1,2,4,8,16,32}，VAD_POOL 取值 {1,2,4,8}，笛卡尔积全组合（6×4=24 组）。
# 按需增删行即可扩展/精简实验。
CONFIGS=(
  # CPU_POOL=1
  "1 1"
  "1 2"
  "1 4"
  "1 8"
  # CPU_POOL=2
  "2 1"
  "2 2"
  "2 4"
  "2 8"
  # CPU_POOL=4
  "4 1"
  "4 2"
  "4 4"
  "4 8"
  # CPU_POOL=8
  "8 1"
  "8 2"
  "8 4"
  "8 8"
  # CPU_POOL=16
  "16 1"
  "16 2"
  "16 4"
  "16 8"
  # CPU_POOL=32
  "32 1"
  "32 2"
  "32 4"
  "32 8"
)

RESULT_CSV="benchmark_grid_result.csv"
echo "cpu_pool,vad_pool,total,success,fail,elapsed_s,qps,throughput" > "$RESULT_CSV"

# ─── 工具函数 ───
_kill_service() {
    # 杀掉 uvicorn 主进程 + 所有 worker（run.sh 末尾 exec 成 uvicorn）
    pkill -f "uvicorn src.main:app" 2>/dev/null
    # 兜底：杀可能残留的 run.sh
    pkill -f "run.sh" 2>/dev/null
    sleep "$COOLDOWN"
}

_wait_ready() {
    local waited=0
    while [ "$waited" -lt "$READY_TIMEOUT" ]; do
        if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    return 1
}

# 从 test_service.py 保存的 JSON 解析指标（比 grep 中文更稳）
# 用法：_json <json文件> <字段名>；浮点数保留 2 位小数，整数原样输出
_json() {
    python3 -c "import json; v=json.load(open('$1')).get('$2',''); print(round(v,2) if isinstance(v,float) else v)" 2>/dev/null
}

# ─── 主循环 ───
echo "======================================================================"
echo "网格压测开始：固定 BATCH=$BATCH WORKERS=$WORKERS BATCH_TIMEOUT=$BATCH_TIMEOUT"
echo "             GPU_STREAM_POOL_SIZE=$GPU_STREAM_POOL_SIZE PRECISION=$MODEL_PRECISION"
echo "             ENABLE_WORD_TIMESTAMP=$ENABLE_WORD_TIMESTAMP"
echo "             ENABLE_HOTWORD=$ENABLE_HOTWORD ENABLE_FAISS_CORRECTION=$ENABLE_FAISS_CORRECTION"
echo "共 ${#CONFIGS[@]} 组实验"
echo "======================================================================"

idx=0
for cfg in "${CONFIGS[@]}"; do
    idx=$((idx + 1))
    read -r cpu_pool vad_pool <<< "$cfg"
    tag="c${cpu_pool}_v${vad_pool}"
    run_log="/tmp/run_${tag}.log"
    bench_log="/tmp/bench_${tag}.log"
    json_file="/tmp/bench_${tag}.json"

    echo ""
    echo "[$idx/${#CONFIGS[@]}] CPU_POOL=$cpu_pool VAD_POOL=$vad_pool"

    # 先确保无残留服务
    _kill_service

    # 启动服务（变量项通过环境变量注入）
    CPU_THREAD_POOL_SIZE=$cpu_pool \
    VAD_SESSION_POOL_SIZE=$vad_pool \
        bash run.sh > "$run_log" 2>&1 &

    # 等就绪
    if ! _wait_ready; then
        echo "  [失败] 服务在 ${READY_TIMEOUT}s 内未就绪，跳过该组（见 $run_log）"
        echo "$cpu_pool,$vad_pool,NA,NA,NA,NOT_READY,NA,NA" >> "$RESULT_CSV"
        _kill_service
        continue
    fi
    echo "  服务就绪，开始压测..."

    # 压测（结果写入独立 JSON 文件，避免多组互相覆盖）
    python tests/test_service.py \
        --concurrency "$CONCURRENCY" --total "$TOTAL" --audio "$AUDIO" \
        --output "$json_file" \
        > "$bench_log" 2>&1

    # 解析（从 test_service.py 保存的 JSON 读取，字段名见 metrics dict）
    total=$(_json "$json_file" total_requests)
    success=$(_json "$json_file" success)
    fail=$(_json "$json_file" failed)
    elapsed=$(_json "$json_file" wall_time_s)
    qps=$(_json "$json_file" qps)
    thr=$(_json "$json_file" throughput_audio_s_per_s)

    echo "  结果：总耗时=${elapsed}s QPS=${qps} 吞吐=${thr} 成功=${success}/${total} 失败=${fail}"
    echo "$cpu_pool,$vad_pool,${total:-NA},${success:-NA},${fail:-NA},${elapsed:-NA},${qps:-NA},${thr:-NA}" >> "$RESULT_CSV"

    # 杀服务 + 冷却
    _kill_service
done

# ─── 汇总表 ───
echo ""
echo "======================================================================"
echo "汇总结果（固定 BATCH=$BATCH WORKERS=$WORKERS BATCH_TIMEOUT=$BATCH_TIMEOUT GPU_STREAM=$GPU_STREAM_POOL_SIZE）"
echo "======================================================================"
printf "%-9s %-9s | %-6s %-6s %-5s %-10s %-8s %-10s\n" \
    CPU_POOL VAD_POOL 总请求 成功 失败 总耗时s QPS 吞吐量
echo "----------------------------------------------------------------------"
tail -n +2 "$RESULT_CSV" | while IFS=',' read -r cpu vad total success fail elapsed qps thr; do
    printf "%-9s %-9s | %-6s %-6s %-5s %-10s %-8s %-10s\n" \
        "$cpu" "$vad" "$total" "$success" "$fail" "$elapsed" "$qps" "$thr"
done
echo "======================================================================"
echo "CSV 已保存: $RESULT_CSV"
