#!/bin/bash
# SeACo-Paraformer 推理镜像启动脚本（转换推理合一）
#
# 启动流程：
#   1. 调 prepare_model.py 按 MODEL_PRECISION 检查产物：
#        - 已存在        → 直接复用
#        - 缺失 + 可构建  → 从本地 PT 权重按依赖链逐级转换
#        - 缺失 + 不可构建 → 返回失败
#   2. 产物就绪 → 启动 uvicorn 服务
#   3. 产物不齐全 → 若有 ORT fp32 兜底则回退，否则报错退出
#
# 精度依赖链与各段精度组合见 scripts/prepare_model.py
# 镜像已内置 nvidia-modelopt（INT8 QDQ）；PT 权重需提前打包到 PT_MODEL_DIR

set -uo pipefail

MODEL_PRECISION="${MODEL_PRECISION:-auto}"
ORT_FP32_MODEL="./models/asr/fp32/model.onnx"

echo "=========================================="
echo "SeACo-Paraformer 推理服务"
echo "=========================================="
echo "MODEL_PRECISION: ${MODEL_PRECISION}"
echo ""

# ─── 1. 检查 + 按需构建模型产物 ───
echo "[准备] 检查模型产物，缺失则按依赖链转换..."
if python scripts/prepare_model.py --precision "${MODEL_PRECISION}"; then
    echo "[OK] 模型产物就绪"
else
    echo "[警告] 目标精度产物准备失败"
    if [ -f "${ORT_FP32_MODEL}" ]; then
        echo "       检测到 ORT fp32 模型，回退 onnx_fp32 启动"
        export MODEL_PRECISION="onnx_fp32"
    else
        echo "[错误] 无法准备目标精度产物，且无 ORT fp32 兜底模型: ${ORT_FP32_MODEL}"
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
echo "[启动] uvicorn 服务 (workers=${WORKS:-1}, MODEL_PRECISION=${MODEL_PRECISION})"
exec python -m uvicorn src.main:app --host 0.0.0.0 --port 8080 --workers "${WORKS:-1}"
