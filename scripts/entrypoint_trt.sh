#!/bin/bash
# TRT 推理镜像启动脚本
# 1. 检测 TRT engine 是否存在
# 2. 不存在则自动构建（首次启动约 5-10 分钟）
# 3. 启动 uvicorn 服务

set -e

ENGINE_DIR="./models/asr/trt"
ONNX_MODEL="./models/asr/fp32/model.onnx"
ONNX_BIAS="./models/asr/fp32/model_eb.onnx"

# 获取 GPU 名称（简化）
GPU_NAME=$(python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0).lower()
    for p in ['nvidia ', 'geforce ', 'rtx ', 'tesla ']:
        name = name.replace(p, '')
    print(name.strip().replace(' ', '_'))
else:
    print('cpu')
" 2>/dev/null || echo "unknown")

echo "=========================================="
echo "SeACo-Paraformer TRT 推理服务"
echo "=========================================="
echo "GPU: ${GPU_NAME}"
echo "MODEL_PRECISION: ${MODEL_PRECISION:-auto}"
echo ""

# 确定需要的精度
PRECISION="fp16"
if [ "${MODEL_PRECISION}" = "trt_int8" ]; then
    PRECISION="int8"
fi

# 检查 engine 是否存在
ENGINE_FILE="${ENGINE_DIR}/${GPU_NAME}_model_${PRECISION}.engine"
BIAS_ENGINE_FILE="${ENGINE_DIR}/${GPU_NAME}_model_eb_${PRECISION}.engine"

mkdir -p "${ENGINE_DIR}"

if [ ! -f "${ENGINE_FILE}" ]; then
    echo "[构建] TRT engine 不存在，开始构建..."
    echo "  目标: ${ENGINE_FILE}"
    echo "  精度: ${PRECISION}"
    echo "  预计耗时: 5-10 分钟"
    echo ""

    if [ -f "${ONNX_MODEL}" ]; then
        python scripts/convert_trt.py \
            --input "${ONNX_MODEL}" \
            --output "${ENGINE_FILE}" \
            --precision "${PRECISION}"
    else
        echo "[警告] ONNX 模型不存在: ${ONNX_MODEL}，跳过 TRT 构建，回退 ORT"
    fi
else
    echo "[缓存] TRT engine 已存在: ${ENGINE_FILE}"
fi

# 构建 bias encoder engine
if [ -f "${ONNX_BIAS}" ] && [ ! -f "${BIAS_ENGINE_FILE}" ]; then
    echo "[构建] Bias encoder TRT engine..."
    python scripts/convert_trt.py \
        --input "${ONNX_BIAS}" \
        --output "${BIAS_ENGINE_FILE}" \
        --precision "${PRECISION}" \
        --profile bias
elif [ -f "${BIAS_ENGINE_FILE}" ]; then
    echo "[缓存] Bias engine 已存在: ${BIAS_ENGINE_FILE}"
fi

echo ""
echo "[启动] uvicorn 服务 (workers=${WORKS:-1})"
exec python -m uvicorn src.main:app --host 0.0.0.0 --port 8080 --workers ${WORKS:-1}
