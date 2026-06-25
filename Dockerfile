# ============================================================
# SeACo-Paraformer 推理镜像（转换 + 推理合一）
# 基础镜像：TRT 10.6 + CUDA 12.6 + cuDNN 9 + Python 3.10 + PyTorch 2.5
# 支持 MODEL_PRECISION：onnx_fp32 / onnx_int8 / trt_fp32 / trt_fp16 /
#                       trt_int8 / trt_int8_enc
# 启动时 prepare_model.py 按精度从本地 PT 权重逐级转换出所需产物
# ============================================================
FROM nvcr.io/nvidia/tensorrt:24.11-py3 AS inference

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 仅安装缺少的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip 清华源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装业务 + 转换依赖（镜像已内置 TRT/torch/numpy）
COPY requirements-infer.txt .
RUN pip install --no-cache-dir -r requirements-infer.txt \
    && rm -rf ~/.cache/pip

# 复制服务代码
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/
COPY test_data/ test_data/
# 模型产物准备 + 转换脚本（entrypoint 现场按需转换 PT→ONNX→engine）
COPY scripts/ scripts/
# seaco_paraformer 包（分段 ONNX 导出 / QDQ 量化时需要；纯推理不依赖）
COPY seaco_paraformer/ seaco_paraformer/

# 模型文件（PT 权重 + 配置；ONNX/engine 按需现场生成或预打包）
# PT 权重提前打包进 models/asr/pt/（或挂载，环境变量 PT_MODEL_DIR 指定）
# 配置文件：models/asr/{am.mvn,tokens.json}
COPY models/ models/

# 环境变量默认值
ENV WORKS=1
ENV BATCH=12
ENV PORT=8080
ENV BATCH_TIMEOUT=10
ENV LOG_LEVEL=INFO
ENV MAX_CONCURRENT_REQUESTS=2000
ENV MODEL_PRECISION=trt_int8_enc
ENV VERBOSE=0

EXPOSE 8080

# 启动脚本：prepare_model 检查/构建产物 → 启动服务
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
