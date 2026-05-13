# ============================================================
# Stage 1: 模型转换阶段（含 PyTorch + ONNX 导出工具）
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS converter

WORKDIR /app

# 禁止交互式提示（时区选择等）
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装 Python 3.12 和系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    tzdata \
    curl \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libsndfile1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 使用 ensurepip 安装 pip（Python 3.12 无 distutils）
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip

COPY requirements-convert.txt .
RUN python -m pip install --no-cache-dir -r requirements-convert.txt

COPY scripts/ scripts/
COPY models/ models/

# 模型转换在任务2中通过脚本执行
# RUN python scripts/export_onnx.py

# ============================================================
# Stage 2: 推理服务阶段（轻量化，仅含 ONNX Runtime）
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS inference

WORKDIR /app

# 禁止交互式提示（时区选择等）
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装 Python 3.12 和系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    tzdata \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libsndfile1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 使用 ensurepip 安装 pip（避免 distutils 缺失问题）
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip

COPY requirements-infer.txt .
RUN python -m pip install --no-cache-dir -r requirements-infer.txt

# 复制服务代码和配置
COPY src/ src/
COPY configs/ configs/

# 将本地模型文件打包进镜像（构建前需先准备好 models/ 目录）
COPY models/ models/

# 环境变量默认值
ENV WORKS=1
ENV BATCH=1
ENV PORT=30960
ENV MODEL_DIR=./models
ENV DEVICE=auto
ENV BATCH_TIMEOUT=10
ENV LOG_LEVEL=INFO
ENV MAX_BATCH_DURATION=30
ENV MAX_CONCURRENT_REQUESTS=2000

EXPOSE ${PORT}

# 启动服务
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKS}"]
