# ============================================================
# Stage 1: 模型转换阶段（含 PyTorch + ONNX 导出工具）
# ============================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS converter

WORKDIR /app

# 禁止交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装 Python 3.12 + cuDNN 9 + 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    tzdata \
    curl \
    wget \
    lrzsz \
    vim \
    gnupg2 \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libsndfile1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 安装 pip + 配置清华源加速
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

COPY requirements-convert.txt .
RUN python -m pip install --no-cache-dir -r requirements-convert.txt \
    && rm -rf ~/.cache/pip

COPY scripts/ scripts/
COPY models/ models/
COPY tests/ tests/
COPY test_data/ test_data/

# 模型转换通过脚本执行
# RUN python scripts/export_onnx.py

# ============================================================
# Stage 2: 推理服务阶段（轻量化，仅含 ONNX Runtime）
# ============================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS inference

WORKDIR /app

# 禁止交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装 Python 3.12 + cuDNN 9 + 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    gnupg2 \
    tzdata \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libsndfile1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 安装 pip + 配置清华源加速
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

COPY requirements-infer.txt .
RUN python -m pip install --no-cache-dir -r requirements-infer.txt \
    && rm -rf ~/.cache/pip

# 复制服务代码和配置
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/
COPY test_data/ test_data/

# 模型文件打包进镜像
COPY models/ models/

# 环境变量默认值（仅保留运行时可调参数）
ENV WORKS=1
ENV BATCH=12
ENV PORT=8080
ENV BATCH_TIMEOUT=10
ENV LOG_LEVEL=INFO
ENV MAX_CONCURRENT_REQUESTS=2000
ENV MODEL_PRECISION=auto
ENV VERBOSE=0

EXPOSE 8080

# 启动服务
# WORKS=1 时单进程模式（推荐 GPU 服务），WORKS>1 时多 worker（仅 CPU 服务）
CMD python -m uvicorn src.main:app --host 0.0.0.0 --port 8080 --workers ${WORKS:-1}
