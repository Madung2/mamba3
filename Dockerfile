FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="12.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    libnuma1 \
    ninja-build \
    python3 \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv venv "${VIRTUAL_ENV}"

WORKDIR /workspace/mamba3

COPY pyproject.toml README.md LICENSE setup.py ./
COPY csrc ./csrc
COPY mamba_ssm ./mamba_ssm

RUN uv pip install --upgrade pip setuptools wheel packaging ninja \
    && uv pip install torch torchvision torchaudio --index-url "${PYTORCH_INDEX_URL}" \
    && MAMBA_FORCE_BUILD=TRUE uv pip install -e . --no-build-isolation

CMD ["bash"]
