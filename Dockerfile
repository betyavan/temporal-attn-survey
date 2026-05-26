FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock* /workspace/
RUN uv sync --frozen --no-install-project

ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/workspace/"
ENV PYTHONPYCACHEPREFIX=/tmp/cpython/
