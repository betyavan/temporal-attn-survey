FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /root

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

COPY pyproject.toml poetry.lock* /root/
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false
RUN poetry lock
RUN poetry install --no-root --only main

ENV PATH="$HOME/.local/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/root/"
ENV PYTHONPYCACHEPREFIX=/tmp/cpython/