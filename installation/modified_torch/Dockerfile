FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    tzdata \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    ca-certificates \
    libssl-dev \
    libnuma-dev \
    openmpi-bin \
    libopenmpi-dev \
    ninja-build \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m venv /opt/venvs/torch39

ENV PATH="/opt/venvs/torch39/bin:$PATH"

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
    pyyaml \
    typing-extensions \
    ninja \
    "cmake<4"

CMD ["bash", "custom_builder.sh", "torch", "2.4.0"]