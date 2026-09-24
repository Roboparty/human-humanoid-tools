ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3.10 -m pip install --no-cache-dir \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128

COPY docker/gvhmr/requirements-runtime.txt /opt/hhtools-gvhmr/requirements-runtime.txt
RUN python3.10 -m pip install --no-cache-dir \
        -r /opt/hhtools-gvhmr/requirements-runtime.txt \
    && PYTORCH3D_NO_EXTENSION=1 python3.10 -m pip install --no-cache-dir \
        --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable"

COPY hhtools/integrations/gvhmr_worker.py /opt/hhtools-gvhmr/run_video.py

WORKDIR /workspace/gvhmr
ENTRYPOINT ["python3.10", "/opt/hhtools-gvhmr/run_video.py"]
