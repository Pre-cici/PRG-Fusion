# Setup Guide

<!--TOC-->

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Downloading Checkpoints](#downloading-checkpoints)

<!--TOC-->

## System Requirements

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver compatible with CUDA 12.8 or CUDA 13.0
* Linux x86-64
* glibc>=2.35 (e.g Ubuntu >=22.04)
* FFmpeg for video decoding

## Installation

Install system dependencies:

```shell
sudo apt update && sudo apt -y install curl ffmpeg libx11-dev tree wget
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Clone the repository and install:

```shell
git clone https://github.com/Pre-cici/PRG-Fusion.git
cd PRG-Fusion
uv venv --python 3.10
uv sync --extra cu130
```

CUDA Variants:

| CUDA Version | Arguments | Notes |
| --- | --- | --- |
| CUDA 12.8 | `--extra cu128` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |
| CUDA 13.0 | `--extra cu130` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |

Use one CUDA stack consistently and do not install both.

Verify the installation:

```shell
uv run --extra cu130 python examples/check_environment.py
```

## Downloading Checkpoints

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission
2. Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `uv tool install -U huggingface_hub`
3. Login: `hf auth login`
4. Download the model and component weights listed in [Inference Checkpoints](checkpoints.md).

`HF_HOME` controls the Hugging Face download cache location.
