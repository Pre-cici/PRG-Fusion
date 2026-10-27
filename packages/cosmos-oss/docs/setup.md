# Setup Guide

<!--TOC-->

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Downloading Checkpoints](#downloading-checkpoints)

<!--TOC-->

## System Requirements

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver >=570.124.06 compatible with [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)
* Linux x86-64
* glibc>=2.35 (e.g Ubuntu >=22.04)

## Installation

Install [git lfs](https://git-lfs.com/):

```bash
sudo apt install git-lfs
git lfs install
```

Clone the repository:

```bash
git clone git@github.com:nvidia-cosmos/<repository_name>.git
cd <repository_name>
git lfs pull
```

Install using a virtual environment:

<details id="virtual-environment"><summary><b>Virtual Environment</b></summary>

Install system dependencies:

```shell
sudo apt update && sudo apt -y install curl ffmpeg libx11-dev tree wget
```

* [uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install the package into a new environment:

```shell
uv sync --python 3.10 --extra=cu128
source .venv/bin/activate
```

Or, install the package into the active environment (e.g. conda):

```shell
uv sync --extra=cu128 --active --inexact
```

CUDA Variants:

| CUDA Version | Arguments | Notes |
| --- | --- | --- |
| CUDA 12.8 | `--extra cu128` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |
| CUDA 13.0 | `--extra cu130` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |

For DGX Spark and Jetson AGX, you must use CUDA 13.0.
</details>

## Downloading Checkpoints

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission
2. Install [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `uv tool install -U "huggingface_hub[cli]"`
3. Login: `hf auth login`
4. Accept the [NVIDIA Open Model License Agreement](https://huggingface.co/nvidia/Cosmos-Guardrail1).

Checkpoints are automatically downloaded during inference and post-training. To modify the checkpoint cache location, set the [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) environment variable.
