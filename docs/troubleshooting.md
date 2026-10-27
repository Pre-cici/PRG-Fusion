# Troubleshooting

## Issues

Please check [GitHub Issues](https://github.com/Pre-cici/PRG-Fusion/issues) for more info and file a new issue if you can't find a solution. Check the
[Inference Guide](inference.md) for the supported command-line interface and output locations.

### No space left on device

Downloads go to the home directory by default. To use another disk, set the following variables.
```shell
export UV_CACHE_DIR=<new_dir>/.cache
export PIP_CACHE_DIR=<new_dir>/.cache
export HF_HOME=<new_dir>/checkpoints
```

### Missing Python.h

Error message: `fatal error: Python.h: No such file or directory`

This is fixed by [installing uv managed python](https://docs.astral.sh/uv/guides/install-python/#installing-python):

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install --reinstall
```

Re-install the package:

```shell
rm -rf .venv
uv venv --python 3.10
uv sync --extra=cu128
```

### CUDA driver version insufficient

**Fix:** Update NVIDIA drivers to latest version compatible with CUDA [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) or [CUDA 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)

Check driver compatibility:

```shell
nvidia-smi | grep "CUDA Version:"
```

### Out of Memory (OOM) errors

Inference requires ~70 GB of GPU memory on a single GPU (see [Hardware Requirements](inference.md#hardware-requirements)). Check free memory with `nvidia-smi` and make sure no other processes are using the GPU.

## Logs

Logs are saved to `<output_dir>/*.log`.
