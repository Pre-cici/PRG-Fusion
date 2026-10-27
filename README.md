<div align="center">
  <img src="assets/logo.png" alt="PRG-Fusion Logo" height="120" />
  <h1>PRG-Fusion: Orchestrating Generative Priors with Reconstruction Evidence for Driving View Synthesis</h1>
</div>

<div align="center">
  <p>Sipeng He<sup>1</sup>, Jialei Chen<sup>2</sup>, Zhen Fang<sup>1</sup>, Dongchun Ren<sup>2</sup>, Feng Zhao<sup>1</sup></p>
  <p><sup>1</sup>University of Science and Technology of China &nbsp;&nbsp; <sup>2</sup>Yootta</p>
</div>

<p align="center">
  <a href="https://pre-cici.github.io/PRG-Fusion/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page" /></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2609.06948"><img src="https://img.shields.io/badge/arXiv-2609.06948-red" alt="Paper" /></a>
  &nbsp;
  <a href="https://huggingface.co/Post-pone/PRG-Fusion"><img src="https://img.shields.io/badge/Hugging_Face-Models-yellow" alt="Model Weights" /></a>
</p>

PRG-Fusion synthesizes driving videos along novel trajectories using dense 3D Gaussian Splatting (3DGS) renderings and sparse colored LiDAR projections. Preserve, Repair, and Generate (PRG) labels route reconstruction evidence and generative video priors according to regional reconstruction quality.

<div align="center">
  <br>
  <img width="100%" src="assets/pipeline.png?v=4" alt="PRG-Fusion pipeline" />
  <br>
</div>

## Installation

Requirements: Linux, an NVIDIA Ampere-or-newer GPU, FFmpeg, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Pre-cici/PRG-Fusion.git
cd PRG-Fusion

uv venv --python 3.10
uv sync --extra cu130
uv run --extra cu130 python examples/check_environment.py
```

For CUDA 12.8, replace `cu130` with `cu128`. See the [setup guide](docs/setup.md) for system requirements and installation details.

## Model Preparation

Download the PRG-Fusion checkpoint from [Hugging Face](https://huggingface.co/Post-pone/PRG-Fusion):

```bash
uv tool install -U huggingface_hub
hf download Post-pone/PRG-Fusion \
  --revision 19af7dc45c73a07bcee5f3a4a92bfc276bd99175 \
  --include '.metadata' '*.distcp' \
  --local-dir checkpoints/PRG-Fusion/model
```

Download the required Cosmos tokenizer, text encoder, and initialization weights following the [checkpoint guide](docs/checkpoints.md).

## Inference

The demo inputs, including the PRG region labels used by the dense gate, ship with the repository:

```text
assets/demo/
├── vis.mp4                  # Dense 3DGS renderings for appearance control
├── lidar.mp4                # Sparse colored LiDAR projections for geometric control
└── prg/                     # Lossless PRG region labels (preserve / repair / generate)
```

The demo configuration runs one 93-frame chunk at 960×640. To synthesize other trajectories, prepare an aligned 3DGS rendering, a sparse colored LiDAR projection, and PRG region labels in the same layout.

Run inference from the repository root:

```bash
uv run --extra cu130 python examples/inference.py \
  --config configs/inference/demo.yaml
```

Results are written to `outputs/debug/<timestamp>_demo/` by default. See the [inference guide](docs/inference.md) for path semantics and configuration details.

## Acknowledgements and License

PRG-Fusion is built on [NVIDIA Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5). We thank the NVIDIA Cosmos team for making their work available.
We also thank the [DriveStudio](https://github.com/ziyc/drivestudio) and [FreeVS](https://github.com/esdolo/FreeVS) teams for sharing their work.

The code is released under [Apache-2.0](LICENSE), with third-party notices in [ATTRIBUTIONS.md](ATTRIBUTIONS.md). Model weights are distributed under the [NVIDIA Open Model License](https://huggingface.co/Post-pone/PRG-Fusion/blob/main/LICENSE.pdf). Dataset and media licenses remain with their respective owners.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{he2026prgfusion,
  title         = {{PRG-Fusion}: Orchestrating Generative Priors with Reconstruction Evidence for Driving View Synthesis},
  author        = {Sipeng He and Jialei Chen and Zhen Fang and Dongchun Ren and Feng Zhao},
  year          = {2026},
  eprint        = {2609.06948},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.06948}
}
```
