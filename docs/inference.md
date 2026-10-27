# PRG-Fusion Inference

This guide provides instructions on running inference with PRG-Fusion for novel-trajectory driving video synthesis.

### Pre-requisites

1. Follow the [Setup guide](setup.md) for environment setup.
2. Download the checkpoints listed in [Inference Checkpoints](checkpoints.md).

### Hardware Requirements

| Model | Required GPU VRAM |
|-------|-------------------|
| PRG-Fusion (2B base + two control branches) | ~70 GB |

Use an 80 GB-class GPU such as H100 or A100-80G.

### Input Requirements

The demo inputs ship with the repository in `assets/demo/`:

```text
assets/demo/
├── vis.mp4                  # Dense 3DGS rendering for appearance control
├── lidar.mp4                # Sparse colored LiDAR projection for geometry control
└── prg/
    ├── class_labels/*.png   # Lossless PRG region labels (0=preserve, 1=repair, 2=generate)
    └── labels.json
```

Both videos are **960×640, 93 frames** and aligned frame-for-frame. The PRG region labels feed the dense gate and are loaded from `prg_aux_dir` frame by frame.

For custom runs, prepare two aligned control videos rendered along the **target trajectory** and matching PRG region labels in the same layout. The LiDAR control must keep its native colored projections; converting it to Canny edges is not supported.

## Demo Inference

```bash
uv run --extra cu130 python examples/inference.py --config configs/inference/demo.yaml
```

Output is written to `outputs/debug/<timestamp>_demo/`.

## Configuration

All inference settings are defined in the YAML config. Key sampling parameters:

| Parameter | Value | Description |
| --- | --- | --- |
| `guidance` | 3 | Classifier-free guidance |
| `num_steps` | 12 | Diffusion sampling steps |
| `seed` | 2025 | Generation seed |
| `num_conditional_frames` | 1 | First-frame conditioning |
| `num_video_frames_per_chunk` | 93 | Frames per generation chunk |
| `resolution` | `"720"` | Internal bucket selecting 960×640 canvas |
| `use_prg_dense_gate` | true | Enable the PRG dense gate |
| `prg_aux_dir` | `assets/demo/prg` | Lossless PRG region labels used by the dense gate |
| `vis.control_weight` | 1.0 | Appearance branch weight |
| `lidar.control_weight` | 1.0 | Geometry branch weight |

Model settings are fixed:

- `state_t=24` and `hint_keys=vis_lidar`
- Two control branches (`control_branch_indices: [0, 1]`) overlaid on the PRG-Fusion DCP checkpoint
- PRG dense gate with branch class weights `[[1.0,0.5,0.25],[0.25,1.0,0.0]]` and net-level flags `net.use_prg_dense_gate=true`, `net.use_vis_prg_teacher_query=false`, `net.use_spatial_degradation_query=false`
- LiDAR control masks the top 256 rows; the output canvas is never cropped or masked

### Path Semantics

- Sample paths (`video_path`, `control_path`) are relative to **the YAML file location**
- Checkpoint paths (`checkpoint_path`, `experiment_overrides`) are relative to **the repository root**; run commands from the repository root
