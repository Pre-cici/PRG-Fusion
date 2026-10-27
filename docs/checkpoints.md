# Inference Checkpoints

Weights are not bundled with this repository. Download them from Hugging Face before running inference.

Run all commands from the repository root.

```bash
# Hugging Face access
uv tool install -U huggingface_hub
hf auth login
```

## Component Weights

The inference configs use the following pinned component weights:

```bash
# Cosmos tokenizer / VAE
hf download nvidia/Cosmos-Predict2.5-2B tokenizer.pth \
  --revision f176dc95b4a70f53ce01c4b302851595e7322b00 \
  --local-dir checkpoints/nvidia/Cosmos-Predict2.5-2B/f176dc95b4a70f53ce01c4b302851595e7322b00

# Cosmos-Reason1 (text encoder)
hf download nvidia/Cosmos-Reason1-7B \
  --revision 3210bec0495fdc7a8d3dbb8d58da5711eab4b423 \
  --local-dir checkpoints/nvidia/Cosmos-Reason1-7B/3210bec0495fdc7a8d3dbb8d58da5711eab4b423

# Qwen2.5-VL processor/tokenizer
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --local-dir checkpoints/Qwen/Qwen2.5-VL-7B-Instruct/cc594898137f460bfe9f0759e9844b3ce807cfb5
```

## PRG-Fusion Checkpoint

Download the PRG-Fusion checkpoint from the public Hugging Face repository:

```bash
hf download Post-pone/PRG-Fusion \
  --revision 19af7dc45c73a07bcee5f3a4a92bfc276bd99175 \
  --include '.metadata' '*.distcp' \
  --local-dir checkpoints/PRG-Fusion/model
```

Total download size is ~16 GB. The resulting layout:

```text
checkpoints/PRG-Fusion/
  model/
    .metadata
    __0_0.distcp
    __1_0.distcp
```

The checkpoint must stay in PyTorch Distributed Checkpoint (DCP) format (`.metadata` + shards); exported `.pt` files are not supported.

SHA-256 values reported by Hugging Face LFS:

| File | SHA-256 |
| --- | --- |
| `.metadata` | `93f8707b53b7f16bb84ddb6a9c7a231feb8f5c4892261ebdc2f4eab9895a13bd` |
| `__0_0.distcp` | `efbee0f62f9a0ad02efc2859483972d33161a018e4f37549ab664aa3a6db15c7` |
| `__1_0.distcp` | `d3846b51b9c4691083fa90d6cf3106add1aa6227584a9896e22e7873dcf1c4f6` |

## Upstream Initialization

The configs initialize the VIS branch from the upstream Transfer2.5 blur checkpoint and the LiDAR branch from the upstream edge checkpoint. These files are downloaded automatically by the checkpoint resolver:

| Role | URI |
| --- | --- |
| Base initialization and VIS branch 0 | `hf://nvidia/Cosmos-Transfer2.5-2B/general/blur/ba2f44f2-c726-4fe7-949f-597069d9b91c_ema_bf16.pt` |
| LiDAR branch 1 initialization from Edge | `hf://nvidia/Cosmos-Transfer2.5-2B/general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt` |

## Path Semantics

- Component and PRG-Fusion checkpoints are stored under **repository-root** paths (`checkpoints/...`), matching the paths in `configs/inference/`
- Media paths in the YAML are relative to **the YAML file location**

See the [Inference Guide](inference.md) for input requirements and commands.
