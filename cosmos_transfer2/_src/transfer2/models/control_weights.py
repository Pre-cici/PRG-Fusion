# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for mapping control masks onto the DiT token grid."""

import torch
from torch import Tensor


def resize_control_weight_maps(
    control_context_scale: Tensor,
    latent_state: Tensor,
    *,
    temporal_compression_factor: int,
    patch_spatial: int,
    normalize: bool,
) -> Tensor:
    _, _, num_frames, height, width = latent_state.shape
    height //= patch_spatial
    width //= patch_spatial
    weight_maps = []
    for weight_map in control_context_scale:
        if weight_map.shape[2:5] != (num_frames, height, width):
            expected_frames = temporal_compression_factor * (num_frames - 1) + 1
            if weight_map.shape[2] != expected_frames:
                raise ValueError(f"Control mask has {weight_map.shape[2]} frames, expected {expected_frames}")
            first = torch.nn.functional.interpolate(
                weight_map[:, :, :1],
                size=(1, height, width),
                mode="trilinear",
                align_corners=False,
            )
            remainder = torch.nn.functional.interpolate(
                weight_map[:, :, 1:],
                size=(num_frames - 1, height, width),
                mode="trilinear",
                align_corners=False,
            )
            weight_map = torch.cat([first, remainder], dim=2)
        weight_maps.append(weight_map.permute(0, 2, 3, 4, 1))

    result = torch.stack(weight_maps)
    if normalize:
        scale_factors = 1.0 / torch.clamp_min(result.sum(dim=0), 1.0)
        result = result * scale_factors[None]
    return result


def resize_video_mask_to_latent(mask: Tensor, latent: Tensor, *, temporal_compression_factor: int) -> Tensor:
    """Resize a frame-space mask to the VAE latent grid without softening its boundary."""
    _, _, num_frames, height, width = latent.shape
    if mask.shape[2:5] == (num_frames, height, width):
        return mask
    expected_frames = temporal_compression_factor * (num_frames - 1) + 1
    if mask.shape[2] != expected_frames:
        raise ValueError(f"Video mask has {mask.shape[2]} frames, expected {expected_frames}")
    first = torch.nn.functional.interpolate(mask[:, :, :1].float(), size=(1, height, width), mode="nearest-exact")
    remainder = torch.nn.functional.interpolate(
        mask[:, :, 1:].float(), size=(num_frames - 1, height, width), mode="nearest-exact"
    )
    return torch.cat([first, remainder], dim=2)


def masked_latent_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Return independently normalized per-instance MSE inside a latent-space mask."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {prediction.shape} != {target.shape}")
    if mask.shape != (prediction.shape[0], 1, *prediction.shape[2:]):
        raise ValueError(f"Mask shape {mask.shape} does not match latent shape {prediction.shape}")
    squared_error = (prediction.float() - target.float()).square()
    mask = mask.to(device=squared_error.device, dtype=squared_error.dtype)
    numerator = (squared_error * mask).sum(dim=(1, 2, 3, 4))
    denominator = (mask.sum(dim=(1, 2, 3, 4)) * prediction.shape[1]).clamp_min(1.0)
    return numerator / denominator
