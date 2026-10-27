# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionalFlowLossOutput:
    """Per-sample and per-class statistics for PRG-balanced flow loss."""

    per_instance_loss: torch.Tensor
    per_class_loss: torch.Tensor
    region_mass: torch.Tensor
    class_active: torch.Tensor


@dataclass(frozen=True)
class HDMapLaneLossOutput:
    """Generic HD-map lane supervision independent of PRG routes."""

    rf: torch.Tensor
    edge: torch.Tensor
    mass: torch.Tensor
    active: torch.Tensor


def build_seg_only_control_routing(num_branches: int, seg_branch_index: int) -> tuple[list[float], tuple[int, ...]]:
    """Select only the SEG residual and detach every other control branch."""
    if num_branches < 2:
        raise ValueError("SEG-only lane routing requires at least two control branches")
    if not 0 <= seg_branch_index < num_branches:
        raise ValueError(f"Invalid SEG branch index {seg_branch_index} for {num_branches} branches")
    branch_weights = [0.0] * num_branches
    branch_weights[seg_branch_index] = 1.0
    detached_indices = tuple(index for index in range(num_branches) if index != seg_branch_index)
    return branch_weights, detached_indices


def should_update_hdmap_lane(iteration: int, interval: int) -> bool:
    """Schedule a lane update on optimizer steps interval, 2*interval, ..."""
    if iteration < 0:
        raise ValueError("Training iteration must be nonnegative")
    if interval < 1:
        raise ValueError("HD-map lane update interval must be positive")
    return (iteration + 1) % interval == 0


def extract_hdmap_lane_mask(
    hdmap: torch.Tensor,
    *,
    color_threshold: float = 60.0,
    dilation: int = 9,
) -> torch.Tensor:
    """Extract white/yellow lane markings from a WorldScenario RGB control."""
    if hdmap.dim() != 5 or hdmap.shape[1] != 3:
        raise ValueError(f"Expected HD-map video [B,3,T,H,W], got {tuple(hdmap.shape)}")
    if color_threshold <= 0:
        raise ValueError("color_threshold must be positive")
    if dilation <= 0 or dilation % 2 == 0:
        raise ValueError("dilation must be a positive odd integer")

    rgb = hdmap.detach().float()
    if hdmap.dtype != torch.uint8:
        rgb = (rgb + 1.0) * 127.5
    rgb = rgb.clamp(0.0, 255.0)
    palette = torch.tensor(
        ((255.0, 255.0, 255.0), (255.0, 255.0, 0.0)),
        device=rgb.device,
        dtype=rgb.dtype,
    )
    distance = (rgb[:, None] - palette[None, :, :, None, None, None]).square().sum(dim=2).sqrt()
    lane = (distance.amin(dim=1) < color_threshold).float()
    batch, frames, height, width = lane.shape
    lane = F.max_pool2d(
        lane.reshape(batch * frames, 1, height, width),
        kernel_size=dilation,
        stride=1,
        padding=dilation // 2,
    )
    return lane.reshape(batch, frames, height, width)


def override_prg_labels_with_hdmap_lane_generate(
    prg_labels: torch.Tensor,
    hdmap: torch.Tensor,
    *,
    color_threshold: float = 60.0,
    dilation: int = 9,
    generate_class_id: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route HD-map lane pixels through the PRG Generate gate at inference."""
    if prg_labels.dim() != 4:
        raise ValueError(f"Expected PRG labels [B,T,H,W], got {tuple(prg_labels.shape)}")
    lane_mask = extract_hdmap_lane_mask(
        hdmap,
        color_threshold=color_threshold,
        dilation=dilation,
    ).bool()
    if lane_mask.shape != prg_labels.shape:
        raise ValueError(
            f"HD-map lane mask {tuple(lane_mask.shape)} does not match PRG labels {tuple(prg_labels.shape)}"
        )
    overridden = prg_labels.clone()
    overridden[lane_mask] = generate_class_id
    return overridden, lane_mask


def downsample_hdmap_lane_mask_to_latent(
    lane_mask: torch.Tensor,
    *,
    latent_shape: tuple[int, int, int],
    temporal_compression_factor: int,
) -> torch.Tensor:
    """Area-average video-frame lane support onto the aligned VAE latent grid."""
    if lane_mask.dim() != 4:
        raise ValueError(f"Expected lane mask [B,T,H,W], got {tuple(lane_mask.shape)}")
    if temporal_compression_factor <= 0:
        raise ValueError("temporal_compression_factor must be positive")
    latent_frames, latent_height, latent_width = latent_shape
    expected_frames = temporal_compression_factor * (latent_frames - 1) + 1
    if lane_mask.shape[1] != expected_frames:
        raise ValueError(
            f"Lane mask has {lane_mask.shape[1]} frames, expected {expected_frames} for latent shape {latent_shape}"
        )

    lane = lane_mask.detach().float().clamp(0.0, 1.0)
    first = lane[:, :1]
    if latent_frames > 1:
        remainder = lane[:, 1:].reshape(
            lane.shape[0],
            latent_frames - 1,
            temporal_compression_factor,
            lane.shape[2],
            lane.shape[3],
        )
        lane = torch.cat((first, remainder.mean(dim=2)), dim=1)
    else:
        lane = first
    return F.interpolate(
        lane[:, None],
        size=(latent_frames, latent_height, latent_width),
        mode="area",
    )[:, 0]


def build_hdmap_lane_valid_mask(
    lane_probs: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None,
    conditional_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Build valid support on a latent-aligned HD-map lane grid."""
    if lane_probs.dim() != 4:
        raise ValueError(f"Expected lane probabilities [B,T,H,W], got {tuple(lane_probs.shape)}")
    batch, frames, height, width = lane_probs.shape
    valid = torch.ones_like(lane_probs, dtype=torch.float32)
    if padding_mask is not None:
        if padding_mask.dim() != 4 or padding_mask.shape[:2] != (batch, 1):
            raise ValueError(f"Expected padding mask [B,1,H,W], got {tuple(padding_mask.shape)}")
        padding = F.interpolate(
            padding_mask.to(device=lane_probs.device, dtype=torch.float32),
            size=(height, width),
            mode="area",
        )[:, 0]
        valid = valid * (1.0 - padding.clamp(0.0, 1.0))[:, None]
    if conditional_frame_mask is not None:
        if conditional_frame_mask.dim() != 5 or conditional_frame_mask.shape[:2] != (batch, 1):
            raise ValueError(f"Expected conditional frame mask [B,1,T,H,W], got {tuple(conditional_frame_mask.shape)}")
        conditional = F.adaptive_avg_pool3d(
            conditional_frame_mask.to(device=lane_probs.device, dtype=torch.float32),
            output_size=(frames, height, width),
        )[:, 0]
        valid = valid * (1.0 - conditional.clamp(0.0, 1.0))
    return valid


def disable_global_prg_queries(
    query_active: torch.Tensor,
    query_counts: tuple[int, int, int, int],
) -> torch.Tensor:
    """Disable only the Global bank while preserving all present typed banks."""
    if query_active.dim() != 2 or query_active.shape[1] != sum(query_counts):
        raise ValueError(f"Expected query activity [B,{sum(query_counts)}], got {tuple(query_active.shape)}")
    result = query_active.clone()
    result[:, sum(query_counts[:3]) :] = False
    return result


def build_prg_valid_token_mask(
    route_probs: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None,
    conditional_frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Build a valid generated-token mask on the PRG route grid.

    ``padding_mask`` follows the repository convention where one denotes
    padding. ``conditional_frame_mask`` similarly uses one for latent frames
    whose velocity is replaced by the exact target.
    """
    if route_probs.dim() != 5 or route_probs.shape[1] != 3:
        raise ValueError(f"Expected route probabilities [B,3,T,H,W], got {tuple(route_probs.shape)}")
    batch_size, _, num_frames, height, width = route_probs.shape
    valid = torch.ones(
        batch_size,
        num_frames,
        height,
        width,
        device=route_probs.device,
        dtype=torch.float32,
    )

    if padding_mask is not None:
        if padding_mask.dim() != 4 or padding_mask.shape[:2] != (batch_size, 1):
            raise ValueError(f"Expected padding mask [B,1,H,W], got {tuple(padding_mask.shape)}")
        padding = F.interpolate(
            padding_mask.to(device=route_probs.device, dtype=torch.float32),
            size=(height, width),
            mode="area",
        )[:, 0]
        valid = valid * (1.0 - padding.clamp(0.0, 1.0))[:, None]

    if conditional_frame_mask is not None:
        if conditional_frame_mask.dim() != 5 or conditional_frame_mask.shape[:2] != (batch_size, 1):
            raise ValueError(f"Expected conditional frame mask [B,1,T,H,W], got {tuple(conditional_frame_mask.shape)}")
        conditional = F.adaptive_avg_pool3d(
            conditional_frame_mask.to(device=route_probs.device, dtype=torch.float32),
            output_size=(num_frames, height, width),
        )[:, 0]
        valid = valid * (1.0 - conditional.clamp(0.0, 1.0))

    return valid


def class_balanced_regional_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    route_probs: torch.Tensor,
    *,
    valid_token_mask: torch.Tensor,
    spatial_importance: torch.Tensor | None = None,
    class_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_region_mass: float = 1.0,
    epsilon: float = 1e-6,
) -> RegionalFlowLossOutput:
    """Reduce velocity MSE equally over present PRG classes.

    Velocity error is first averaged over channels and over the exact latent
    patches represented by each routed DiT token. Each class is normalized by
    its own valid mass, so large generate regions cannot dominate smaller
    preserve or repair regions.
    """
    if prediction.shape != target.shape or prediction.dim() != 5:
        raise ValueError(
            "prediction and target must have the same [B,C,T,H,W] shape, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if route_probs.dim() != 5 or route_probs.shape[:2] != (prediction.shape[0], 3):
        raise ValueError(f"Expected route probabilities [B,3,T,H,W], got {tuple(route_probs.shape)}")
    if valid_token_mask.shape != route_probs[:, 0].shape:
        raise ValueError(
            f"Expected valid token mask {tuple(route_probs[:, 0].shape)}, got {tuple(valid_token_mask.shape)}"
        )
    if min_region_mass < 0:
        raise ValueError("min_region_mass must be nonnegative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    target_shape = route_probs.shape[-3:]
    source_shape = prediction.shape[-3:]
    if any(source % target_size != 0 for source, target_size in zip(source_shape, target_shape, strict=True)):
        raise ValueError(f"Velocity grid {source_shape} must be divisible by route grid {target_shape}")
    patch_size = tuple(source // target_size for source, target_size in zip(source_shape, target_shape, strict=True))

    squared_error = (prediction.float() - target.float()).square().mean(dim=1, keepdim=True)
    token_error = F.avg_pool3d(squared_error, kernel_size=patch_size, stride=patch_size)[:, 0]

    routes = route_probs.detach().float().clamp_min(0.0)
    valid = valid_token_mask.detach().float().clamp(0.0, 1.0)
    weights = routes * valid[:, None]
    if spatial_importance is not None:
        if spatial_importance.shape != valid_token_mask.shape:
            raise ValueError(
                f"Expected spatial importance {tuple(valid_token_mask.shape)}, got {tuple(spatial_importance.shape)}"
            )
        weights = weights * spatial_importance.detach().float().clamp_min(0.0)[:, None]
    region_mass = weights.sum(dim=(2, 3, 4))
    weighted_error = (weights * token_error[:, None]).sum(dim=(2, 3, 4))
    per_class_loss = weighted_error / region_mass.clamp_min(epsilon)
    class_active = region_mass >= min_region_mass

    policy_weights = torch.tensor(class_weights, device=prediction.device, dtype=torch.float32)[None]
    active_weights = policy_weights * class_active
    per_instance_loss = (per_class_loss * active_weights).sum(dim=1) / active_weights.sum(dim=1).clamp_min(epsilon)
    return RegionalFlowLossOutput(
        per_instance_loss=per_instance_loss,
        per_class_loss=per_class_loss,
        region_mass=region_mass,
        class_active=class_active,
    )


def _spatial_laplacian(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(tensor.shape)}")
    batch, channels, frames, height, width = tensor.shape
    flattened = tensor.float().permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    flattened = F.pad(flattened, (1, 1, 1, 1), mode="replicate")
    kernel = tensor.new_tensor(((0.0, -1.0, 0.0), (-1.0, 4.0, -1.0), (0.0, -1.0, 0.0))).float()
    kernel = kernel.reshape(1, 1, 3, 3).expand(channels, 1, 3, 3)
    edges = F.conv2d(flattened, kernel, groups=channels)
    return edges.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def _masked_per_instance(
    error: torch.Tensor,
    support: torch.Tensor,
    *,
    min_region_mass: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if error.shape != support.shape:
        raise ValueError(f"Expected matching error/support shapes, got {tuple(error.shape)} and {tuple(support.shape)}")
    weights = support.detach().float().clamp_min(0.0)
    mass = weights.sum(dim=(1, 2, 3))
    active = mass >= min_region_mass
    values = (weights * error.float()).sum(dim=(1, 2, 3)) / mass.clamp_min(epsilon)
    return values * active, mass, active


def hdmap_lane_fidelity_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    clean_latent: torch.Tensor,
    lane_probs: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    charbonnier_epsilon: float = 1e-3,
    min_region_mass: float = 1.0,
    epsilon: float = 1e-6,
    compute_edge: bool = True,
) -> HDMapLaneLossOutput:
    """Complete-GT RF and endpoint-edge fidelity on HD-map lane support."""
    if not (prediction.shape == target.shape == clean_latent.shape) or prediction.dim() != 5:
        raise ValueError("prediction, target, and clean_latent must share [B,C,T,H,W]")
    if lane_probs.shape != prediction[:, 0].shape:
        raise ValueError(f"Expected lane probabilities {tuple(prediction[:, 0].shape)}, got {tuple(lane_probs.shape)}")
    if valid_mask.shape != lane_probs.shape:
        raise ValueError(f"Expected valid mask {tuple(lane_probs.shape)}, got {tuple(valid_mask.shape)}")
    if charbonnier_epsilon <= 0 or epsilon <= 0:
        raise ValueError("epsilon values must be positive")

    support = lane_probs.detach().float().clamp(0.0, 1.0) * valid_mask.detach().float().clamp(0.0, 1.0)
    velocity_error = (prediction.float() - target.detach().float()).square().mean(dim=1)
    lane_rf, mass, active = _masked_per_instance(
        velocity_error,
        support,
        min_region_mass=min_region_mass,
        epsilon=epsilon,
    )

    if compute_edge:
        sigma = sigmas.reshape(prediction.shape[0], -1).float().mean(dim=1).reshape(-1, 1, 1, 1, 1)
        clean = clean_latent.detach().float()
        predicted_clean = clean + sigma * (target.detach().float() - prediction.float())
        edge_delta = _spatial_laplacian(predicted_clean) - _spatial_laplacian(clean)
        edge_error = (edge_delta.square() + charbonnier_epsilon**2).sqrt() - charbonnier_epsilon
        edge_error = edge_error.mean(dim=1)
        lane_edge, _, _ = _masked_per_instance(
            edge_error,
            support,
            min_region_mass=min_region_mass,
            epsilon=epsilon,
        )
    else:
        lane_edge = torch.zeros_like(lane_rf)
    return HDMapLaneLossOutput(rf=lane_rf, edge=lane_edge, mass=mass, active=active)
