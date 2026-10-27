# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionQueryOutput:
    """Typed degradation queries and their spatial routing metadata."""

    queries: torch.Tensor
    query_class: torch.Tensor
    query_active: torch.Tensor
    class_presence: torch.Tensor
    region_probs_by_level: tuple[torch.Tensor, ...]


def apply_prg_query_intervention(
    queries: torch.Tensor,
    query_counts: Sequence[int],
    *,
    intervention: str = "none",
) -> torch.Tensor:
    """Apply an inference-only intervention to typed PRG query banks."""
    if intervention == "none":
        return queries
    if intervention != "swap_repair_generate":
        raise ValueError(f"Unsupported PRG query intervention: {intervention}")
    if len(query_counts) != 4 or sum(query_counts) != queries.shape[1]:
        raise ValueError("query_counts must describe the Preserve/Repair/Generate/Global query slots")
    preserve_count, repair_count, generate_count, _ = query_counts
    if repair_count != generate_count:
        raise ValueError("Repair/Generate query swap requires equal query-bank sizes")

    repair_start = preserve_count
    generate_start = repair_start + repair_count
    output = queries.clone()
    output[:, repair_start:generate_start] = queries[:, generate_start : generate_start + generate_count]
    output[:, generate_start : generate_start + generate_count] = queries[:, repair_start:generate_start]
    return output


def disable_prg_query_classes(
    query_active: torch.Tensor,
    query_class: torch.Tensor,
    disabled_classes: Sequence[str],
) -> torch.Tensor:
    """Disable selected typed query slots while preserving all other slots."""
    if query_active.dim() != 2 or query_class.dim() != 1:
        raise ValueError("Expected query_active [B,N] and query_class [N]")
    if query_active.shape[1] != query_class.shape[0]:
        raise ValueError("query_active and query_class must describe the same number of queries")
    class_ids = {"preserve": 0, "repair": 1, "generate": 2}
    unknown = set(disabled_classes).difference(class_ids)
    if unknown:
        raise ValueError(f"Unknown PRG query classes: {sorted(unknown)}")

    output = query_active.clone()
    for class_name in disabled_classes:
        output[:, query_class == class_ids[class_name]] = False
    return output


def build_degradation_route_probs(
    prg_labels: torch.Tensor,
    *,
    latent_shape: tuple[int, int, int, int, int],
    temporal_compression_factor: int,
    patch_temporal: int,
    patch_spatial: int,
    dtype: torch.dtype,
    device: torch.device,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map video-frame PRG labels to a normalized DiT token-grid route map."""
    if prg_labels.dim() != 4:
        raise ValueError(f"Expected PRG labels shape [B,T,H,W], got {tuple(prg_labels.shape)}")
    batch_size, num_frames, height, width = prg_labels.shape
    if batch_size != latent_shape[0]:
        raise ValueError("PRG labels and latent state must have the same batch size")

    target_t = latent_shape[2] // patch_temporal
    target_h = latent_shape[3] // patch_spatial
    target_w = latent_shape[4] // patch_spatial
    labels = prg_labels.to(device=device).long()
    flattened_labels = labels.reshape(batch_size * num_frames, 1, height, width)
    class_maps = []
    for class_index in range(3):
        class_map = (flattened_labels == class_index).to(dtype=dtype)
        class_map = F.interpolate(class_map, size=(target_h, target_w), mode="area")
        class_maps.append(class_map.reshape(batch_size, num_frames, target_h, target_w))
    route_probs = torch.stack(class_maps, dim=1)

    expected_num_frames = temporal_compression_factor * (target_t - 1) + 1
    if num_frames == expected_num_frames and target_t > 1:
        first_frame = route_probs[:, :, :1]
        remaining = route_probs[:, :, 1:].reshape(
            batch_size,
            3,
            target_t - 1,
            temporal_compression_factor,
            target_h,
            target_w,
        )
        route_probs = torch.cat([first_frame, remaining.mean(dim=3)], dim=2)
    elif num_frames != target_t:
        route_probs = F.interpolate(
            route_probs,
            size=(target_t, target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
    if valid_mask is not None:
        if valid_mask.dim() == 4:
            if valid_mask.shape[:2] != (batch_size, 1):
                raise ValueError(f"Expected valid mask [B,1,H,W], got {tuple(valid_mask.shape)}")
            valid = F.interpolate(
                valid_mask.to(device=device, dtype=dtype),
                size=(target_h, target_w),
                mode="area",
            )[:, :, None].expand(-1, -1, target_t, -1, -1)
        elif valid_mask.dim() == 5:
            if valid_mask.shape[:3] != (batch_size, 1, num_frames):
                raise ValueError(f"Expected valid mask [B,1,T,H,W], got {tuple(valid_mask.shape)}")
            valid = F.interpolate(
                valid_mask.to(device=device, dtype=dtype),
                size=(target_t, target_h, target_w),
                mode="trilinear",
                align_corners=False,
            )
        else:
            raise ValueError(f"Expected valid mask [B,1,H,W] or [B,1,T,H,W], got {tuple(valid_mask.shape)}")
        route_probs = route_probs * valid.clamp(0.0, 1.0)
    normalizer = route_probs.sum(dim=1, keepdim=True)
    return torch.where(normalizer > 0, route_probs / normalizer.clamp_min(1e-12), torch.zeros_like(route_probs))


class RegionAwareQueryExtractor(nn.Module):
    """Pool typed queries only from their corresponding PRG regions.

    Features use channel-first video layout ``[B,C,T,H,W]``. Region
    probabilities use ``[B,3,T,H,W]`` with class order preserve, repair, and
    generate. The same interface supports oracle one-hot teacher masks and
    predicted soft student masks.
    """

    NUM_REGION_CLASSES = 3

    def __init__(
        self,
        feature_dim: int,
        query_dim: int,
        crossattn_dim: int,
        num_preserve_queries: int = 4,
        num_repair_queries: int = 8,
        num_generate_queries: int = 8,
        num_global_queries: int = 4,
        num_heads: int = 8,
        num_feature_levels: int = 3,
        presence_threshold: float = 1e-6,
        routing_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})")
        if num_feature_levels < 1:
            raise ValueError("num_feature_levels must be >= 1")
        if min(num_preserve_queries, num_repair_queries, num_generate_queries, num_global_queries) < 1:
            raise ValueError("Each query bank must contain at least one query")

        self.query_dim = query_dim
        self.crossattn_dim = crossattn_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.num_feature_levels = num_feature_levels
        self.presence_threshold = presence_threshold
        self.routing_epsilon = routing_epsilon
        self.query_counts = (num_preserve_queries, num_repair_queries, num_generate_queries)

        self.preserve_queries = nn.Parameter(torch.empty(1, num_preserve_queries, query_dim))
        self.repair_queries = nn.Parameter(torch.empty(1, num_repair_queries, query_dim))
        self.generate_queries = nn.Parameter(torch.empty(1, num_generate_queries, query_dim))
        self.global_queries = nn.Parameter(torch.empty(1, num_global_queries, query_dim))
        self.preserve_null_queries = nn.Parameter(torch.empty(1, num_preserve_queries, query_dim))
        self.repair_null_queries = nn.Parameter(torch.empty(1, num_repair_queries, query_dim))
        self.generate_null_queries = nn.Parameter(torch.empty(1, num_generate_queries, query_dim))

        self.class_embeddings = nn.Parameter(torch.empty(self.NUM_REGION_CLASSES, query_dim))
        self.level_embeddings = nn.Parameter(torch.empty(num_feature_levels, query_dim))
        self.presence_embeddings = nn.Parameter(torch.empty(2, query_dim))
        self.position_proj = nn.Linear(9, query_dim, bias=False)
        self.mass_proj = nn.Linear(1, query_dim)
        self.feature_proj = nn.Linear(feature_dim, query_dim)
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(query_dim, query_dim)
        self.value_proj = nn.Linear(query_dim, query_dim)
        self.attention_out = nn.Linear(query_dim, query_dim)
        self.out_proj = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, crossattn_dim))
        self.init_weights()

    @property
    def num_queries(self) -> int:
        return (
            self.preserve_queries.shape[1]
            + self.repair_queries.shape[1]
            + self.generate_queries.shape[1]
            + self.global_queries.shape[1]
        )

    def init_weights(self) -> None:
        for queries in (
            self.preserve_queries,
            self.repair_queries,
            self.generate_queries,
            self.global_queries,
            self.preserve_null_queries,
            self.repair_null_queries,
            self.generate_null_queries,
            self.class_embeddings,
            self.level_embeddings,
            self.presence_embeddings,
        ):
            nn.init.trunc_normal_(queries, std=0.02)
        for layer in (
            self.position_proj,
            self.mass_proj,
            self.feature_proj,
            self.query_proj,
            self.key_proj,
            self.value_proj,
            self.attention_out,
            self.out_proj[1],
        ):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        # The full model is materialized from meta tensors with to_empty().
        # Restore LayerNorm affine parameters explicitly after materialization.
        self.out_proj[0].reset_parameters()

    @staticmethod
    def _as_feature_list(features: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
        feature_list = [features] if isinstance(features, torch.Tensor) else list(features)
        if not feature_list:
            raise ValueError("features must not be empty")
        for feature in feature_list:
            if feature.dim() != 5:
                raise ValueError(f"Expected feature shape [B,C,T,H,W], got {tuple(feature.shape)}")
        return feature_list

    @staticmethod
    def _normalize_region_probs(region_probs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(region_probs).all():
            raise ValueError("region_probs must contain only finite values")
        if (region_probs < 0).any():
            raise ValueError("region_probs must be nonnegative")
        probs = region_probs * valid_mask
        normalizer = probs.sum(dim=1, keepdim=True)
        return torch.where(normalizer > 0, probs / normalizer.clamp_min(1e-12), torch.zeros_like(probs))

    @staticmethod
    def _resize_spatial(tensor: torch.Tensor, target_hw: tuple[int, int], *, probability: bool) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = tensor.shape
        target_h, target_w = target_hw
        if (height, width) == (target_h, target_w):
            return tensor
        flattened = tensor.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        if probability and target_h <= height and target_w <= width:
            resized = F.interpolate(flattened, size=target_hw, mode="area")
        else:
            resized = F.interpolate(flattened, size=target_hw, mode="bilinear", align_corners=False)
        return resized.reshape(batch_size, num_frames, channels, target_h, target_w).permute(0, 2, 1, 3, 4)

    def _align_routing_inputs(
        self,
        features: list[torch.Tensor],
        region_probs: torch.Tensor | Sequence[torch.Tensor],
        valid_mask: torch.Tensor | Sequence[torch.Tensor] | None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
        probability_list = [region_probs] if isinstance(region_probs, torch.Tensor) else list(region_probs)
        if len(probability_list) not in (1, len(features)):
            raise ValueError("region_probs must be one canonical tensor or one tensor per feature level")

        if valid_mask is None:
            canonical = probability_list[0]
            valid_list = [torch.ones_like(canonical[:, :1])]
        else:
            valid_list = [valid_mask] if isinstance(valid_mask, torch.Tensor) else list(valid_mask)
            if len(valid_list) not in (1, len(features)):
                raise ValueError("valid_mask must be one canonical tensor or one tensor per feature level")

        canonical_probs = probability_list[0]
        canonical_valid = valid_list[0]
        if canonical_probs.dim() != 5 or canonical_probs.shape[1] != self.NUM_REGION_CLASSES:
            raise ValueError(
                f"Expected region_probs shape [B,{self.NUM_REGION_CLASSES},T,H,W], got {tuple(canonical_probs.shape)}"
            )
        if canonical_valid.shape != canonical_probs[:, :1].shape:
            raise ValueError(
                f"valid_mask must have shape {tuple(canonical_probs[:, :1].shape)}, got {tuple(canonical_valid.shape)}"
            )
        canonical_valid = (canonical_valid > 0).to(dtype=canonical_probs.dtype)
        canonical_probs = self._normalize_region_probs(canonical_probs, canonical_valid)
        class_presence = canonical_probs.sum(dim=(2, 3, 4)) > self.presence_threshold

        probs_by_level = []
        valid_by_level = []
        for level_idx, feature in enumerate(features):
            source_probs = probability_list[0] if len(probability_list) == 1 else probability_list[level_idx]
            source_valid = valid_list[0] if len(valid_list) == 1 else valid_list[level_idx]
            if source_probs.shape[0] != feature.shape[0] or source_probs.shape[2] != feature.shape[2]:
                raise ValueError(
                    "Region probabilities must match feature batch and temporal dimensions: "
                    f"feature={tuple(feature.shape)}, region_probs={tuple(source_probs.shape)}"
                )
            if source_valid.shape != source_probs[:, :1].shape:
                raise ValueError("Each valid mask must match its region probability tensor")
            level_probs = self._resize_spatial(source_probs, feature.shape[-2:], probability=True)
            level_valid = self._resize_spatial(source_valid.float(), feature.shape[-2:], probability=False)
            level_valid = (level_valid > 0.5).to(dtype=level_probs.dtype)
            probs_by_level.append(self._normalize_region_probs(level_probs, level_valid))
            valid_by_level.append(level_valid)
        return tuple(probs_by_level), tuple(valid_by_level), class_presence

    def _position_embedding(
        self,
        shape: tuple[int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_frames, height, width = shape
        axes = [
            torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=torch.float32)
            for size in (num_frames, height, width)
        ]
        coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        basis = torch.cat([coordinates, torch.sin(torch.pi * coordinates), torch.cos(torch.pi * coordinates)], dim=-1)
        return self.position_proj(basis.to(dtype=self.position_proj.weight.dtype)).to(dtype=dtype)

    def _build_tokens(self, features: list[torch.Tensor]) -> torch.Tensor:
        tokens = []
        for level_idx, feature in enumerate(features):
            flattened = feature.flatten(2).transpose(1, 2)
            token = self.feature_proj(flattened)
            token = token + self._position_embedding(feature.shape[-3:], device=feature.device, dtype=token.dtype)
            token = token + self.level_embeddings[level_idx].to(dtype=token.dtype)
            tokens.append(token)
        return torch.cat(tokens, dim=1)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tensor.shape
        return tensor.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_tokens, _ = tensor.shape
        return tensor.transpose(1, 2).reshape(batch_size, num_tokens, self.query_dim)

    def _attend(
        self,
        query_slots: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        query = self._split_heads(self.query_proj(query_slots))
        safe_weights = routing_weights.clamp_min(self.routing_epsilon)
        masked_value = torch.finfo(query.dtype).min
        attention_bias = torch.where(
            routing_weights[:, None, None] > 0,
            safe_weights[:, None, None].log(),
            torch.full_like(safe_weights[:, None, None], masked_value),
        ).to(dtype=query.dtype)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_bias, dropout_p=0.0)
        return self.attention_out(self._merge_heads(attended))

    def _typed_queries(
        self,
        query_bank: torch.Tensor,
        null_bank: torch.Tensor,
        class_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
        class_weights: torch.Tensor,
        valid_weights: torch.Tensor,
        class_presence: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = key.shape[0]
        presence = class_presence[:, class_index]
        class_mass = class_weights.sum(dim=1, keepdim=True) / valid_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        slots = query_bank.expand(batch_size, -1, -1)
        slots = slots + self.class_embeddings[class_index]
        slots = slots + self.presence_embeddings[presence.long()][:, None]
        mass_embedding = self.mass_proj(class_mass.to(dtype=self.mass_proj.weight.dtype))
        slots = slots + mass_embedding.to(dtype=slots.dtype)[:, None]

        # Avoid all-negative-infinity attention rows for samples where this
        # class is absent; their result is replaced by the learned null bank.
        routed_weights = torch.where(presence[:, None], class_weights, valid_weights)
        attended = self.out_proj(self._attend(slots, key, value, routed_weights) + slots)

        null_slots = null_bank.expand(batch_size, -1, -1)
        null_slots = null_slots + self.class_embeddings[class_index]
        null_slots = null_slots + self.presence_embeddings[0]
        null_queries = self.out_proj(null_slots)
        return torch.where(presence[:, None, None], attended, null_queries)

    def forward(
        self,
        features: torch.Tensor | Sequence[torch.Tensor],
        region_probs: torch.Tensor | Sequence[torch.Tensor],
        *,
        valid_mask: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> RegionQueryOutput:
        """Return class-routed degradation queries and routing metadata."""
        feature_list = self._as_feature_list(features)
        if len(feature_list) > self.num_feature_levels:
            raise ValueError(
                f"Received {len(feature_list)} feature levels, but only {self.num_feature_levels} are configured"
            )
        batch_size = feature_list[0].shape[0]
        feature_dim = feature_list[0].shape[1]
        for feature in feature_list[1:]:
            if feature.shape[0] != batch_size or feature.shape[1] != feature_dim:
                raise ValueError("All feature levels must share batch size and channel dimension")

        probs_by_level, valid_by_level, class_presence = self._align_routing_inputs(
            feature_list, region_probs, valid_mask
        )
        tokens = self._build_tokens(feature_list)
        key = self._split_heads(self.key_proj(tokens))
        value = self._split_heads(self.value_proj(tokens))
        class_weights = torch.cat([probs.flatten(2) for probs in probs_by_level], dim=2)
        valid_weights = torch.cat([valid.flatten(2) for valid in valid_by_level], dim=2)[:, 0]

        typed_queries = []
        for class_index, (query_bank, null_bank) in enumerate(
            (
                (self.preserve_queries, self.preserve_null_queries),
                (self.repair_queries, self.repair_null_queries),
                (self.generate_queries, self.generate_null_queries),
            )
        ):
            typed_queries.append(
                self._typed_queries(
                    query_bank,
                    null_bank,
                    class_index,
                    key,
                    value,
                    class_weights[:, class_index],
                    valid_weights,
                    class_presence,
                )
            )

        global_slots = self.global_queries.expand(batch_size, -1, -1)
        global_queries = self.out_proj(self._attend(global_slots, key, value, valid_weights) + global_slots)
        queries = torch.cat([*typed_queries, global_queries], dim=1)

        query_class = torch.cat(
            [
                torch.full((count,), class_index, device=queries.device, dtype=torch.long)
                for class_index, count in enumerate(self.query_counts)
            ]
            + [torch.full((self.global_queries.shape[1],), -1, device=queries.device, dtype=torch.long)]
        )
        query_active = torch.cat(
            [
                class_presence[:, class_index : class_index + 1].expand(-1, count)
                for class_index, count in enumerate(self.query_counts)
            ]
            + [torch.ones(batch_size, self.global_queries.shape[1], device=queries.device, dtype=torch.bool)],
            dim=1,
        )
        return RegionQueryOutput(
            queries=queries,
            query_class=query_class,
            query_active=query_active,
            class_presence=class_presence,
            region_probs_by_level=probs_by_level,
        )
