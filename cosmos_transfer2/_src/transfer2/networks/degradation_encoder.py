# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_transfer2._src.transfer2.networks.degradation_query import RegionAwareQueryExtractor, RegionQueryOutput


class VISPRGTeacherQueryBranch(nn.Module):
    """Build typed queries from source appearance, target geometry, and privileged policy semantics.

    Inputs are hidden states from three pretrained VACE control branches in
    channel-last layout ``[B,T,H,W,D]``. Oracle PRG probabilities remain the
    only source of exact routing. PRG-SEG supplies privileged policy semantics,
    HD-map-SEG supplies target geometry, and VIS supplies source appearance.
    """

    def __init__(
        self,
        hidden_dim: int,
        feature_dim: int,
        query_dim: int,
        crossattn_dim: int,
        query_heads: int,
        query_counts: tuple[int, int, int, int] = (4, 8, 8, 4),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.query_counts = tuple(query_counts)
        self.vis_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, feature_dim))
        self.hdmap_seg_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, feature_dim))
        self.prg_seg_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, feature_dim))
        self.query_extractor = RegionAwareQueryExtractor(
            feature_dim=feature_dim,
            query_dim=query_dim,
            crossattn_dim=crossattn_dim,
            num_preserve_queries=query_counts[0],
            num_repair_queries=query_counts[1],
            num_generate_queries=query_counts[2],
            num_global_queries=query_counts[3],
            num_heads=query_heads,
            num_feature_levels=4,
        )

    def init_weights(self) -> None:
        for projection in (self.vis_projection, self.hdmap_seg_projection, self.prg_seg_projection):
            projection[0].reset_parameters()
            nn.init.xavier_uniform_(projection[1].weight)
            nn.init.zeros_(projection[1].bias)
        self.query_extractor.init_weights()

    @staticmethod
    def _to_channel_first(feature: torch.Tensor) -> torch.Tensor:
        return feature.permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def _build_evidence_routes(region_probs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if region_probs.dim() != 5 or region_probs.shape[1] != 3:
            raise ValueError(f"Expected PRG probabilities [B,3,T,H,W], got {tuple(region_probs.shape)}")

        # Privileged PRG semantics and target HD-map geometry are both read
        # through oracle P/R/G support, but remain separate feature streams.
        prg_seg_routes = region_probs
        hdmap_seg_routes = region_probs

        # Local render evidence is useful for identity preservation and
        # residual repair, but must not anchor generation to unsupported RGB.
        local_vis_routes = region_probs.clone()
        local_vis_routes[:, 2] = 0

        # Generate queries may still use nearby spatiotemporal context. Exclude
        # the generate region itself so local render artifacts cannot leak into
        # the generation policy.
        generate = region_probs[:, 2:3]
        contextual_support = F.max_pool3d(generate, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))
        contextual_support = (contextual_support - generate).clamp_min(0.0)
        context_vis_routes = torch.zeros_like(region_probs)
        context_vis_routes[:, 2:3] = contextual_support
        return prg_seg_routes, hdmap_seg_routes, local_vis_routes, context_vis_routes

    def forward(
        self,
        vis_features: torch.Tensor,
        hdmap_seg_features: torch.Tensor,
        prg_seg_features: torch.Tensor,
        region_probs: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        detach_control_features: bool = True,
    ) -> RegionQueryOutput:
        if vis_features.shape != hdmap_seg_features.shape or vis_features.shape != prg_seg_features.shape:
            raise ValueError(
                "VIS, HD-map-SEG, and PRG-SEG features must have identical shapes: "
                f"{tuple(vis_features.shape)}, {tuple(hdmap_seg_features.shape)}, {tuple(prg_seg_features.shape)}"
            )
        if vis_features.dim() != 5 or vis_features.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected control features [B,T,H,W,{self.hidden_dim}], got {tuple(vis_features.shape)}")
        if detach_control_features:
            vis_features = vis_features.detach()
            hdmap_seg_features = hdmap_seg_features.detach()
            prg_seg_features = prg_seg_features.detach()

        projected_vis = self._to_channel_first(self.vis_projection(vis_features))
        projected_hdmap_seg = self._to_channel_first(self.hdmap_seg_projection(hdmap_seg_features))
        projected_seg = self._to_channel_first(self.prg_seg_projection(prg_seg_features))
        evidence_routes = self._build_evidence_routes(region_probs)

        # Three explicit evidence streams make the policy semantics
        # operational rather than relying only on differently named slots.
        feature_levels = [
            projected_seg,
            projected_hdmap_seg,
            projected_vis,
            projected_vis,
        ]
        return self.query_extractor(feature_levels, evidence_routes, valid_mask=valid_mask)
