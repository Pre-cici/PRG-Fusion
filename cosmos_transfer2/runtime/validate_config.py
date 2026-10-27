# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate resolved runtime configs before execution."""

from __future__ import annotations

from typing import Any

from cosmos_transfer2.runtime.resolve_config import (
    ResolvedInferenceConfig,
    ResolvedMultiviewConfig,
    ResolvedTrainConfig,
)


def validate_train_config(resolved: ResolvedTrainConfig) -> None:
    resolved.final_config.validate()


def validate_inference_config(resolved: ResolvedInferenceConfig) -> None:
    if not resolved.sample_args:
        raise ValueError("No inference samples resolved")
    teacher_samples = [sample for sample in resolved.sample_args if sample.use_teacher_degradation_query]
    if teacher_samples and not resolved.setup_args.use_spatial_degradation_query:
        raise ValueError("Teacher degradation query inference requires setup.use_spatial_degradation_query=true")
    missing_labels = [sample.name for sample in teacher_samples if sample.prg_aux_dir is None]
    if missing_labels:
        raise ValueError(f"Teacher degradation query samples are missing prg_aux_dir: {missing_labels}")
    missing_hdmap = [sample.name for sample in teacher_samples if sample.hdmap_path is None]
    if missing_hdmap:
        raise ValueError(f"Teacher degradation query samples are missing hdmap_path: {missing_hdmap}")
    gate_samples = [sample for sample in resolved.sample_args if sample.use_prg_dense_gate]
    missing_gate_labels = [sample.name for sample in gate_samples if sample.prg_aux_dir is None]
    if missing_gate_labels:
        raise ValueError(f"PRG dense-gate samples are missing prg_aux_dir: {missing_gate_labels}")
    invalid_gate_samples = [sample.name for sample in gate_samples if sample.use_teacher_degradation_query]
    if invalid_gate_samples:
        raise ValueError(f"Pure PRG dense gating cannot be combined with teacher Query mode: {invalid_gate_samples}")
    if resolved.setup_args.prg_dense_gate_hdmap_lane_as_generate:
        if not gate_samples:
            raise ValueError("HD-map lane Generate routing requires at least one PRG dense-gate sample")
        if resolved.setup_args.prg_dense_gate_hdmap_lane_color_threshold <= 0:
            raise ValueError("HD-map lane Generate routing color threshold must be positive")
        dilation = resolved.setup_args.prg_dense_gate_hdmap_lane_dilation
        if dilation <= 0 or dilation % 2 == 0:
            raise ValueError("HD-map lane Generate routing dilation must be a positive odd integer")
        missing_lane_hdmap = [sample.name for sample in gate_samples if sample.hdmap_path is None]
        if missing_lane_hdmap:
            raise ValueError(f"HD-map lane Generate routing samples are missing hdmap_path: {missing_lane_hdmap}")


def validate_image_context_conditioning(inference: Any, samples: list[Any]) -> None:
    if not any(sample.image_context_path is not None for sample in samples):
        return

    model = inference.model
    conditioner_embedders = getattr(model.conditioner, "embedders", {})
    has_image_embedder = "reference_image_context" in conditioner_embedders
    has_image_attention = getattr(model.net, "extra_image_context_dim", None) is not None
    if not has_image_embedder or not has_image_attention:
        raise ValueError(
            "image_context_path requires an image-context experiment with both the "
            "reference_image_context conditioner embedder and image cross-attention; "
            f"got experiment={inference.registered_exp_name!r}"
        )


def validate_multiview_config(resolved: ResolvedMultiviewConfig) -> None:
    if not resolved.sample_args:
        raise ValueError("No multiview samples resolved")
