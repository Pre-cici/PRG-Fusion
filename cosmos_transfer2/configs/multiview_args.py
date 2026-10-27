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

"""Inference argument schemas for auto multiview runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, ClassVar

import pydantic

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2.configs.common import MODEL_CHECKPOINTS, ModelKey, ModelVariant, get_model_literal
from cosmos_transfer2.configs.inference_args import (
    CommonInferenceArguments,
    CommonSetupArguments,
    ResolvedFilePath,
    get_overrides_cls,
)

DEFAULT_MODEL_KEY = ModelKey(variant=ModelVariant.AUTO_MULTIVIEW)
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_MODEL_KEY]
DEFAULT_MULTIVIEW_CONFIG_FILE = "cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py"
MULTIVIEW_CAMERA_KEYS: tuple[str, ...] = (
    "front_wide",
    "cross_right",
    "rear_right",
    "rear",
    "rear_left",
    "cross_left",
    "front_tele",
)


class MultiviewSetupArguments(CommonSetupArguments):
    """Setup arguments for auto multiview inference."""

    model: get_model_literal([ModelVariant.AUTO_MULTIVIEW]) = DEFAULT_MODEL_KEY.name
    config_file: str = DEFAULT_MULTIVIEW_CONFIG_FILE


class ViewConfig(pydantic.BaseModel):
    """Configuration for a single camera view."""

    model_config = pydantic.ConfigDict(extra="forbid")

    input_path: ResolvedFilePath | None = None
    control_path: ResolvedFilePath | None = None
    num_conditional_frames_per_view: int = pydantic.Field(default=0, ge=0, le=5)


class MultiviewInferenceArguments(CommonInferenceArguments):
    """Inference arguments for auto multiview generation."""

    view_key_order: ClassVar[tuple[str, ...]] = MULTIVIEW_CAMERA_KEYS

    num_conditional_frames: int = pydantic.Field(default=1)
    control_weight: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 1.0

    front_wide: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    rear: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    rear_left: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    rear_right: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    cross_left: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    cross_right: ViewConfig = pydantic.Field(default_factory=ViewConfig)
    front_tele: ViewConfig = pydantic.Field(default_factory=ViewConfig)

    fps: pydantic.PositiveInt = 10
    num_steps: int = pydantic.Field(default=1 if SMOKE else 35)
    enable_autoregressive: bool = False
    num_chunks: int = pydantic.Field(default=2, ge=1)
    chunk_overlap: int = pydantic.Field(default=1)
    save_combined_views: bool = True

    @pydantic.model_validator(mode="after")
    def validate_input_paths(self) -> "MultiviewInferenceArguments":
        active_views = self.active_view_configs
        if not active_views:
            raise ValueError("At least one view configuration with a control_path must be provided.")

        if self.num_conditional_frames > 0 and not self.enable_autoregressive:
            missing_input_paths = [
                view_name for view_name, view_config in active_views if view_config.input_path is None
            ]
            if missing_input_paths:
                raise ValueError(
                    "input_path is required for all active views when num_conditional_frames > 0. "
                    f"Missing input_path for views: {', '.join(missing_input_paths)}"
                )

        if self.enable_autoregressive:
            per_view_frames = [view_config.num_conditional_frames_per_view for _, view_config in active_views]
            if any(frames not in (0, 1, 5) for frames in per_view_frames):
                raise ValueError("num_conditional_frames_per_view must be one of [0, 1, 5] in autoregressive mode.")
            if any(frames == 0 for frames in per_view_frames) and not all(frames == 0 for frames in per_view_frames):
                raise ValueError(
                    "num_conditional_frames_per_view must be consistent across all active views in autoregressive mode."
                )
            if any(frames > 0 for frames in per_view_frames):
                missing_input_paths = [
                    view_name for view_name, view_config in active_views if view_config.input_path is None
                ]
                if missing_input_paths:
                    raise ValueError(
                        "input_path is required for all active views when autoregressive conditioning uses input frames. "
                        f"Missing input_path for views: {', '.join(missing_input_paths)}"
                    )
        return self

    @property
    def active_view_configs(self) -> list[tuple[str, ViewConfig]]:
        active_views: list[tuple[str, ViewConfig]] = []
        for view_name in self.view_key_order:
            view_config = getattr(self, view_name)
            if view_config.control_path is not None:
                active_views.append((view_name, view_config))
        return active_views

    @property
    def active_camera_keys(self) -> tuple[str, ...]:
        return tuple(view_name for view_name, _ in self.active_view_configs)

    @property
    def input_and_control_paths(self) -> dict[str, ResolvedFilePath | None]:
        input_and_control_paths: dict[str, ResolvedFilePath | None] = {}
        for view_name, view_config in self.active_view_configs:
            input_and_control_paths[f"{view_name}_input"] = view_config.input_path
            assert view_config.control_path is not None
            input_and_control_paths[f"{view_name}_control"] = view_config.control_path
        return input_and_control_paths


MultiviewInferenceOverrides = get_overrides_cls(
    MultiviewInferenceArguments,
    exclude=[
        "name",
        "front_wide",
        "rear",
        "rear_left",
        "rear_right",
        "cross_left",
        "cross_right",
        "front_tele",
    ],
)


class MultiviewConfigFile(pydantic.BaseModel):
    """Top-level multiview config loaded from YAML."""

    model_config = pydantic.ConfigDict(extra="forbid")

    config_path: Path | None = None
    setup: dict[str, Any] = pydantic.Field(default_factory=dict)
    inference: list[dict[str, Any]]
