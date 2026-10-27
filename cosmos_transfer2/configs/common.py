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

"""Shared registries and defaults used by training and inference configs."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from cosmos_oss.checkpoints_transfer2 import register_checkpoints

from cosmos_transfer2._src.imaginaire.flags import EXPERIMENTAL_CHECKPOINTS
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import CheckpointConfig

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelVariant(str, enum.Enum):
    DEPTH = "depth"
    EDGE = "edge"
    SEG = "seg"
    VIS = "vis"
    AUTO_MULTIVIEW = "auto/multiview"
    ROBOT_MULTIVIEW_AGIBOT_DEPTH = "robot/multiview-agibot-depth"
    ROBOT_MULTIVIEW_AGIBOT_EDGE = "robot/multiview-agibot-edge"
    ROBOT_MULTIVIEW_AGIBOT_VIS = "robot/multiview-agibot-vis"
    ROBOT_MULTIVIEW_AGIBOT_SEG = "robot/multiview-agibot-seg"
    ROBOT_MULTIVIEW_MANY_CAMERA = "robot/multiview-many-camera"


@dataclass(frozen=True, kw_only=True)
class ModelKey:
    variant: ModelVariant = ModelVariant.EDGE
    distilled: bool = False

    @cached_property
    def name(self) -> str:
        if self.distilled:
            return f"{self.variant.value}/distilled"
        return self.variant.value

    def __str__(self) -> str:
        return self.name


BASE_MODEL_VARIANTS = [
    ModelVariant.EDGE,
    ModelVariant.DEPTH,
    ModelVariant.SEG,
    ModelVariant.VIS,
]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_KEY = ModelKey()
DEFAULT_BASE_EXPERIMENT = "vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow"
DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a game playing, with bad crappy graphics and cartoonish frames. "
    "It represents a recording of old outdated games. The lighting looks very fake. "
    "The textures are very raw and basic. The geometries are very primitive. "
    "The images are very pixelated and of poor CG quality. "
    "There are many subtitles in the footage. Overall, the video is unrealistic at all."
)
CONTROL_KEYS = ["edge", "vis", "depth", "seg"]

# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

register_checkpoints()

MODEL_CHECKPOINTS = {
    ModelKey(variant=ModelVariant.DEPTH): CheckpointConfig.from_uri("626e6618-bfcd-4d9a-a077-1409e2ce353f"),
    ModelKey(variant=ModelVariant.EDGE): CheckpointConfig.from_uri("61f5694b-0ad5-4ecd-8ad7-c8545627d125"),
    ModelKey(variant=ModelVariant.SEG): CheckpointConfig.from_uri("5136ef49-6d8d-42e8-8abf-7dac722a304a"),
    ModelKey(variant=ModelVariant.VIS): CheckpointConfig.from_uri("ba2f44f2-c726-4fe7-949f-597069d9b91c"),
    ModelKey(variant=ModelVariant.AUTO_MULTIVIEW): CheckpointConfig.from_uri("4ecc66e9-df19-4aed-9802-0d11e057287a"),
}
if EXPERIMENTAL_CHECKPOINTS:
    MODEL_CHECKPOINTS |= {
        ModelKey(variant=ModelVariant.EDGE, distilled=True): CheckpointConfig.from_uri(
            "41f07f13-f2e4-4e34-ba4c-86f595acbc20"
        ),
        ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT_DEPTH): CheckpointConfig.from_uri(
            "32514ba1-6d05-4ce5-997d-a3b5bf894cab"
        ),
        ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT_EDGE): CheckpointConfig.from_uri(
            "fffbd388-89c9-4604-ad4f-6c6b36272c48"
        ),
        ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT_VIS): CheckpointConfig.from_uri(
            "2eca9f80-bf8f-4257-b05f-278065d21500"
        ),
        ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT_SEG): CheckpointConfig.from_uri(
            "c5a9a58b-7f3e-4b45-9e5d-8f7b3d4e5a6c"
        ),
        ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_MANY_CAMERA): CheckpointConfig.from_uri(
            "a8794d70-842c-44a5-95bb-9010d5ace7be"
        ),
    }

MODEL_KEYS = {model_key.name: model_key for model_key in MODEL_CHECKPOINTS}


# pyrefly: ignore  # invalid-annotation
def get_model_literal(variants: list[ModelVariant] | None = None) -> Literal:
    model_names: list[str] = []
    for model_key in MODEL_CHECKPOINTS:
        if variants is not None and model_key.variant not in variants:
            continue
        model_names.append(model_key.name)
    # pyrefly: ignore  # bad-return, invalid-literal
    return Literal[tuple(model_names)]


# ---------------------------------------------------------------------------
# Experiment / preset registry
# ---------------------------------------------------------------------------

DEFAULT_SINGLEVIEW_TRAIN_CONFIG_FILE = "cosmos_transfer2/train/base.py"
DEFAULT_INFERENCE_CONFIG_FILE = "cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py"
DEFAULT_DISTILLED_INFERENCE_CONFIG_FILE = "cosmos_transfer2/_src/interactive/configs/registry_transfer2p5.py"


@dataclass(frozen=True)
class ModelPreset:
    model_key: ModelKey
    checkpoint_path: str
    experiment: str
    config_file: str


@dataclass(frozen=True)
class TrainExperimentPreset:
    name: str
    base_config_file: str = DEFAULT_SINGLEVIEW_TRAIN_CONFIG_FILE


MODEL_PRESETS = {
    model_key.name: ModelPreset(
        model_key=model_key,
        checkpoint_path=checkpoint.s3.uri,
        experiment=checkpoint.experiment,
        config_file=(DEFAULT_DISTILLED_INFERENCE_CONFIG_FILE if model_key.distilled else DEFAULT_INFERENCE_CONFIG_FILE),
    )
    for model_key, checkpoint in MODEL_CHECKPOINTS.items()
}


def get_model_preset(model_name: str) -> ModelPreset:
    return MODEL_PRESETS[model_name]


def resolve_train_base_config_file(explicit_config_file: str | None = None) -> str:
    if explicit_config_file:
        return explicit_config_file
    return DEFAULT_SINGLEVIEW_TRAIN_CONFIG_FILE


__all__ = [
    "BASE_MODEL_VARIANTS",
    "CONTROL_KEYS",
    "DEFAULT_BASE_EXPERIMENT",
    "DEFAULT_DISTILLED_INFERENCE_CONFIG_FILE",
    "DEFAULT_INFERENCE_CONFIG_FILE",
    "DEFAULT_MODEL_KEY",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_SINGLEVIEW_TRAIN_CONFIG_FILE",
    "MODEL_CHECKPOINTS",
    "MODEL_KEYS",
    "MODEL_PRESETS",
    "ModelKey",
    "ModelPreset",
    "ModelVariant",
    "TrainExperimentPreset",
    "get_model_literal",
    "get_model_preset",
    "resolve_train_base_config_file",
]
