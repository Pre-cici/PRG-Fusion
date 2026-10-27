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

"""Training argument schemas for YAML-driven train runs."""

from __future__ import annotations

from pathlib import Path

import pydantic

from cosmos_transfer2.configs.common import resolve_train_base_config_file
from cosmos_transfer2.configs.inference_args import ResolvedDirectoryPath


def _serialize_override_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_serialize_override_value(item) for item in value) + "]"
    return str(value)


class TrainSetupArguments(pydantic.BaseModel):
    """Run-level train settings owned by the external YAML layer."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    experiment: str
    config_file: str = ""
    project: str | None = None
    group: str | None = None
    name: str | None = None
    wandb_mode: str | None = None
    dryrun: bool = False
    profile: bool = False

    @pydantic.model_validator(mode="before")
    @classmethod
    def populate_defaults(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data["config_file"] = resolve_train_base_config_file(data.get("config_file") or None)
        return data


class TrainArguments(pydantic.BaseModel):
    """Common structured training arguments exposed in YAML."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    dataset_dir: ResolvedDirectoryPath
    val_dataset_dir: ResolvedDirectoryPath | None = None
    batch_size: pydantic.PositiveInt | None = None
    num_workers: int | None = pydantic.Field(default=None, ge=0)
    resolution: str | None = None
    video_size: tuple[pydantic.PositiveInt, pydantic.PositiveInt] | None = None
    num_frames: pydantic.PositiveInt | None = None
    max_iter: pydantic.PositiveInt | None = None
    logging_iter: pydantic.PositiveInt | None = None
    validation_iter: pydantic.PositiveInt | None = None
    run_validation: bool | None = None
    checkpoint_path: str | None = None
    load_training_state: bool | None = None
    strict_resume: bool | None = None
    require_hdmap: bool | None = None
    require_lidar: bool | None = None
    lidar_mask_top_rows: int | None = pydantic.Field(default=None, ge=0)
    load_prg_labels: bool | None = None


class TrainConfigFile(pydantic.BaseModel):
    """Top-level train config loaded from a YAML file."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    config_path: Path | None = None
    setup: TrainSetupArguments
    train: TrainArguments
    overrides: list[str] = pydantic.Field(default_factory=list)

    def to_override_tokens(self, *, smoke: bool = False) -> list[str]:
        overrides = [
            "--",
            f"experiment={self.setup.experiment}",
            f"dataloader_train.dataset.dataset_dir={_serialize_override_value(self.train.dataset_dir)}",
            f"++dataloader_train.sampler.dataset.dataset_dir={_serialize_override_value(self.train.dataset_dir)}",
        ]

        if self.train.val_dataset_dir is not None:
            overrides.extend(
                [
                    f"dataloader_val.dataset.dataset_dir={_serialize_override_value(self.train.val_dataset_dir)}",
                    f"++dataloader_val.sampler.dataset.dataset_dir={_serialize_override_value(self.train.val_dataset_dir)}",
                ]
            )

        structured_mappings: list[tuple[object | None, str]] = [
            (self.train.batch_size, "dataloader_train.batch_size"),
            (self.train.num_workers, "dataloader_train.num_workers"),
            (self.train.resolution, "dataloader_train.dataset.resolution"),
            (self.train.resolution, "++dataloader_train.sampler.dataset.resolution"),
            (self.train.video_size, "dataloader_train.dataset.video_size"),
            (self.train.video_size, "++dataloader_train.sampler.dataset.video_size"),
            (self.train.num_frames, "dataloader_train.dataset.num_frames"),
            (self.train.num_frames, "++dataloader_train.sampler.dataset.num_frames"),
            (self.train.max_iter, "trainer.max_iter"),
            (self.train.logging_iter, "trainer.logging_iter"),
            (self.train.validation_iter, "trainer.validation_iter"),
            (self.train.run_validation, "trainer.run_validation"),
            (self.train.checkpoint_path, "checkpoint.load_path"),
            (self.train.load_training_state, "checkpoint.load_training_state"),
            (self.train.strict_resume, "checkpoint.strict_resume"),
            (self.train.require_hdmap, "dataloader_train.dataset.require_hdmap"),
            (self.train.require_hdmap, "++dataloader_train.sampler.dataset.require_hdmap"),
            (self.train.require_lidar, "dataloader_train.dataset.require_lidar"),
            (self.train.require_lidar, "++dataloader_train.sampler.dataset.require_lidar"),
            (self.train.lidar_mask_top_rows, "dataloader_train.dataset.lidar_mask_top_rows"),
            (self.train.lidar_mask_top_rows, "++dataloader_train.sampler.dataset.lidar_mask_top_rows"),
            (self.train.load_prg_labels, "dataloader_train.dataset.load_prg_labels"),
            (self.train.load_prg_labels, "++dataloader_train.sampler.dataset.load_prg_labels"),
            (self.setup.project, "job.project"),
            (self.setup.group, "job.group"),
            (self.setup.name, "job.name"),
            (self.setup.wandb_mode, "job.wandb_mode"),
        ]
        for value, key in structured_mappings:
            if value is None:
                continue
            overrides.append(f"{key}={_serialize_override_value(value)}")

        overrides.extend(self.overrides)
        if smoke:
            overrides.extend(
                [
                    "trainer.max_iter=2",
                    "trainer.logging_iter=1",
                    "trainer.validation_iter=1",
                ]
            )
        return overrides
