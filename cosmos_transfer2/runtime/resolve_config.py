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

"""Resolve validated external inputs into final runtime configs."""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from cosmos_transfer2.configs import InferenceArguments, InferenceConfigFile, SetupArguments, TrainConfigFile
from cosmos_transfer2.configs.common import CONTROL_KEYS, DEFAULT_MODEL_KEY
from cosmos_transfer2.configs.multiview_args import (
    DEFAULT_MODEL_KEY as DEFAULT_MULTIVIEW_MODEL_KEY,
)
from cosmos_transfer2.configs.multiview_args import (
    DEFAULT_MULTIVIEW_CONFIG_FILE,
    MULTIVIEW_CAMERA_KEYS,
    MultiviewConfigFile,
    MultiviewInferenceArguments,
    MultiviewSetupArguments,
)

if TYPE_CHECKING:
    import pydantic

    from cosmos_transfer2._src.imaginaire.config import Config


@dataclass(frozen=True)
class ResolvedTrainConfig:
    args: TrainConfigFile
    base_config_file: str
    overrides: list[str]
    final_config: "Config"
    output_dir: Path


@dataclass(frozen=True)
class ResolvedInferenceConfig:
    setup_args: SetupArguments
    sample_args: list[InferenceArguments]
    batch_hint_keys: list[str]
    output_dir: Path
    sample_overrides: "pydantic.BaseModel | None"
    final_config: dict[str, Any]


@dataclass(frozen=True)
class ResolvedMultiviewConfig:
    setup_args: MultiviewSetupArguments
    sample_args: list[MultiviewInferenceArguments]
    output_dir: Path


def _resolve_inference_output_dir(
    setup_args: SetupArguments,
    config_path: Path | None,
    inference_samples: list[InferenceArguments],
) -> Path:
    from cosmos_transfer2._src.imaginaire.utils import log

    if setup_args.output_dir is not None:
        return setup_args.output_dir

    if len(inference_samples) > 1:
        if config_path is not None:
            run_name = config_path.stem
        else:
            run_name = f"batch_plus{len(inference_samples) - 1}"
        output_dir = Path("outputs") / "inference" / run_name
        log.info(f"No output_dir provided for batch inference, using fixed directory: {output_dir}")
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = inference_samples[0].name
    output_dir = Path("outputs") / "debug" / f"{timestamp}_{run_name}"
    log.info(f"No output_dir provided, using auto-generated directory: {output_dir}")
    return output_dir


def _resolve_multiview_output_dir(
    setup_args: MultiviewSetupArguments,
    config_path: Path | None,
    inference_samples: list[MultiviewInferenceArguments],
) -> Path:
    if setup_args.output_dir is not None:
        return setup_args.output_dir

    if len(inference_samples) > 1:
        run_name = config_path.stem if config_path is not None else "multiview-batch"
        return Path("outputs") / "inference" / run_name

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and config_path is not None:
        stamp_dir = Path("outputs") / "debug"
        stamp_dir.mkdir(parents=True, exist_ok=True)
        run_token = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "distributed"
        stamp_file = stamp_dir / f".{config_path.stem}.{run_token}.stamp"
        if rank == 0:
            stamp_file.write_text(timestamp)
        else:
            deadline = time.time() + 30
            while not stamp_file.exists():
                if time.time() > deadline:
                    raise TimeoutError(f"Timed out waiting for output stamp file: {stamp_file}")
                time.sleep(0.1)
            timestamp = stamp_file.read_text().strip() or timestamp
    run_name = inference_samples[0].name
    return Path("outputs") / "debug" / f"{timestamp}_{run_name}"


def resolve_train_config(args: TrainConfigFile) -> ResolvedTrainConfig:
    from cosmos_transfer2._src.imaginaire.flags import SMOKE
    from cosmos_transfer2._src.imaginaire.utils.config_helper import get_config_module, override

    config_module = get_config_module(args.setup.config_file)
    base_config = importlib.import_module(config_module).make_config()

    overrides = args.to_override_tokens(smoke=SMOKE)

    final_config = override(base_config, overrides)
    output_dir = Path(final_config.job.path_local)
    return ResolvedTrainConfig(
        args=args,
        base_config_file=args.setup.config_file,
        overrides=overrides,
        final_config=final_config,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# YAML config loading helpers
# ---------------------------------------------------------------------------


def _resolve_path_value(value: Any, *, config_dir: Path) -> Any:
    if value is None or not isinstance(value, (str, Path)):
        return value
    path_value = Path(value)
    return path_value if path_value.is_absolute() else config_dir / path_value


def _resolve_sample_paths(sample: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    resolved = dict(sample)
    for key in [
        "prompt_path",
        "video_path",
        "image_context_path",
        "guided_generation_mask",
        "prg_aux_dir",
        "hdmap_path",
    ]:
        if key in resolved:
            resolved[key] = _resolve_path_value(resolved[key], config_dir=config_dir)

    for control_key in ["edge", "depth", "vis", "seg", "lidar"]:
        control = resolved.get(control_key)
        if not isinstance(control, dict):
            continue
        control = dict(control)
        for key in ["control_path", "mask_path"]:
            if key in control:
                control[key] = _resolve_path_value(control[key], config_dir=config_dir)
        resolved[control_key] = control
    return resolved


def _resolve_multiview_sample_paths(sample: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    resolved = dict(sample)
    if "prompt_path" in resolved:
        resolved["prompt_path"] = _resolve_path_value(resolved["prompt_path"], config_dir=config_dir)

    for view_key in MULTIVIEW_CAMERA_KEYS:
        view_config = resolved.get(view_key)
        if not isinstance(view_config, dict):
            continue
        view_config = dict(view_config)
        for path_key in ["input_path", "control_path"]:
            if path_key in view_config:
                view_config[path_key] = _resolve_path_value(view_config[path_key], config_dir=config_dir)
        resolved[view_key] = view_config
    return resolved


def _infer_setup_model(sample_dicts: list[dict[str, Any]]) -> str:
    for sample in sample_dicts:
        for key in CONTROL_KEYS:
            if sample.get(key) is not None:
                return key
    return DEFAULT_MODEL_KEY.name


def load_inference_config(config_path: Path) -> InferenceConfigFile:
    raw_config = yaml.safe_load(config_path.read_text())
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError("YAML config must be a mapping")

    sample_dicts = [
        _resolve_sample_paths(sample, config_dir=config_path.parent.resolve())
        for sample in raw_config.get("inference", [])
    ]

    setup = dict(raw_config.get("setup") or {})
    if "model" not in setup or setup["model"] is None:
        setup["model"] = _infer_setup_model(sample_dicts)
    if "output_dir" in setup and setup["output_dir"] is not None:
        output_dir = Path(setup["output_dir"])
        setup["output_dir"] = output_dir if output_dir.is_absolute() else config_path.parent.resolve() / output_dir

    config_data = dict(raw_config)
    config_data["setup"] = setup
    config_data["inference"] = sample_dicts
    return InferenceConfigFile.model_validate(config_data | {"config_path": config_path.resolve()})


def load_multiview_config(config_path: Path) -> MultiviewConfigFile:
    raw_config = yaml.safe_load(config_path.read_text())
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError("YAML config must be a mapping")

    sample_dicts = [
        _resolve_multiview_sample_paths(sample, config_dir=config_path.parent.resolve())
        for sample in raw_config.get("inference", [])
    ]

    setup = dict(raw_config.get("setup") or {})
    setup.setdefault("model", DEFAULT_MULTIVIEW_MODEL_KEY.name)
    setup.setdefault("config_file", DEFAULT_MULTIVIEW_CONFIG_FILE)
    if "output_dir" in setup and setup["output_dir"] is not None:
        output_dir = Path(setup["output_dir"])
        setup["output_dir"] = output_dir if output_dir.is_absolute() else config_path.parent.resolve() / output_dir

    config_data = dict(raw_config)
    config_data["setup"] = setup
    config_data["inference"] = sample_dicts
    return MultiviewConfigFile.model_validate(config_data | {"config_path": config_path.resolve()})


def load_train_config(config_path: Path) -> TrainConfigFile:
    raw_config = yaml.safe_load(config_path.read_text())
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError("YAML config must be a mapping")

    config_data = dict(raw_config)
    train = dict(raw_config.get("train") or {})
    if "dataset_dir" in train and train["dataset_dir"] is not None:
        train["dataset_dir"] = _resolve_path_value(train["dataset_dir"], config_dir=config_path.parent.resolve())
    if "val_dataset_dir" in train and train["val_dataset_dir"] is not None:
        train["val_dataset_dir"] = _resolve_path_value(
            train["val_dataset_dir"], config_dir=config_path.parent.resolve()
        )
    config_data["train"] = train
    return TrainConfigFile.model_validate(config_data | {"config_path": config_path.resolve()})


def resolve_inference_config(
    *,
    setup_args: SetupArguments,
    sample_dicts: list[dict[str, Any]],
    config_path: Path | None = None,
    sample_overrides: "pydantic.BaseModel | None" = None,
) -> ResolvedInferenceConfig:
    override_data = {} if sample_overrides is None else sample_overrides.model_dump(exclude_none=True)
    inference_samples = [InferenceArguments.model_validate(sample | override_data) for sample in sample_dicts]
    inference_control_keys = [*CONTROL_KEYS, "lidar"]
    batch_hint_keys = sorted(
        {key for sample in inference_samples for key in sample.hint_keys},
        key=lambda key: inference_control_keys.index(key),
    )
    output_dir = _resolve_inference_output_dir(setup_args, config_path, inference_samples)
    resolved_setup_args = setup_args.model_copy(update={"output_dir": output_dir})
    return ResolvedInferenceConfig(
        setup_args=resolved_setup_args,
        sample_args=inference_samples,
        batch_hint_keys=batch_hint_keys,
        output_dir=output_dir,
        sample_overrides=sample_overrides,
        final_config={
            "setup": resolved_setup_args.model_dump(mode="json"),
            "samples": [sample.model_dump(mode="json") for sample in inference_samples],
            "batch_hint_keys": batch_hint_keys,
            "config_path": None if config_path is None else str(config_path),
        },
    )


def resolve_multiview_config(
    config: MultiviewConfigFile,
    sample_overrides: "pydantic.BaseModel | None" = None,
) -> ResolvedMultiviewConfig:
    setup_args = MultiviewSetupArguments.model_validate(config.setup)
    override_data = {} if sample_overrides is None else sample_overrides.model_dump(exclude_none=True)
    samples = [MultiviewInferenceArguments.model_validate(sample | override_data) for sample in config.inference]
    max_active_views = max(len(sample.active_camera_keys) for sample in samples)
    if setup_args.context_parallel_size is None:
        raise ValueError("context_parallel_size must be set for multiview inference")
    if max_active_views > setup_args.context_parallel_size:
        raise ValueError(
            "context_parallel_size must be greater than or equal to the number of active views. "
            f"Got context_parallel_size={setup_args.context_parallel_size}, max_active_views={max_active_views}."
        )
    output_dir = _resolve_multiview_output_dir(setup_args, config.config_path, samples)
    resolved_setup_args = setup_args.model_copy(update={"output_dir": output_dir})
    return ResolvedMultiviewConfig(setup_args=resolved_setup_args, sample_args=samples, output_dir=output_dir)
