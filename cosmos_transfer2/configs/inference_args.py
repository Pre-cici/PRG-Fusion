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

"""Inference argument schemas for singleview control-conditioned runs."""

from __future__ import annotations

import enum
import json
import os
import sys
from functools import cache, cached_property
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, Optional, TypeVar

import pydantic
import tyro
from pydantic_core import PydanticUndefined
from typing_extensions import Self

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2.configs.common import (
    BASE_MODEL_VARIANTS,
    CONTROL_KEYS,
    DEFAULT_MODEL_KEY,
    DEFAULT_NEGATIVE_PROMPT,
    MODEL_KEYS,
    ModelKey,
    get_model_literal,
    get_model_preset,
)


@cache
def is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def path_to_str(v: Path | None) -> str | None:
    if v is None:
        return None
    return str(v)


def load_callable(name: str):
    idx = name.rfind(".")
    assert idx > 0, "expected <module_name>.<identifier>"
    module_name = name[0:idx]
    fn_name = name[idx + 1 :]
    module = import_module(module_name)
    return getattr(module, fn_name)


_PydanticModelT = TypeVar("_PydanticModelT", bound=pydantic.BaseModel)


def get_overrides_cls(cls: type[_PydanticModelT], *, exclude: list[str] | None = None) -> type[pydantic.BaseModel]:
    """Build a nullable override model for CLI/file-level overrides."""
    names = [name for name in cls.model_fields.keys() if exclude is None or name not in exclude]
    fields = {}
    for name in names:
        model_field = cls.model_fields[name]
        behavior_hint = (
            f"(default: {model_field.default})"
            if model_field.default is not PydanticUndefined
            else "(default: None) (required)"
        )
        annotation = Annotated[
            Optional[cls.model_fields[name].rebuild_annotation()],
            tyro.conf.arg(help_behavior_hint=behavior_hint),
        ]
        fields[name] = (annotation, pydantic.Field(default=None, description=model_field.description))
    return pydantic.create_model(f"{cls.__name__}Overrides", **fields)


def _get_root_exception(exception: Exception) -> Exception:
    if exception.__cause__ is not None:
        return _get_root_exception(exception.__cause__)
    if exception.__context__ is not None:
        return _get_root_exception(exception.__context__)
    return exception


def handle_tyro_exception(exception: Exception) -> NoReturn:
    root_exception = _get_root_exception(exception)
    if isinstance(root_exception, pydantic.ValidationError):
        if is_rank0():
            print(root_exception, file=sys.stderr)
        sys.exit(1)
    raise exception


def _resolve_path(v: Path) -> Path:
    return v.expanduser().absolute()


ResolvedFilePath = Annotated[pydantic.FilePath, pydantic.AfterValidator(_resolve_path)]
ResolvedDirectoryPath = Annotated[pydantic.DirectoryPath, pydantic.AfterValidator(_resolve_path)]


def _validate_checkpoint_path(v: str) -> str:
    from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_uri

    return get_checkpoint_uri(v, check_exists=True)


CheckpointPath = Annotated[str, pydantic.AfterValidator(_validate_checkpoint_path)]
Guidance = Annotated[int, pydantic.Field(ge=0, le=7)]
ControlWeight = Annotated[float, pydantic.Field(ge=0.0, le=1.0, step=0.01)]
Threshold = Literal["very_low", "low", "medium", "high", "very_high"]
ModelName = get_model_literal()
BaseModelName = get_model_literal(BASE_MODEL_VARIANTS)


class CompileMode(str, enum.Enum):
    NONE = "none"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class CommonSetupArguments(pydantic.BaseModel):
    """Common arguments for model setup."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    output_dir: Path | None = None
    model: ModelName = DEFAULT_MODEL_KEY.name
    checkpoint_path: CheckpointPath | None = None
    prg_checkpoint_path: CheckpointPath | None = None
    control_branch_checkpoint_path: CheckpointPath | None = None
    control_branch_index: pydantic.NonNegativeInt = 1
    control_branch_indices: tuple[pydantic.NonNegativeInt, ...] = ()
    experiment: str | None = None
    config_file: str = ""
    context_parallel_size: pydantic.PositiveInt | None = None
    disable_guardrails: bool = True
    offload_guardrail_models: bool = True
    offload_diffusion_model: bool = True
    offload_text_encoder: bool = True
    offload_image_context_encoder: bool = True
    offload_tokenizer: bool = True
    keep_going: bool = True
    profile: bool = False
    benchmark: bool = False
    compile_tokenizer: CompileMode = CompileMode.NONE
    enable_parallel_tokenizer: bool = False
    parallel_tokenizer_grid: tuple[int, int] = (-1, -1)
    use_spatial_degradation_query: bool = False
    experiment_overrides: tuple[str, ...] = ()

    @cached_property
    def enable_guardrails(self) -> bool:
        return not self.disable_guardrails

    @cached_property
    def model_key(self) -> ModelKey:
        return MODEL_KEYS[self.model]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        model_name: str | None = data.get("model")
        if model_name is None:
            raise ValueError("model is required")
        preset = get_model_preset(model_name)
        if data.get("checkpoint_path") is None:
            data["checkpoint_path"] = preset.checkpoint_path
        if data.get("experiment") is None:
            data["experiment"] = preset.experiment
        if not data.get("config_file"):
            data["config_file"] = preset.config_file
        if data.get("context_parallel_size") is None:
            data["context_parallel_size"] = int(os.environ.get("WORLD_SIZE", "1"))
        return data

    @cached_property
    def has_checkpoint_override(self) -> bool:
        preset = get_model_preset(self.model)
        return self.checkpoint_path != preset.checkpoint_path

    @cached_property
    def has_experiment_override(self) -> bool:
        preset = get_model_preset(self.model)
        return self.experiment != preset.experiment


class SetupArguments(CommonSetupArguments):
    """Base model setup arguments."""

    model: BaseModelName = DEFAULT_MODEL_KEY.name
    prg_dense_gate_hdmap_lane_as_generate: bool = False
    prg_dense_gate_hdmap_lane_color_threshold: float = 60.0
    prg_dense_gate_hdmap_lane_dilation: int = 9


class CommonInferenceArguments(pydantic.BaseModel):
    """Common inference arguments."""

    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    name: str
    prompt_path: ResolvedFilePath | None = pydantic.Field(None, init_var=True)
    prompt: str | None = None
    negative_prompt: str | None = None
    seed: int = 0
    guidance: Guidance = 3

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_prompt(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        prompt: str | None = data.get("prompt")
        if prompt is not None:
            return data
        prompt_path: str | None = data.get("prompt_path")
        if prompt_path is not None:
            prompt_file = ResolvedFilePath(prompt_path)
            data["prompt"] = Path(prompt_file).read_text().strip()
        return data

    @classmethod
    def _from_file(cls, path: Path, override_data: dict[str, Any]) -> list["CommonInferenceArguments"]:
        if path.suffix == ".json":
            data_list = [json.loads(path.read_text())]
        elif path.suffix == ".jsonl":
            data_list = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        cwd = os.getcwd()
        os.chdir(path.parent)
        objs = []
        for i, data in enumerate(data_list):
            try:
                objs.append(cls.model_validate(data | override_data))
            except pydantic.ValidationError as e:
                if is_rank0():
                    print(f"Error validating parameters from '{path}' at line {i}\n{e}", file=sys.stderr)
                sys.exit(1)
        os.chdir(cwd)
        return objs

    @classmethod
    def from_files(
        cls,
        paths: list[Path],
        overrides: pydantic.BaseModel | None = None,
    ) -> tuple[list["CommonInferenceArguments"], list[str]]:
        if not paths:
            if is_rank0():
                print("Error: No inference parameter files", file=sys.stderr)
            sys.exit(1)

        override_data = {} if overrides is None else overrides.model_dump(exclude_none=True)

        objs = []
        for path in paths:
            objs.extend(cls._from_file(path, override_data))
        if not objs:
            if is_rank0():
                print("Error: No inference samples", file=sys.stderr)
            sys.exit(1)

        return objs, []


class ControlConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    control_path: ResolvedFilePath | None = None
    control_weight: ControlWeight = 1.0
    mask_path: ResolvedFilePath | None = None
    mask_prompt: str | None = None
    mask_top_rows: pydantic.NonNegativeInt = 0


class DepthConfig(ControlConfig):
    control_path: ResolvedFilePath | None = None


class BlurConfig(ControlConfig):
    control_path: ResolvedFilePath | None = None
    preset_blur_strength: Threshold = "medium"


class EdgeConfig(ControlConfig):
    control_path: ResolvedFilePath | None = None
    preset_edge_threshold: Threshold = "medium"


class SegConfig(ControlConfig):
    control_path: ResolvedFilePath | None = None
    control_prompt: str | None = None


class InferenceArguments(CommonInferenceArguments):
    video_path: ResolvedFilePath
    max_frames: pydantic.PositiveInt | None = None
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    prompt: str
    context_frame_index: pydantic.PositiveInt | None = None
    image_context_path: ResolvedFilePath | None = None
    num_conditional_frames: Literal[0, 1, 2] = 1
    resolution: str = "720"
    sigma_max: str | None = None
    num_video_frames_per_chunk: pydantic.PositiveInt = 93
    num_steps: pydantic.PositiveInt = 1 if SMOKE else 35
    show_control_condition: bool = False
    show_input: bool = False
    keep_input_resolution: bool = True
    edge: EdgeConfig | None = None
    depth: DepthConfig | None = None
    vis: BlurConfig | None = None
    seg: SegConfig | None = None
    lidar: ControlConfig | None = None
    seed: int = 2025
    guided_generation_mask: ResolvedFilePath | ResolvedDirectoryPath | None = None
    guided_generation_step_threshold: int = 25
    guided_generation_foreground_labels: list[int] | None = None
    use_teacher_degradation_query: bool = False
    use_prg_dense_gate: bool = False
    prg_aux_dir: ResolvedDirectoryPath | None = None
    hdmap_path: ResolvedFilePath | None = None
    degradation_query_scale: float = 1.0
    disabled_prg_query_classes: tuple[Literal["preserve", "repair", "generate"], ...] = ()
    prg_query_intervention: Literal["none", "swap_repair_generate"] = "none"

    @cached_property
    def hint_keys(self) -> list[str]:
        return [key for key in [*CONTROL_KEYS, "lidar"] if getattr(self, key, None) is not None]

    def model_post_init(self, __context: Any) -> None:
        if len(self.hint_keys) == 0:
            raise ValueError("No controls provided, please provide at least one control key (edge, blur, depth, seg)")
        if "vis" in self.hint_keys and self.image_context_path and self.vis.control_weight > 0:
            raise ValueError(
                "vis control and image_context_path are both used to transfer style. Using these modes together leads to conflicts. Please only provide one"
            )

    @cached_property
    def control_weight_dict(self) -> dict[str, str]:
        return {key: str(getattr(self, key).control_weight) for key in self.hint_keys}

    @cached_property
    def control_modalities(self) -> dict[str, str | None]:
        control_modalities = {}
        for key in self.hint_keys:
            control_modalities[key] = path_to_str(getattr(self, key).control_path)
            control_modalities[f"{key}_mask"] = path_to_str(getattr(self, key).mask_path)
            control_modalities[f"{key}_mask_prompt"] = getattr(self, key).mask_prompt
            control_modalities[f"{key}_mask_top_rows"] = str(getattr(self, key).mask_top_rows)
        return control_modalities

    @cached_property
    def preset_edge_threshold(self) -> Threshold:
        if "edge" in self.hint_keys:
            return self.edge.preset_edge_threshold
        return "medium"

    @cached_property
    def preset_blur_strength(self) -> Threshold:
        if "vis" in self.hint_keys:
            return self.vis.preset_blur_strength
        return "medium"

    @cached_property
    def seg_control_prompt(self) -> str | None:
        if "seg" not in self.hint_keys or self.seg.control_path is not None:
            return None
        if self.seg.control_prompt is not None:
            return self.seg.control_prompt
        default_prompt = " ".join(self.prompt.split()[:128])
        log.warning(
            'No "control_prompt" provided for on-the-fly segmentation, using the first 128 words of the input prompt'
        )
        return default_prompt

    @cached_property
    def not_keep_input_resolution(self) -> bool:
        return not self.keep_input_resolution

    @classmethod
    def _from_file(cls, path: Path, override_data: dict[str, Any]) -> list[Self]:
        return [obj for obj in super()._from_file(path, override_data)]

    @classmethod
    def from_files(cls, paths: list[Path], overrides: pydantic.BaseModel | None = None) -> tuple[list[Self], list[str]]:
        objs, _ = super().from_files(paths, overrides=overrides)
        names: set[str] = set()
        batch_hint_keys: set[str] = set()
        typed_objs: list[Self] = []
        for obj in objs:
            typed_obj = obj if isinstance(obj, cls) else cls.model_validate(obj.model_dump())
            if typed_obj.name in names:
                if is_rank0():
                    print(f"Error: Inference samplename {typed_obj.name} is not unique")
                raise SystemExit(1)
            names.add(typed_obj.name)
            batch_hint_keys.update(typed_obj.hint_keys)
            typed_objs.append(typed_obj)
        inference_control_keys = [*CONTROL_KEYS, "lidar"]
        sorted_batch_hint_keys = sorted(batch_hint_keys, key=lambda key: inference_control_keys.index(key))
        return typed_objs, sorted_batch_hint_keys


InferenceOverrides = get_overrides_cls(
    InferenceArguments,
    exclude=["name", "edge", "depth", "vis", "seg", "lidar"],
)


class InferenceConfigFile(pydantic.BaseModel):
    """Top-level inference config loaded from a YAML file."""

    model_config = pydantic.ConfigDict(extra="forbid")

    config_path: Path | None = None
    setup: dict[str, Any] = pydantic.Field(default_factory=dict)
    inference: list[dict[str, Any]]
