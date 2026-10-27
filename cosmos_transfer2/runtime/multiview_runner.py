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

"""Runtime execution for auto multiview inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import decord
import numpy as np
import torch
from cosmos_oss.init import init_output_dir

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.conditioner import (
    ConditionLocation,
    ConditionLocationList,
)
from cosmos_transfer2._src.predict2_multiview.datasets.local import LocalMultiViewDataset
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import AugmentationConfig, collate_fn
from cosmos_transfer2._src.transfer2.inference.utils import color_message
from cosmos_transfer2._src.transfer2_multiview.inference.inference import ControlVideo2WorldInference
from cosmos_transfer2.configs.multiview_args import (
    MULTIVIEW_CAMERA_KEYS,
    MultiviewConfigFile,
    MultiviewInferenceArguments,
    MultiviewSetupArguments,
)
from cosmos_transfer2.runtime.resolve_config import ResolvedMultiviewConfig, resolve_multiview_config
from cosmos_transfer2.runtime.save_config import save_final_config
from cosmos_transfer2.runtime.validate_config import validate_multiview_config

RESOLUTIONS: Mapping[str, tuple[int, int]] = {
    "720p": (720, 1280),
}

DEFAULT_CAMERA_KEYS = MULTIVIEW_CAMERA_KEYS
DEFAULT_CAMERA_VIEW_MAPPING = {camera_key: idx for idx, camera_key in enumerate(DEFAULT_CAMERA_KEYS)}
DEFAULT_CAMERA_PREFIX_MAPPING = {
    "front_wide": "The video is captured from a camera mounted on a car. The camera is facing forward.",
    "cross_right": "The video is captured from a camera mounted on a car. The camera is facing to the right.",
    "rear_right": "The video is captured from a camera mounted on a car. The camera is facing the rear right side.",
    "rear": "The video is captured from a camera mounted on a car. The camera is facing backwards.",
    "rear_left": "The video is captured from a camera mounted on a car. The camera is facing the rear left side.",
    "cross_left": "The video is captured from a camera mounted on a car. The camera is facing to the left.",
    "front_tele": "The video is captured from a telephoto camera mounted on a car. The camera is facing forward.",
}

GRID_ROWS = 3
GRID_COLS = 3
CAMERA_GRID_POSITIONS: Mapping[str, tuple[int, int]] = {
    "front_tele": (0, 1),
    "cross_left": (1, 0),
    "front_wide": (1, 1),
    "cross_right": (1, 2),
    "rear_left": (2, 0),
    "rear": (2, 1),
    "rear_right": (2, 2),
}


def _split_generated_views(
    video: torch.Tensor, camera_keys: tuple[str, ...] | list[str]
) -> list[tuple[str, torch.Tensor]]:
    total_frames = video.shape[1]
    n_views = len(camera_keys)
    if total_frames % n_views != 0:
        raise ValueError(f"Video frames ({total_frames}) not divisible by number of views ({n_views}).")

    frames_per_view = total_frames // n_views
    view_tensors: list[tuple[str, torch.Tensor]] = []
    for view_index, view_name in enumerate(camera_keys):
        start = view_index * frames_per_view
        end = start + frames_per_view
        view_tensors.append((view_name, video[:, start:end]))
    return view_tensors


def _make_view_grid(view_tensors: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
    c, t, h, w = view_tensors[0][1].shape
    grid_tensor = torch.zeros(
        (c, t, GRID_ROWS * h, GRID_COLS * w), dtype=view_tensors[0][1].dtype, device=view_tensors[0][1].device
    )
    for view_name, view_tensor in view_tensors:
        row, col = CAMERA_GRID_POSITIONS.get(view_name, (0, 0))
        grid_tensor[:, :, row * h : (row + 1) * h, col * w : (col + 1) * w] = view_tensor
    return grid_tensor


def _load_video_tensor(path: Path, *, fps_downsample_factor: int, max_frames: int) -> torch.Tensor:
    vr = decord.VideoReader(path.as_posix())
    frame_indices = list(range(0, len(vr), fps_downsample_factor))[:max_frames]
    if not frame_indices:
        raise ValueError(f"No frames available in video: {path}")
    frames = vr.get_batch(frame_indices).asnumpy()
    return torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 255.0


def _load_reference_view_tensors(
    video_file_dict: dict[str, Path],
    camera_keys: tuple[str, ...],
    *,
    fps_downsample_factor: int,
    max_frames: int,
) -> list[tuple[str, torch.Tensor]]:
    view_tensors: list[tuple[str, torch.Tensor]] = []
    min_loaded_frames: int | None = None
    for camera_key in camera_keys:
        view_tensor = _load_video_tensor(
            video_file_dict[camera_key], fps_downsample_factor=fps_downsample_factor, max_frames=max_frames
        )
        min_loaded_frames = (
            view_tensor.shape[1] if min_loaded_frames is None else min(min_loaded_frames, view_tensor.shape[1])
        )
        view_tensors.append((camera_key, view_tensor))

    assert min_loaded_frames is not None
    return [(camera_key, view_tensor[:, :min_loaded_frames]) for camera_key, view_tensor in view_tensors]


def setup_config(
    resolution_hw: tuple[int, int],
    num_video_frames_per_view: int,
    fps_downsample_factor: int,
    camera_keys: tuple[str, ...] | None = None,
    single_caption_camera_name: str | None = "front_wide",
) -> AugmentationConfig:
    camera_keys = camera_keys or DEFAULT_CAMERA_KEYS
    if not camera_keys:
        raise ValueError("At least one camera key must be provided for multiview inference.")
    invalid_keys = set(camera_keys) - set(DEFAULT_CAMERA_KEYS)
    if invalid_keys:
        raise ValueError(f"Unknown camera keys provided: {', '.join(sorted(invalid_keys))}")
    if single_caption_camera_name not in camera_keys:
        single_caption_camera_name = camera_keys[0]

    kwargs = dict(
        resolution_hw=resolution_hw,
        fps_downsample_factor=fps_downsample_factor,
        num_video_frames=num_video_frames_per_view,
        camera_keys=camera_keys,
        camera_view_mapping={key: DEFAULT_CAMERA_VIEW_MAPPING[key] for key in camera_keys},
        camera_caption_key_mapping={k: f"caption_{k}" for k in camera_keys},
        camera_video_key_mapping={k: f"video_{k}" for k in camera_keys},
        camera_control_key_mapping={k: f"control_{k}" for k in camera_keys},
        add_view_prefix_to_caption=False,
        camera_prefix_mapping={k: DEFAULT_CAMERA_PREFIX_MAPPING[k] for k in camera_keys},
        single_caption_camera_name=single_caption_camera_name,
    )
    return AugmentationConfig(**kwargs)


class AutoMultiviewInference:
    def __init__(self, args: MultiviewSetupArguments):
        log.debug(f"{args.__class__.__name__}({args})")
        torch.enable_grad(False)
        self.setup_args = args
        self.pipe = ControlVideo2WorldInference(
            experiment_name=args.experiment,
            ckpt_path=args.checkpoint_path,
            context_parallel_size=args.context_parallel_size,
        )

        self.rank0 = True
        if args.context_parallel_size > 1:
            self.rank0 = torch.distributed.get_rank() == 0

        if self.rank0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            LazyConfig.save_yaml(self.pipe.config, args.output_dir / "config.yaml")

        if self.rank0 and args.enable_guardrails:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            self.text_guardrail_runner = None
            self.video_guardrail_runner = None

    def generate(self, samples: list[MultiviewInferenceArguments], output_dir: Path) -> list[str]:
        if SMOKE:
            samples = samples[:1]
        output_paths: list[str] = []
        for i_sample, sample in enumerate(samples):
            log.info(f"[{i_sample + 1}/{len(samples)}] Processing sample {sample.name}")
            output_path = self._generate_sample(sample, output_dir)
            if output_path is not None:
                output_paths.append(output_path)
        return output_paths

    def _generate_sample(self, sample: MultiviewInferenceArguments, output_dir: Path) -> str | None:
        output_path = output_dir / sample.name
        if self.rank0:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.with_suffix(".json").write_text(sample.model_dump_json())

        input_video_file_dict: dict[str, Path] = {}
        control_video_file_dict: dict[str, Path] = {}
        fps_values: set[float] = set()
        min_control_frames = float("inf")
        for key, value in sample.input_and_control_paths.items():
            if "_input" in key and value is not None:
                input_video_file_dict[key.removesuffix("_input")] = value
                fps_values.add(decord.VideoReader(value.as_posix()).get_avg_fps())
            elif "_control" in key:
                assert value is not None
                control_video_file_dict[key.removesuffix("_control")] = value
                vr = decord.VideoReader(value.as_posix())
                fps_values.add(vr.get_avg_fps())
                min_control_frames = min(min_control_frames, len(vr))

        if len(fps_values) != 1:
            raise ValueError(f"Control and input videos must share the same FPS. Got: {fps_values}")
        source_fps = fps_values.pop()
        desired_fps = sample.fps
        if source_fps % desired_fps != 0:
            raise ValueError(f"Video fps {source_fps} is not evenly divisible by desired fps {desired_fps}")
        fps_downsample_factor = int(source_fps / desired_fps)

        chunk_size = self.pipe.model.tokenizer.get_pixel_num_frames(self.pipe.config.model.config.state_t)
        available_control_frames_after_downsample = int(min_control_frames) // fps_downsample_factor
        if available_control_frames_after_downsample < chunk_size:
            raise ValueError(
                f"Not enough frames in control videos. Need at least {chunk_size} after downsampling, "
                f"but only have {available_control_frames_after_downsample}."
            )

        effective_num_chunks = sample.num_chunks
        if sample.enable_autoregressive:
            max_chunks = 1 + (available_control_frames_after_downsample - chunk_size) // (
                chunk_size - sample.chunk_overlap
            )
            max_chunks = max(1, max_chunks)
            if sample.num_chunks > max_chunks:
                effective_num_chunks = max_chunks

        num_video_frames_per_view = chunk_size
        if sample.enable_autoregressive:
            num_video_frames_per_view += (num_video_frames_per_view - sample.chunk_overlap) * (effective_num_chunks - 1)

        camera_keys = sample.active_camera_keys
        primary_caption_view = "front_wide" if "front_wide" in camera_keys else camera_keys[0]
        augmentation_config = setup_config(
            resolution_hw=RESOLUTIONS[self.pipe.config.model.config.resolution],
            num_video_frames_per_view=num_video_frames_per_view,
            fps_downsample_factor=fps_downsample_factor,
            camera_keys=camera_keys,
            single_caption_camera_name=primary_caption_view,
        )
        if SMOKE:
            augmentation_config.camera_keys = augmentation_config.camera_keys[:1]
        if sample.enable_autoregressive:
            self.pipe.config.model.config.condition_locations = ConditionLocationList(
                [ConditionLocation.FIRST_RANDOM_N]
            )

        if self.rank0 and self.text_guardrail_runner is not None:
            assert sample.prompt is not None
            if not guardrail_presets.run_text_guardrail(sample.prompt, self.text_guardrail_runner):
                if self.setup_args.keep_going:
                    return None
                raise RuntimeError(f"Guardrail blocked generation. Prompt: {sample.prompt}")

        if sample.num_conditional_frames == 0:
            input_video_file_dict = control_video_file_dict

        assert sample.prompt is not None
        dataset = LocalMultiViewDataset(
            video_file_dicts=[input_video_file_dict],
            prompts=[sample.prompt],
            control_file_dicts=[control_video_file_dict],
            augmentation_config=augmentation_config,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        if len(dataloader) == 0:
            raise ValueError("No input data found")

        for _, batch in enumerate(dataloader):
            batch["control_weight"] = sample.control_weight
            if sample.enable_autoregressive:
                per_view_frames = [
                    getattr(sample, k).num_conditional_frames_per_view for k in augmentation_config.camera_keys
                ]
                num_conditional_frames = (
                    sample.num_conditional_frames if all(frames == 0 for frames in per_view_frames) else per_view_frames
                )
                video, _ = self.pipe.generate_autoregressive_from_batch(
                    batch,
                    n_views=len(augmentation_config.camera_keys),
                    chunk_overlap=sample.chunk_overlap,
                    chunk_size=chunk_size,
                    guidance=sample.guidance,
                    seed=sample.seed,
                    num_conditional_frames=num_conditional_frames,
                    num_steps=sample.num_steps,
                    use_negative_prompt=True,
                )
            else:
                batch["num_conditional_frames"] = self.pipe.model.tokenizer.get_latent_num_frames(
                    sample.num_conditional_frames
                )
                video = self.pipe.generate_from_batch(
                    batch,
                    guidance=sample.guidance,
                    seed=sample.seed,
                    num_steps=sample.num_steps,
                    use_negative_prompt=True,
                )

            if self.rank0:
                if not sample.enable_autoregressive:
                    video = video[0]
                if self.video_guardrail_runner is not None:
                    frames = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
                    processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                    if processed_frames is None:
                        if self.setup_args.keep_going:
                            return None
                        raise RuntimeError("Guardrail blocked generation. Video")
                    video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0

                view_tensors = _split_generated_views(video, augmentation_config.camera_keys)
                gt_view_tensors = _load_reference_view_tensors(
                    input_video_file_dict,
                    augmentation_config.camera_keys,
                    fps_downsample_factor=fps_downsample_factor,
                    max_frames=view_tensors[0][1].shape[1],
                )
                control_view_tensors = _load_reference_view_tensors(
                    control_video_file_dict,
                    augmentation_config.camera_keys,
                    fps_downsample_factor=fps_downsample_factor,
                    max_frames=view_tensors[0][1].shape[1],
                )
                gt_grid = _make_view_grid(gt_view_tensors)
                control_grid = _make_view_grid(control_view_tensors)

                if sample.save_combined_views:
                    combined_grid = _make_view_grid(view_tensors)
                    save_img_or_video(combined_grid, str(output_path), fps=sample.fps, quality=8)
                    save_img_or_video(gt_grid, f"{output_path}_gt", fps=sample.fps, quality=8)
                    save_img_or_video(control_grid, f"{output_path}_control", fps=sample.fps, quality=8)
                    return f"{output_path}.mp4"

                output_messages = []
                for view_name, view_tensor in view_tensors:
                    view_output_path = f"{output_path}_{view_name}"
                    save_img_or_video(view_tensor, view_output_path, fps=sample.fps, quality=8)
                    output_messages.append(f"{view_output_path}.mp4")

                grid_tensor = _make_view_grid(view_tensors)
                grid_output_path = f"{output_path}_grid"
                save_img_or_video(grid_tensor, grid_output_path, fps=sample.fps, quality=8)
                output_messages.append(f"{grid_output_path}.mp4")
                save_img_or_video(gt_grid, f"{output_path}_gt", fps=sample.fps, quality=8)
                output_messages.append(f"{output_path}_gt.mp4")
                save_img_or_video(control_grid, f"{output_path}_control", fps=sample.fps, quality=8)
                output_messages.append(f"{output_path}_control.mp4")
                log.success(color_message("Generated videos saved to:\n" + "\n".join(output_messages) + "\n", "green"))

        return f"{output_path}.mp4"


def run_multiview(config: MultiviewConfigFile) -> None:
    resolved: ResolvedMultiviewConfig = resolve_multiview_config(config)
    validate_multiview_config(resolved)
    init_output_dir(resolved.output_dir, profile=resolved.setup_args.profile)
    save_final_config(
        {
            "setup": resolved.setup_args.model_dump(mode="json"),
            "samples": [sample.model_dump(mode="json") for sample in resolved.sample_args],
        },
        resolved.output_dir,
    )
    pipeline = AutoMultiviewInference(resolved.setup_args)
    pipeline.generate(resolved.sample_args, output_dir=resolved.output_dir)
