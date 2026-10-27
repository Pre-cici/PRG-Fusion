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

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from cosmos_oss.init import init_output_dir

from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_transfer2._src.imaginaire.utils import distributed, log, misc
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.experiment.experiment_list import EXPERIMENTS
from cosmos_transfer2._src.transfer2.inference.inference_pipeline import ControlVideo2WorldInference
from cosmos_transfer2._src.transfer2.inference.utils import compile_tokenizer_if_enabled
from cosmos_transfer2.configs.common import CONTROL_KEYS, MODEL_CHECKPOINTS, ModelKey
from cosmos_transfer2.configs.inference_args import (
    InferenceArguments,
    InferenceConfigFile,
    SetupArguments,
    is_rank0,
    path_to_str,
)
from cosmos_transfer2.runtime.resolve_config import resolve_inference_config
from cosmos_transfer2.runtime.save_config import save_final_config
from cosmos_transfer2.runtime.validate_config import validate_image_context_conditioning, validate_inference_config


class Control2WorldInference:
    def __init__(
        self,
        args: SetupArguments,
        batch_hint_keys: list[str],
    ) -> None:
        log.debug(f"{args.__class__.__name__}({args})({batch_hint_keys})")
        self.setup_args = args
        self.batch_hint_keys = batch_hint_keys
        self.is_distilled = args.model_key.distilled

        # Get checkpoint paths - same pattern for distilled and non-distilled
        if "lidar" in self.batch_hint_keys:
            if not args.has_experiment_override:
                raise ValueError("LiDAR inference requires an explicit VIS+LiDAR experiment")
            if args.checkpoint_path is None:
                raise ValueError("LiDAR inference requires setup.checkpoint_path")
            self.checkpoint_list = [args.checkpoint_path]
            self.experiment = args.experiment
        elif len(self.batch_hint_keys) == 1:
            # pyrefly: ignore  # bad-argument-type
            checkpoint = MODEL_CHECKPOINTS[ModelKey(variant=self.batch_hint_keys[0], distilled=self.is_distilled)]
            self.checkpoint_list = [checkpoint.s3.uri]
            self.experiment = checkpoint.experiment
            if args.has_checkpoint_override:
                self.checkpoint_list = [args.checkpoint_path]  # pyrefly: ignore [bad-assignment]
                log.debug(f"Using checkpoint path override: {args.checkpoint_path}")
            if args.has_experiment_override:
                self.experiment = args.experiment
                log.debug(f"Using experiment override: {args.experiment}")

        else:
            # Multi-control: load ALL control modalities even if some have control weight = 0
            self.checkpoint_list = [
                MODEL_CHECKPOINTS[
                    ModelKey(variant=key, distilled=self.is_distilled)  # pyrefly: ignore [bad-argument-type]
                ].s3.uri
                for key in CONTROL_KEYS
            ]
            self.experiment = "multibranch_720p_t24_spaced_layer4_cr1pt1_rectified_flow_inference"

        torch.enable_grad(False)  # Disable gradient calculations for inference

        self.device_rank = 0

        process_group = None
        # pyrefly: ignore  # unsupported-operation
        if args.context_parallel_size > 1:
            from megatron.core import parallel_state

            distributed.init()

            # pyrefly: ignore  # bad-argument-type
            parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
            process_group = parallel_state.get_context_parallel_group()
            self.device_rank = distributed.get_rank(process_group)

        if args.enable_guardrails and self.device_rank == 0:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            # pyrefly: ignore  # bad-assignment
            self.text_guardrail_runner = None
            # pyrefly: ignore  # bad-assignment
            self.video_guardrail_runner = None

        self.benchmark_timer = misc.TrainingTimer()

        # Build experiment override options and resolve registered experiment name
        if self.is_distilled:
            # For distilled models, experiment is already the registered exp name
            registered_exp_name = self.experiment
            exp_override_opts: list[str] = []
            # Compatible with DMD2 distilled model, whose configs are specified at
            # imaginaire4/projects/cosmos3/interactive/configs/method_configs/config_dmd2.py
            exp_override_opts.append("model.config.load_teacher_weights=False")
            # For post-training, the experiment is the registered exp name
        elif args.has_experiment_override:
            registered_exp_name = args.experiment
            exp_override_opts = []
        else:
            # For non-distilled models, look up the experiment in EXPERIMENTS to get
            # the registered_exp_name and command_args
            registered_exp_name = EXPERIMENTS[self.experiment].registered_exp_name
            exp_override_opts = EXPERIMENTS[self.experiment].command_args.copy()

        if args.use_spatial_degradation_query:
            exp_override_opts.append("model.config.net.use_spatial_degradation_query=True")
        exp_override_opts.extend(args.experiment_overrides)

        # Initialize the inference pipeline - same class for both distilled and non-distilled
        if self.device_rank == 0:
            log.info(
                f"Model load | controls={','.join(self.batch_hint_keys)} | checkpoints={len(self.checkpoint_list)} | distilled={self.is_distilled}"
            )
        self.inference_pipeline = ControlVideo2WorldInference(
            # pyrefly: ignore [bad-argument-type]
            registered_exp_name=registered_exp_name,
            checkpoint_paths=self.checkpoint_list,
            prg_checkpoint_path=args.prg_checkpoint_path,
            control_branch_checkpoint_path=args.control_branch_checkpoint_path,
            control_branch_index=args.control_branch_index,
            control_branch_indices=args.control_branch_indices,
            s3_credential_path="",
            exp_override_opts=exp_override_opts,
            process_group=process_group,
            use_cp_wan=args.enable_parallel_tokenizer,
            wan_cp_grid=args.parallel_tokenizer_grid,
            benchmark_timer=self.benchmark_timer if args.benchmark else None,
            config_file=args.config_file,
            offload_diffusion_model=args.offload_diffusion_model,
            offload_text_encoder=args.offload_text_encoder,
            offload_image_context_encoder=args.offload_image_context_encoder,
            offload_tokenizer=args.offload_tokenizer,
        )

        # For distilled models, disable net_fake_score (not needed for inference)
        if self.is_distilled:
            log.info("Setting net_fake_score to None for distilled model inference")
            # pyrefly: ignore [missing-attribute, missing-attribute, missing-attribute, missing-attribute, missing-attribute, missing-attribute]
            self.inference_pipeline.model.net_fake_score = None

        compile_tokenizer_if_enabled(self.inference_pipeline, args.compile_tokenizer.value)

        if self.device_rank == 0:
            log.info(f"Found {len(self.batch_hint_keys)} hint keys across all samples")
            if len(self.batch_hint_keys) > 1:
                log.warning(
                    "Loading the multicontrol model. Multicontrol inference is not strictly equal to single control"
                )

            log.info(f"Inference setup | output={args.output_dir}")

            args.output_dir.mkdir(parents=True, exist_ok=True)
            config_path = args.output_dir / "pipeline_config.yaml"
            # pyrefly: ignore  # bad-argument-type
            LazyConfig.save_yaml(self.inference_pipeline.config, config_path)
            log.info(f"Saved config to {config_path}")

    def generate(self, samples: list[InferenceArguments], output_dir: Path) -> list[str]:
        if SMOKE:
            samples = samples[:1]

        self.is_eval_run = len(samples) > 1
        self.eval_summary_path = output_dir / "summary.json"
        self.eval_summary = self._load_eval_summary(samples, output_dir)
        sample_names = [sample.name for sample in samples]
        log.info(f"Inference batch | samples={len(samples)} | names={sample_names}")

        output_paths: list[str] = []
        for i_sample, sample in enumerate(samples):
            prompt_source = "prompt_path" if sample.prompt_path is not None else "inline_prompt"
            control_summary = ", ".join(
                f"{key}:{sample.control_weight_dict.get(key, '0.0')}" for key in sample.hint_keys
            )
            log.info(
                f"Inference progress [{i_sample + 1}/{len(samples)}] | "
                f"sample={sample.name} | resolution={sample.resolution} | steps={sample.num_steps} | "
                f"controls={control_summary} | prompt={prompt_source}"
            )
            try:
                output_path = self._generate_sample(sample, output_dir, sample_id=i_sample)
                if output_path is not None:
                    output_paths.append(output_path)
                    self._update_summary_entry(sample.name, status="completed", output_path=output_path)
            except Exception as exc:
                self._update_summary_entry(sample.name, status="failed", error=str(exc))
                raise

        if is_rank0() and self.setup_args.benchmark:
            log.info("=" * 50)
            log.info("BENCHMARK RESULTS")
            log.info("=" * 50)
            log.info("Benchmark runs:")
            for key, value in self.benchmark_timer.results.items():
                log.info(f"{key}: {value} seconds")
            log.info("Average times:")
            for key, value in self.benchmark_timer.compute_average_results().items():
                log.info(f"{key}: {value:.2f} seconds")
            log.info("=" * 50)
        return output_paths

    def _load_eval_summary(self, samples: list[InferenceArguments], output_dir: Path) -> dict:
        if not getattr(self, "is_eval_run", False):
            return {}

        output_dir.mkdir(parents=True, exist_ok=True)
        if self.eval_summary_path.exists():
            with self.eval_summary_path.open() as f:
                summary = json.load(f)
        else:
            summary = {
                "mode": "eval",
                "output_dir": str(output_dir),
                "samples": {},
            }

        for sample in samples:
            summary["samples"].setdefault(sample.name, {"status": "pending"})
        self._write_eval_summary(summary)
        return summary

    def _write_eval_summary(self, summary: dict | None = None) -> None:
        if not getattr(self, "is_eval_run", False) or self.device_rank != 0:
            return
        summary = self.eval_summary if summary is None else summary
        with self.eval_summary_path.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    def _update_summary_entry(self, sample_name: str, **updates) -> None:
        if not getattr(self, "is_eval_run", False) or self.device_rank != 0:
            return
        entry = self.eval_summary["samples"].setdefault(sample_name, {})
        entry.update(updates)
        self._write_eval_summary()

    def _sample_output_paths(self, sample: InferenceArguments, output_dir: Path) -> dict[str, Path]:
        if getattr(self, "is_eval_run", False):
            return {
                "input_dir": output_dir / "input",
                "control_dir": output_dir / "control",
                "generation_dir": output_dir / "generation",
                "config_dir": output_dir / "config",
                "sample_json": output_dir / "config" / f"{sample.name}.json",
                "input_video": output_dir / "input" / f"{sample.name}_input.mp4",
                "reference_image": output_dir / "input" / f"{sample.name}_reference_image.png",
                "generation_base": output_dir / "generation" / sample.name,
                "control_base": output_dir / "control" / sample.name,
            }

        return {
            "input_dir": output_dir,
            "control_dir": output_dir,
            "generation_dir": output_dir,
            "config_dir": output_dir,
            "sample_json": output_dir / f"{sample.name}.json",
            "input_video": output_dir / f"{sample.name}_input.mp4",
            "reference_image": output_dir / f"{sample.name}_reference_image.png",
            "generation_base": output_dir / sample.name,
            "control_base": output_dir / sample.name,
        }

    def _existing_generation_path(self, generation_base: Path) -> Path | None:
        for ext in ("mp4", "jpg"):
            candidate = Path(f"{generation_base}.{ext}")
            if candidate.exists():
                return candidate
        return None

    def _generate_sample(self, sample: InferenceArguments, output_dir: Path, sample_id: int = 0) -> str | None:
        log.debug(f"{sample.__class__.__name__}({sample})")
        sample_paths = self._sample_output_paths(sample, output_dir)
        output_path = sample_paths["generation_base"]

        existing_generation = self._existing_generation_path(sample_paths["generation_base"])
        if getattr(self, "is_eval_run", False) and existing_generation is not None:
            log.info(f"Skipping sample {sample.name}; existing generation found at {existing_generation}")
            self._update_summary_entry(sample.name, status="skipped", output_path=str(existing_generation))
            return str(existing_generation)

        assert sample.prompt is not None
        prompt: str = sample.prompt

        # For distilled models, negative_prompt is not needed (CFG is distilled into the model)
        if self.is_distilled:
            negative_prompt = None
        else:
            assert sample.negative_prompt is not None
            negative_prompt = sample.negative_prompt

        guided_generation_mask = (
            str(sample.guided_generation_mask) if sample.guided_generation_mask is not None else None
        )
        guided_generation_step_threshold = sample.guided_generation_step_threshold
        guided_generation_foreground_labels = sample.guided_generation_foreground_labels

        if self.device_rank == 0:
            sample_paths["input_dir"].mkdir(parents=True, exist_ok=True)
            sample_paths["control_dir"].mkdir(parents=True, exist_ok=True)
            sample_paths["generation_dir"].mkdir(parents=True, exist_ok=True)
            sample_paths["config_dir"].mkdir(parents=True, exist_ok=True)
            sample_paths["sample_json"].write_text(sample.model_dump_json())
            shutil.copy2(path_to_str(sample.video_path), sample_paths["input_video"])
            log.info(f"Saved input video to {sample_paths['input_video']}")
            log.info(f"Saved arguments to {sample_paths['sample_json']}")

            with self.benchmark_timer("text_guardrail"):
                # run text guardrail on the prompt
                if self.text_guardrail_runner is not None:
                    log.info("Running guardrail check on prompt...")

                    if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                        message = f"Guardrail blocked generation. Prompt: {prompt}"
                        log.critical(message)
                        if self.setup_args.keep_going:
                            return None
                        else:
                            raise Exception(message)
                    else:
                        log.success("Passed guardrail on prompt")

                    if negative_prompt is not None:
                        if not guardrail_presets.run_text_guardrail(
                            negative_prompt,
                            self.text_guardrail_runner,
                        ):
                            message = f"Guardrail blocked generation. Negative prompt: {negative_prompt}"
                            log.critical(message)
                            if self.setup_args.keep_going:
                                return None
                            else:
                                raise Exception(message)
                        else:
                            log.success("Passed guardrail on negative prompt")
                elif self.text_guardrail_runner is None:
                    log.warning("Guardrail checks on prompt are disabled")

        input_control_video_paths = sample.control_modalities

        sigma_max = None if sample.sigma_max is None else float(sample.sigma_max)

        # control_weight is a string because of multi-control
        control_weight = ""
        for key in self.batch_hint_keys:
            # pyrefly: ignore  # missing-attribute
            control_weight += sample.control_weight_dict.get(key, "0.0") + ","
        control_weight = control_weight[:-1]

        if self.setup_args.benchmark:
            torch.cuda.synchronize()

        with self.benchmark_timer("generate_img2world"):
            # For distilled models, guidance is not needed (CFG is distilled into the model)
            guidance = None if self.is_distilled else sample.guidance
            # Run model inference
            output_video, control_video_dict, mask_video_dict, fps, _ = self.inference_pipeline.generate_img2world(
                # pyrefly: ignore  # bad-argument-type
                video_path=path_to_str(sample.video_path),
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_context_path=path_to_str(sample.image_context_path),
                reference_image_output_path=str(sample_paths["reference_image"]),
                context_frame_idx=sample.context_frame_index,
                max_frames=sample.max_frames,
                # pyrefly: ignore [bad-argument-type]
                guidance=guidance,
                seed=sample.seed,
                resolution=sample.resolution,
                control_weight=control_weight,
                sigma_max=sigma_max,
                hint_key=sample.hint_keys,
                # pyrefly: ignore  # bad-argument-type
                input_control_video_paths=input_control_video_paths,
                show_control_condition=sample.show_control_condition,
                seg_control_prompt=sample.seg_control_prompt,
                show_input=sample.show_input,
                keep_input_resolution=not sample.not_keep_input_resolution,
                preset_blur_strength=sample.preset_blur_strength,
                preset_edge_threshold=sample.preset_edge_threshold,
                num_conditional_frames=sample.num_conditional_frames,
                num_video_frames_per_chunk=sample.num_video_frames_per_chunk,
                num_steps=sample.num_steps,
                guided_generation_mask=guided_generation_mask,
                guided_generation_step_threshold=guided_generation_step_threshold,
                guided_generation_foreground_labels=guided_generation_foreground_labels,
                use_teacher_degradation_query=sample.use_teacher_degradation_query,
                use_prg_dense_gate=sample.use_prg_dense_gate,
                prg_dense_gate_hdmap_lane_as_generate=self.setup_args.prg_dense_gate_hdmap_lane_as_generate,
                prg_dense_gate_hdmap_lane_color_threshold=self.setup_args.prg_dense_gate_hdmap_lane_color_threshold,
                prg_dense_gate_hdmap_lane_dilation=self.setup_args.prg_dense_gate_hdmap_lane_dilation,
                prg_aux_dir=path_to_str(sample.prg_aux_dir),
                hdmap_path=path_to_str(sample.hdmap_path),
                degradation_query_scale=sample.degradation_query_scale,
                disabled_prg_query_classes=sample.disabled_prg_query_classes,
                prg_query_intervention=sample.prg_query_intervention,
            )
            if self.setup_args.benchmark:
                torch.cuda.synchronize()

        log.info(f"Generate done | frames={output_video.shape[2]} | fps={fps}")
        if self.device_rank == 0 and sample_paths["reference_image"].exists():
            log.info(f"Saved processed reference image to {sample_paths['reference_image']}")

        if output_video.shape[2] == 1:
            ext = "jpg"
        else:
            ext = "mp4"

        if self.is_distilled and output_video.shape[2] > 93:
            log.warning(
                "Generated output has "
                f"{output_video.shape[2]} frames (> 93). "
                "The distilled Transfer 2.5 model is not trained to support auto-regressive generation"
            )

        torch.cuda.empty_cache()
        # Save video/image
        if self.device_rank == 0:
            output_video = (1.0 + output_video[0]) / 2
            # Save control video
            for key in control_video_dict:
                control_video_dict[key] = (1.0 + control_video_dict[key][0]) / 2
                control_output_base = sample_paths["control_base"].with_name(
                    f"{sample_paths['control_base'].name}_control_{key}"
                )
                save_img_or_video(control_video_dict[key], str(control_output_base), fps=fps)
                log.info(f"{key} control video saved to {control_output_base}.{ext}")

            with self.benchmark_timer("video_guardrail"):
                for key in mask_video_dict:
                    mask_output_base = sample_paths["control_base"].with_name(
                        f"{sample_paths['control_base'].name}_mask_{key}"
                    )
                    save_img_or_video(mask_video_dict[key], str(mask_output_base), fps=fps)
                    log.info(f"Mask for {key} saved to {mask_output_base}.{ext}")
                # run video guardrail on the video
                if self.video_guardrail_runner is not None:
                    log.info("Running guardrail check on video...")
                    frames = (output_video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                    frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
                    processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                    if processed_frames is None:
                        if self.setup_args.keep_going:
                            return None
                        else:
                            raise Exception("Guardrail blocked video2world generation.")
                    else:
                        log.success("Passed guardrail on generated video")

                    # Convert processed frames back to tensor format
                    processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                    output_video = processed_video.to(output_video.device, dtype=output_video.dtype)
                else:
                    log.warning("Guardrail checks on video are disabled")

            # Remove batch dimension and normalize to [0, 1] range
            # Save generative video
            save_img_or_video(output_video, str(output_path), fps=fps)
            # save prompt
            # prompt_save_path = f"{output_path}.txt"
            # with open(prompt_save_path, "w") as f:
            #     f.write(sample.prompt)
            log.success(f"Generated video saved to {output_path}.{ext}")

        if sample_id == 0 and self.setup_args.benchmark:
            # discard first warmup sample from timing
            self.benchmark_timer.reset()

        return f"{output_path}.{ext}"


def run_inference(config: InferenceConfigFile) -> None:
    setup_args = SetupArguments.model_validate(config.setup)
    resolved = resolve_inference_config(
        setup_args=setup_args,
        sample_dicts=config.inference,
        config_path=config.config_path,
    )
    validate_inference_config(resolved)
    if len(resolved.batch_hint_keys) == 1 and setup_args.model != resolved.batch_hint_keys[0]:
        log.info(
            "Detected single-control input from inference config: "
            f"{resolved.batch_hint_keys[0]!r}. For base singleview inference, the control key in the input file is the "
            f"source of truth, so setup.model={setup_args.model!r} is not required."
        )
    if setup_args.benchmark and len(resolved.sample_args) == 1:
        resolved.sample_args[:] = resolved.sample_args * 4
        log.info("Repeating inference sample 4 times for benchmarking.")
    init_output_dir(resolved.output_dir, profile=resolved.setup_args.profile)
    save_final_config(resolved.final_config, resolved.output_dir)
    inference = Control2WorldInference(resolved.setup_args, batch_hint_keys=resolved.batch_hint_keys)
    validate_image_context_conditioning(inference.inference_pipeline, resolved.sample_args)
    inference.generate(resolved.sample_args, output_dir=resolved.output_dir)
