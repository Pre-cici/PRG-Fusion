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

import contextlib
import os
import random
import time
from typing import Optional, Union

import cv2
import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.flags import INTERNAL
from cosmos_transfer2._src.imaginaire.utils import distributed, log, misc
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.interactive.utils.model_loader import (
    load_model_from_checkpoint as load_model_interactive,
)
from cosmos_transfer2._src.predict2.checkpointer.dcp import (
    load_model_state_dict_subset_from_dcp,
    map_multi_branch_key_to_single_branch,
)
from cosmos_transfer2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_transfer2._src.predict2.utils.model_loader import (
    load_model_from_checkpoint as load_model_predict2,
)
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import get_augmentor_for_eval
from cosmos_transfer2._src.transfer2.datasets.prg import load_prg_label_frames, prg_labels_to_rgb
from cosmos_transfer2._src.transfer2.inference.utils import (
    get_t5_from_prompt,
    normalized_float_to_uint8,
    read_and_process_control_input,
    read_and_process_image_context,
    read_and_process_video,
    reshape_output_video_to_input_resolution,
    resize_video,
    uint8_to_normalized_float,
)
from cosmos_transfer2._src.transfer2.models.prg_losses import override_prg_labels_with_hdmap_lane_generate
from cosmos_transfer2._src.transfer2.networks.siglip2_image_context import SigLip2EmbImgContext


def _maybe_get_timer(
    benchmark_timer: Optional[misc.TrainingTimer], func_name: str
) -> contextlib.nullcontext | misc.TrainingTimer:
    return benchmark_timer(func_name) if benchmark_timer is not None else contextlib.nullcontext()


class ControlVideo2WorldInference:
    """
    Handles the Control2Video inference process, including model loading, data preparation,
    and video transfer from an input video and text prompt.
    """

    def __init__(
        self,
        registered_exp_name: str,
        checkpoint_paths: Union[str, list[str]],
        s3_credential_path: str,
        prg_checkpoint_path: Optional[str] = None,
        control_branch_checkpoint_path: Optional[str] = None,
        control_branch_index: int = 1,
        control_branch_indices: tuple[int, ...] = (),
        exp_override_opts: Optional[list[str]] = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        cache_dir: Optional[str] = None,
        skip_load_model: bool = False,
        base_load_from: Optional[str] = None,
        use_cp_wan: bool = False,
        wan_cp_grid: tuple[int, int] = (-1, -1),
        benchmark_timer: Optional[misc.TrainingTimer] = None,
        cache_text_encoder: bool = True,
        use_cuda_graphs: bool = False,
        cfg_parallel: bool = False,
        hierarchical_cp: bool = False,
        config_file: str = "cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py",
        offload_diffusion_model: bool = True,
        offload_text_encoder: bool = True,
        offload_image_context_encoder: bool = True,
        offload_tokenizer: bool = True,
    ):
        """
        Initializes the ControlVideo2WorldInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path.

        Args:
            registered_exp_name (str): Name of the experiment configuration.
            checkpoint_paths (Union[str, list[str]]): Single checkpoint path or List of checkpoint paths for multi-branch models.
            s3_credential_path (str): Path to S3 credentials file for ckpt & negative embedding (if loading from S3).
            exp_override_opts (list[str]): List of experiment override options.
            process_group (torch.distributed.ProcessGroup): Process group for distributed training.
            cache_dir (str): Cache directory for storing pre-computed embeddings.
            skip_load_model (bool): Whether to skip loading model from checkpoint for multi-control models.
            use_cp_wan (bool, optional): Whether to use parallel tokenizer. Defaults to False.
            wan_cp_grid (tuple[int, int], optional): The grid for parallel tokenizer. Used only when use_cp_wan is True. Defaults to (1, cp_size).
            use_cuda_graphs (bool, optional): Whether to use CUDA Graphs for inference. Defaults to False.
            cfg_parallel (bool, optional): Whether to parallelize Classifier Free Guidance.
            hierarchical_cp (bool, optional): Whether to use hierarchical CP algorithm.
        """
        self.registered_exp_name = registered_exp_name
        self.checkpoint_path = checkpoint_paths if isinstance(checkpoint_paths, str) else checkpoint_paths[0]
        self.s3_credential_path = s3_credential_path
        self.cache_dir = cache_dir
        self.cache_text_encoder = cache_text_encoder
        self.offload_diffusion_model = offload_diffusion_model
        self.offload_text_encoder = offload_text_encoder
        self.offload_image_context_encoder = offload_image_context_encoder
        self.offload_tokenizer = offload_tokenizer
        if exp_override_opts is None:
            exp_override_opts = []
        # no need to load base model separately at inference
        exp_override_opts.append("++model.config.base_load_from=null")
        if use_cuda_graphs:
            exp_override_opts.append("model.config.net.use_cuda_graphs=True")
        # data_train is training-only; distilled interactive inference does not define it,
        # so avoid deleting it for interactive configs to prevent Hydra errors.
        if not INTERNAL and (not config_file or "interactive" not in config_file):
            exp_override_opts.append("~data_train")
        if hierarchical_cp:
            exp_override_opts.append("model.config.net.atten_backend='transformer_engine'")
        model_device = None if offload_diffusion_model else "cuda"
        if self.offload_diffusion_model:
            os.environ["COSMOS_PREDICT2_OFFLOAD_DIT"] = "1"
        local_cache_dir = cache_dir if not (isinstance(checkpoint_paths, list) and len(checkpoint_paths) > 1) else None
        # Load the model and config. Each trained model's config is composed by
        # loading a pre-registered experiment config, and then (optionally) overriding with some command-line
        # arguments. That is done in experiment_list.py. Here we simply replicate that process.
        # Use interactive loader for distilled models (interactive config), predict2 loader otherwise
        if config_file and "interactive" in config_file:
            model, config = load_model_interactive(
                experiment_name=self.registered_exp_name,
                s3_checkpoint_dir=self.checkpoint_path,
                config_file=config_file,
                load_ema_to_reg=True,
                local_cache_dir=local_cache_dir,  # for multi-control models, need to load other branches before caching
                experiment_opts=exp_override_opts,
            )
        else:
            model, config = load_model_predict2(
                experiment_name=self.registered_exp_name,
                s3_checkpoint_dir=self.checkpoint_path,
                config_file=config_file,
                load_ema_to_reg=True,
                local_cache_dir=local_cache_dir,  # for multi-control models, need to load other branches before caching
                experiment_opts=exp_override_opts,
                cache_text_encoder=self.cache_text_encoder,
                to_device=model_device,
            )
        if (
            isinstance(checkpoint_paths, list) and len(checkpoint_paths) > 1 and not skip_load_model
        ):  # load other branches for multi-control models
            load_from_local = False
            if cache_dir is not None:
                # build a unique path for s3checkpoint dir
                local_s3_ckpt_fp = os.path.join(
                    cache_dir,
                    self.checkpoint_path.split("s3://")[1],
                    "torch_model",
                    f"_rank_{distributed.get_rank()}.pt",
                )
                if os.path.exists(local_s3_ckpt_fp):
                    load_from_local = True

            if load_from_local:
                log.info(f"Loading model cached locally from {local_s3_ckpt_fp}")
                model.load_state_dict(easy_io.load(local_s3_ckpt_fp))
            else:
                model.load_multi_branch_checkpoints(checkpoint_paths=checkpoint_paths)
                if cache_dir is not None:
                    log.info(f"Caching model state dict to {local_s3_ckpt_fp}")
                    easy_io.dump(model.state_dict(), local_s3_ckpt_fp)

        configured_branch_paths = list(getattr(config.model.config, "control_branch_checkpoint_paths", ()))
        if configured_branch_paths and not (isinstance(checkpoint_paths, list) and len(checkpoint_paths) > 1):
            model.load_multi_branch_checkpoints(configured_branch_paths)

        configured_overlay_path = getattr(config.model.config, "control_branch_overlay_checkpoint_path", None)
        if configured_overlay_path is not None:
            configured_overlay_indices = tuple(
                getattr(config.model.config, "control_branch_overlay_indices", ())
                or (getattr(config.model.config, "control_branch_overlay_index", 1),)
            )
            overlay_from_single = getattr(config.model.config, "control_branch_overlay_from_single_branch", False)
            if overlay_from_single and len(configured_overlay_indices) != 1:
                raise ValueError("A single-branch checkpoint overlay requires exactly one target branch")
            loaded_keys = load_model_state_dict_subset_from_dcp(
                model,
                configured_overlay_path,
                lambda key: any(
                    key.startswith(f"net.control_embedder.{index}.") or key.startswith(f"net.control_blocks_{index}.")
                    for index in configured_overlay_indices
                ),
                load_ema_to_reg=True,
                model_to_checkpoint_key=(
                    (lambda key: map_multi_branch_key_to_single_branch(key, configured_overlay_indices[0]))
                    if overlay_from_single
                    else None
                ),
            )
            log.info(
                f"Configured control branch overlay | branches={configured_overlay_indices} | "
                f"path={configured_overlay_path} | tensors={len(loaded_keys)}"
            )

        if base_load_from is not None:
            log.info(f"Loading base model from {base_load_from}")
            model.config.base_load_from = {
                "load_path": base_load_from,
                "credentials": s3_credential_path,
            }
            model.load_base_model(load_ema_to_reg=True)

        if control_branch_checkpoint_path is not None:
            overlay_indices = control_branch_indices or (control_branch_index,)
            if len(set(overlay_indices)) != len(overlay_indices):
                raise ValueError(f"Duplicate control branch overlay indices: {overlay_indices}")
            invalid_indices = [
                index for index in overlay_indices if index < 0 or index >= model.net.num_control_branches
            ]
            if invalid_indices:
                raise ValueError(
                    f"Invalid control branch overlay indices {invalid_indices}; "
                    f"model has {model.net.num_control_branches} branches"
                )
            if model.net.num_control_branches == 1 and overlay_indices != (0,):
                raise ValueError(f"A single-branch model only supports overlay index 0, got {overlay_indices}")
            include_adaptive_gate = bool(getattr(model.config, "use_prg_adaptive_dense_gate", False))
            loaded_keys = load_model_state_dict_subset_from_dcp(
                model,
                control_branch_checkpoint_path,
                lambda key: (
                    key.startswith("net.control_embedder.") or key.startswith("net.control_blocks.")
                    if model.net.num_control_branches == 1
                    else any(
                        key.startswith(f"net.control_embedder.{index}.")
                        or key.startswith(f"net.control_blocks_{index}.")
                        for index in overlay_indices
                    )
                )
                or (include_adaptive_gate and key.startswith("net.prg_adaptive_vis_gate.")),
                load_ema_to_reg=True,
            )
            log.info(
                f"Control branch checkpoint overlay | branches={overlay_indices} | "
                f"adaptive_gate={include_adaptive_gate} | path={control_branch_checkpoint_path} | "
                f"tensors={len(loaded_keys)} | frozen base/other branches preserved"
            )

        if prg_checkpoint_path is not None:
            loaded_keys = load_model_state_dict_subset_from_dcp(
                model,
                prg_checkpoint_path,
                lambda key: key.startswith("net.prg_teacher_query_branch.") or ".routed_query_adapter." in key,
                load_ema_to_reg=True,
            )
            log.info(
                f"PRG checkpoint overlay | path={prg_checkpoint_path} | tensors={len(loaded_keys)} | "
                "frozen base/control tensors preserved"
            )

        # Get text_encoder_class from model or config depending on model type
        if hasattr(model, "text_encoder_class"):
            self.text_encoder_class = model.text_encoder_class
        else:
            # For distilled models (DMD2Model), get from config
            self.text_encoder_class = config.model.config.text_encoder_class

        if process_group is not None:
            cp_comm_type = "a2a+p2p" if hierarchical_cp else "p2p"
            log.info(f"Enabling CP in base model with {cp_comm_type}\n")
            model.net.enable_context_parallel(process_group, cfg_parallel=cfg_parallel, cp_comm_type=cp_comm_type)

            cp_size = process_group.size()

            if use_cp_wan:
                wan_cp_grid = wan_cp_grid if wan_cp_grid != (-1, -1) else (1, cp_size)
                assert wan_cp_grid[0] * wan_cp_grid[1] == cp_size, (
                    "Parallel Tokenizer grid needs to multiply to CP size."
                )

                model.tokenizer.model.model = model.tokenizer.model.model.to("cuda")
                model.tokenizer.initialize_context_parallel(process_group, wan_cp_grid)

        self.model = model
        self.config = config
        self.batch_size = 1
        self.benchmark_timer = benchmark_timer
        self._apply_entry_offloading()

    def _apply_entry_offloading(self) -> None:
        if self.offload_diffusion_model:
            log.info("[Memory Optimization] Offloading diffusion network to CPU")
            if hasattr(self.model, "conditioner") and self.model.conditioner is not None:
                self.model.conditioner = self.model.conditioner.to("cpu")
        else:
            self.model.net = self.model.net.to("cuda")

        if self.offload_tokenizer:
            log.info("[Memory Optimization] Offloading tokenizer encoder & decoder to CPU")
            if hasattr(self.model.tokenizer, "encoder") and self.model.tokenizer.encoder is not None:
                self.model.tokenizer.encoder = self.model.tokenizer.encoder.to("cpu")
            if hasattr(self.model.tokenizer, "decoder") and self.model.tokenizer.decoder is not None:
                self.model.tokenizer.decoder = self.model.tokenizer.decoder.to("cpu")

        if self.offload_image_context_encoder:
            moved_any = False
            for embedder in self._iter_image_context_embedders():
                if getattr(embedder, "model", None) is not None:
                    embedder.model = embedder.model.to("cpu")
                    moved_any = True
            if moved_any:
                log.info("[Memory Optimization] Offloading image-context encoder to CPU")

        if self.offload_tokenizer or self.offload_image_context_encoder:
            torch.cuda.empty_cache()
        self._log_runtime_state("post_init_offload")

    def _iter_image_context_embedders(self):
        conditioner = getattr(self.model, "conditioner", None)
        if conditioner is None or not hasattr(conditioner, "embedders"):
            return
        for embedder in conditioner.embedders.values():
            if isinstance(embedder, SigLip2EmbImgContext):
                yield embedder

    def _get_module_device(self, module) -> str:
        if module is None:
            return "none"
        try:
            return str(next(module.parameters()).device)
        except StopIteration:
            pass
        except AttributeError:
            return "n/a"
        try:
            return str(next(module.buffers()).device)
        except StopIteration:
            return "no-params"
        except AttributeError:
            return "n/a"

    def _log_gpu_memory(self, stage: str) -> None:
        if not torch.cuda.is_available():
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated_bytes = torch.cuda.memory_allocated()
        reserved_bytes = torch.cuda.memory_reserved()
        max_allocated_bytes = torch.cuda.max_memory_allocated()
        log.debug(
            "[GPU Memory] "
            f"stage={stage} "
            f"free={free_bytes / 1024**3:.2f}GiB "
            f"total={total_bytes / 1024**3:.2f}GiB "
            f"allocated={allocated_bytes / 1024**3:.2f}GiB "
            f"reserved={reserved_bytes / 1024**3:.2f}GiB "
            f"max_allocated={max_allocated_bytes / 1024**3:.2f}GiB"
        )

    def _log_module_devices(self, stage: str) -> None:
        text_encoder = getattr(self.model, "text_encoder", None)
        text_encoder_model = getattr(text_encoder, "model", None) if text_encoder is not None else None
        tokenizer = getattr(self.model, "tokenizer", None)
        tokenizer_encoder = getattr(tokenizer, "encoder", None) if tokenizer is not None else None
        tokenizer_decoder = getattr(tokenizer, "decoder", None) if tokenizer is not None else None
        image_context_devices = [
            self._get_module_device(getattr(embedder, "model", None))
            for embedder in self._iter_image_context_embedders()
        ]
        image_context_state = ",".join(image_context_devices) if image_context_devices else "none"
        log.debug(
            "[Model Devices] "
            f"stage={stage} "
            f"diffusion_net={self._get_module_device(getattr(self.model, 'net', None))} "
            f"conditioner={self._get_module_device(getattr(self.model, 'conditioner', None))} "
            f"text_encoder={self._get_module_device(text_encoder_model)} "
            f"tokenizer_encoder={self._get_module_device(tokenizer_encoder)} "
            f"tokenizer_decoder={self._get_module_device(tokenizer_decoder)} "
            f"image_context_encoder={image_context_state}"
        )

    def _log_runtime_state(self, stage: str) -> None:
        self._log_module_devices(stage)
        self._log_gpu_memory(stage)

    def _maybe_move_text_encoder(self, device: str) -> None:
        if not self.offload_text_encoder or self.model.text_encoder is None:
            return
        if hasattr(self.model.text_encoder, "model") and self.model.text_encoder.model is not None:
            self.model.text_encoder.model = self.model.text_encoder.model.to(device)
            torch.cuda.empty_cache()
            self._log_runtime_state(f"text_encoder->{device}")

    def _maybe_move_tokenizer(self, encoder_device: str | None = None, decoder_device: str | None = None) -> None:
        if not self.offload_tokenizer:
            return
        if (
            encoder_device is not None
            and hasattr(self.model.tokenizer, "encoder")
            and self.model.tokenizer.encoder is not None
        ):
            self.model.tokenizer.encoder = self.model.tokenizer.encoder.to(encoder_device)
        if (
            decoder_device is not None
            and hasattr(self.model.tokenizer, "decoder")
            and self.model.tokenizer.decoder is not None
        ):
            self.model.tokenizer.decoder = self.model.tokenizer.decoder.to(decoder_device)
        torch.cuda.empty_cache()
        self._log_runtime_state(f"tokenizer:encoder={encoder_device or '-'} decoder={decoder_device or '-'}")

    def _maybe_move_image_context_encoder(self, device: str) -> None:
        if not self.offload_image_context_encoder:
            return
        moved_any = False
        for embedder in self._iter_image_context_embedders():
            if getattr(embedder, "model", None) is not None:
                embedder.model = embedder.model.to(device)
                moved_any = True
        if moved_any:
            torch.cuda.empty_cache()
            self._log_runtime_state(f"image_context_encoder->{device}")

    def _maybe_move_conditioner(self, device: str) -> None:
        conditioner = getattr(self.model, "conditioner", None)
        if conditioner is None:
            return
        conditioner = conditioner.to(device)
        self.model.conditioner = conditioner
        torch.cuda.empty_cache()
        self._log_runtime_state(f"conditioner->{device}")

    def _maybe_move_diffusion(self, device: str) -> None:
        if not self.offload_diffusion_model:
            return
        self.model.net = self.model.net.to(device)
        torch.cuda.empty_cache()
        self._log_runtime_state(f"diffusion->{device}")

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prev_output: torch.Tensor,
        text_embedding: torch.Tensor,
        fps: int,
        negative_prompt: str = None,
        control_weight: str = "1.0",
        image_context: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.

            image_context (torch.Tensor, optional): Image context tensor for conditioning. Can be (B, C, H, W).

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = prev_output.shape
        input_key = "video" if T > 1 else "images"

        data_batch = {
            "dataset_name": "video_data",
            input_key: prev_output.squeeze(2),
            "t5_text_embeddings": text_embedding,  # positive prompt embedding. Name has t5 but also supports Reason1.
            "fps": torch.randint(16, 32, (self.batch_size,)).cuda(),  # Random FPS (might be used by model)
            "padding_mask": torch.zeros(
                self.batch_size, 1, H, W, device="cuda"
            ),  # Padding mask (assumed no padding here)
            "num_conditional_frames": 1,  # Specify that the first frame is conditional
            "control_weight": [float(w) for w in control_weight.split(",")],
            "input_video": video,
        }

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.to(dtype=torch.bfloat16, device="cuda", non_blocking=True)

        # Add image context
        if image_context is not None:
            data_batch["image_context"] = image_context.to(
                dtype=torch.bfloat16, device="cuda", non_blocking=True
            ).contiguous()

        # Handle negative prompts for classifier-free guidance
        if negative_prompt is not None:
            assert self.neg_t5_embeddings is not None, "Negative prompt embedding is not computed."
            data_batch["neg_t5_text_embeddings"] = self.neg_t5_embeddings

        return data_batch

    def _pad_prg_labels(self, labels: torch.Tensor, num_video_frames_per_chunk: int) -> torch.Tensor:
        if labels.shape[0] >= num_video_frames_per_chunk:
            return labels[:num_video_frames_per_chunk]
        if labels.shape[0] == 0:
            raise ValueError("No PRG labels; cannot pad")
        if labels.shape[0] == 1:
            padding = labels[-1:].repeat(num_video_frames_per_chunk - labels.shape[0], 1, 1)
            return torch.cat([labels, padding], dim=0)
        while labels.shape[0] < num_video_frames_per_chunk:
            padding = min(labels.shape[0] - 1, num_video_frames_per_chunk - labels.shape[0])
            labels = torch.cat([labels, labels.flip(dims=[0])[:padding]], dim=0)
        return labels

    @staticmethod
    def _prg_labels_to_video(labels: torch.Tensor) -> torch.Tensor:
        """Convert PRG class labels to an RGB video tensor in [0, 1]."""
        return prg_labels_to_rgb(labels).float().div(255.0).unsqueeze(0)

    def _add_teacher_degradation_queries(
        self,
        data_batch: dict[str, torch.Tensor],
        *,
        prg_aux_dir: str,
        chunk_start_frame: int,
        chunk_end_frame: int,
        num_video_frames_per_chunk: int,
        scale: float,
        disabled_query_classes: tuple[str, ...],
        query_intervention: str,
    ) -> torch.Tensor:
        """Attach PRG-privileged teacher queries for teacher-only validation."""
        frame_ids = list(range(chunk_start_frame, chunk_end_frame))
        prg_labels = load_prg_label_frames(prg_aux_dir, frame_ids=frame_ids)
        prg_labels = self._pad_prg_labels(prg_labels, num_video_frames_per_chunk)
        data_batch["prg_label"] = prg_labels.unsqueeze(0)
        data_batch["control_input_prg"] = prg_labels_to_rgb(prg_labels).unsqueeze(0)
        prg_video = self._prg_labels_to_video(prg_labels)

        data_batch["degradation_query_scale"] = scale
        data_batch["disabled_prg_query_classes"] = disabled_query_classes
        data_batch["prg_query_intervention"] = query_intervention
        log.info(
            "VIS+SEG PRG teacher input | "
            f"scale={scale} | disabled={disabled_query_classes} | "
            f"query_intervention={query_intervention}"
        )
        return prg_video

    def _add_prg_dense_gate(
        self,
        data_batch: dict[str, torch.Tensor],
        *,
        prg_aux_dir: str,
        chunk_start_frame: int,
        chunk_end_frame: int,
        num_video_frames_per_chunk: int,
        scale: float,
        hdmap: torch.Tensor | None,
        hdmap_lane_as_generate: bool,
        hdmap_lane_color_threshold: float,
        hdmap_lane_dilation: int,
    ) -> torch.Tensor:
        """Attach lossless PRG labels for dense VIS gating without Query/adapters."""
        frame_ids = list(range(chunk_start_frame, chunk_end_frame))
        prg_labels = load_prg_label_frames(prg_aux_dir, frame_ids=frame_ids)
        prg_labels = self._pad_prg_labels(prg_labels, num_video_frames_per_chunk)
        if hdmap_lane_as_generate:
            if hdmap is None:
                raise ValueError("HD-map is required when routing HD-map lanes through the Generate gate")
            prg_labels_batch, lane_mask = override_prg_labels_with_hdmap_lane_generate(
                prg_labels.unsqueeze(0),
                hdmap.unsqueeze(0),
                color_threshold=hdmap_lane_color_threshold,
                dilation=hdmap_lane_dilation,
            )
            prg_labels = prg_labels_batch[0]
            log.info(
                "PRG dense VIS gate HD-map lane override | "
                f"route=generate | coverage={lane_mask.float().mean().item():.6f}"
            )
        data_batch["prg_label"] = prg_labels.unsqueeze(0)
        data_batch["degradation_query_scale"] = scale
        log.info(f"PRG dense VIS gate input | scale={scale}")
        return self._prg_labels_to_video(prg_labels)

    def _get_num_chunks(
        self, input_frames: torch.Tensor, num_video_frames_per_chunk: int, num_conditional_frames: int
    ) -> tuple[int, int, int]:
        """
        Get the number of chunks for chunk-wise long video generation.
        """
        # Frame number settting for chunk-wise long video generation
        num_total_frames = input_frames.shape[1]
        num_frames_per_chunk = num_video_frames_per_chunk - num_conditional_frames
        if num_video_frames_per_chunk == 1 or num_total_frames <= num_video_frames_per_chunk:
            # Single-chunk path: either explicit 1-frame chunks, or the input is
            # shorter than one chunk and will be padded up by _pad_input_frames.
            num_chunks = 1
        else:
            num_generated_frames_vid2vid = num_total_frames - num_video_frames_per_chunk
            num_chunks = 1 + num_generated_frames_vid2vid // num_frames_per_chunk
            if num_generated_frames_vid2vid % num_frames_per_chunk != 0:
                num_chunks += 1

        return num_total_frames, num_chunks, num_frames_per_chunk

    def _pad_input_frames(
        self,
        input_frames: torch.Tensor,
        num_total_frames: int,
        num_video_frames_per_chunk: int,
        padding_mode: str = "reflect",
    ) -> torch.Tensor:
        """
        Pad input frames if total frames is less than chunk size
        """
        if num_total_frames < num_video_frames_per_chunk:
            # Check whether the input_frames is empty. If so, there is nothing to pad.
            if num_total_frames == 0:
                raise ValueError("No input frames; cannot pad. Verify that video frame counts match.")
            if padding_mode == "repeat":
                last_frame = input_frames[:, -1:, :, :]  # Get the last frame
                padding = last_frame.repeat(1, num_video_frames_per_chunk - num_total_frames, 1, 1)
                input_frames = torch.cat([input_frames, padding], dim=1)
            elif padding_mode == "reflect":
                if input_frames.shape[1] == 1:
                    # Reflect padding is undefined for T=1 (nothing to mirror); fall back to repeat.
                    last_frame = input_frames[:, -1:, :, :]
                    padding_frames = last_frame.repeat(1, num_video_frames_per_chunk - input_frames.shape[1], 1, 1)
                    input_frames = torch.cat([input_frames, padding_frames], dim=1)
                else:
                    while input_frames.shape[1] < num_video_frames_per_chunk:
                        padding = min(input_frames.shape[1] - 1, num_video_frames_per_chunk - input_frames.shape[1])
                        padding_frames = input_frames.flip(dims=[1])[:, :padding, :, :]
                        input_frames = torch.cat([input_frames, padding_frames], dim=1)
            else:
                raise ValueError(f"Invalid padding mode: {padding_mode}")
        return input_frames

    def _read_guided_generation_mask(
        self,
        input_path: str,
        foreground_labels: list[int] | None = None,
        h: int = 704,
        w: int = 1280,
        interpolation: int = cv2.INTER_LINEAR,
        resolution: str = "720",
        max_frames: int | None = None,
    ) -> torch.Tensor:
        """
        Read guided generation mask from path.
        Args:
            input_path (str): Path to guided generation mask. Must be a mp4 or npz file. If a npz file, it must contain
                an 'arr_0' key with shape (T, H, W).
            foreground_labels (list[int], optional): List of label indices to treat as foreground in the mask.
                If None, any non-zero value is treated as foreground. Must be provided and not empty if provided. It is
                effective only when npz file is used for the input_path.
            h (int, optional): Height of the guided generation mask. Defaults to 704.
            w (int, optional): Width of the guided generation mask. Defaults to 1280.
            interpolation (int, optional): Interpolation method to use for resizing the guided generation mask. Defaults
                to cv2.INTER_LINEAR.
            resolution (str, optional): Resolution of the guided generation mask. Defaults to "720".
            max_frames (int, optional): Maximum number of frames to read from the guided generation mask. Defaults to
                None. If provided, only the first max_frames frames will be read from the file.
        Returns:
            torch.Tensor: Guided generation mask tensor (BCTHW, range [0, 1]).
        """
        if str(input_path).endswith(".mp4"):
            control_input, _, _, _ = read_and_process_video(
                str(input_path), resolution=resolution, max_frames=max_frames
            )
            guided_generation_mask = control_input.float() / 255.0  # BCTHW, range [0, 1]
            if foreground_labels is not None:
                log.warning("foreground_labels is ignored for mp4 format")
        elif str(input_path).endswith(".npz"):
            frames = np.load(input_path)
            if "arr_0" in frames:
                frames = frames["arr_0"][:max_frames] if max_frames is not None else frames["arr_0"]
            else:
                raise ValueError(f"Unknown video mask format: {input_path} npz file does not contain 'arr_0'")

            if foreground_labels is not None:
                assert len(foreground_labels) > 0, "foreground_labels must be provided and not empty"
                # set any labels not in foreground_labels to 0
                frames[~np.isin(frames, foreground_labels)] = 0

            frames[frames > 0] = 255
            frames = np.stack([frames, frames, frames], axis=0)[None]
            control_input = resize_video(
                frames.astype(np.float32), h, w, interpolation=interpolation
            )  # BCTHW, range [0, 255]
            guided_generation_mask = torch.from_numpy(control_input).float() / 255.0  # BCTHW, range [0, 1]
        else:
            raise ValueError(f"Unknown video mask format: {input_path} not a mp4 or npz file")

        log.debug(f"resized video mask frames: {guided_generation_mask.shape}")

        return guided_generation_mask

    def construct_latent_weight_map(
        self,
        guided_generation_mask: torch.Tensor,
        h: int = 704,
        w: int = 1280,
        c: int = 16,
    ) -> torch.Tensor:
        """
        Construct latent weight map from guided generation mask.
        Args:
            guided_generation_mask (torch.Tensor): Guided generation mask tensor (B, 3, T, H, W).
            h (int, optional): Height of the latent weight map. Defaults to 704.
            w (int, optional): Width of the latent weight map. Defaults to 1280.
            c (int, optional): Number of channels of the latent weight map. Defaults to 16.
        Returns:
            torch.Tensor: Latent weight map tensor (B, c, T, H, W).
        """
        assert guided_generation_mask.shape[1] == 3, f"guided_generation_mask must have 3 channels,"
        f"but got {guided_generation_mask.shape[1]}."

        weight_map_i = [
            torch.nn.functional.interpolate(
                guided_generation_mask[:, :1, :1, :, :],
                size=(1, h, w),
                mode="trilinear",
                align_corners=False,
            )
        ]
        for wi in range(1, guided_generation_mask.shape[2], 4):
            weight_map_i += [
                torch.nn.functional.interpolate(
                    guided_generation_mask[:, :1, wi : wi + 4],
                    size=(1, h, w),
                    mode="trilinear",
                    align_corners=False,
                )
            ]
        weight_map = torch.cat(weight_map_i, dim=2).repeat(1, c, 1, 1, 1)

        return weight_map

    @staticmethod
    def _save_processed_image_context(image_context: torch.Tensor, output_path: str) -> None:
        image = normalized_float_to_uint8(image_context[0]).permute(1, 2, 0).cpu().numpy()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image_bgr)

    @torch.no_grad()
    def generate_img2world(
        self,
        prompt: str | torch.Tensor | list[str] | dict[str, str],
        video_path: str,
        guidance: int = 7,
        seed: int = 1,
        resolution: str = "720",
        num_conditional_frames: int = 1,
        num_video_frames_per_chunk: int = 93,
        num_steps: int = 35,
        control_weight: str = "1.0",
        sigma_max: float | None = None,
        hint_key: list[str] = ["edge"],
        preset_edge_threshold: str = "medium",
        preset_blur_strength: str = "medium",
        seg_control_prompt: str | None = None,
        input_control_video_paths: dict[str, str] | None = None,
        show_control_condition: bool = False,
        show_input: bool = False,
        image_context_path: Optional[str] = None,
        reference_image_output_path: Optional[str] = None,
        keep_input_resolution: bool = True,
        negative_prompt: str | None = None,
        max_frames: int | None = None,
        context_frame_idx: int | None = None,
        guided_generation_mask: str | None = None,
        guided_generation_step_threshold: int = 25,
        guided_generation_foreground_labels: list[int] | None = None,
        use_teacher_degradation_query: bool = False,
        use_prg_dense_gate: bool = False,
        prg_dense_gate_hdmap_lane_as_generate: bool = False,
        prg_dense_gate_hdmap_lane_color_threshold: float = 60.0,
        prg_dense_gate_hdmap_lane_dilation: int = 9,
        prg_aux_dir: str | None = None,
        hdmap_path: str | None = None,
        degradation_query_scale: float = 1.0,
        disabled_prg_query_classes: tuple[str, ...] = (),
        prg_query_intervention: str = "none",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], int, tuple[int, int]]:
        """
        Generates a video based on an input video and text prompt.
        Supports chunk-wise long video generation.

        Args:
            prompt (str): The text prompt describing the desired video content/style.
            video_path (str): Path to the input conditional video.
            guidance (int, optional): Classifier-free guidance scale. Defaults to 7.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
            resolution (str, optional): Resolution of the video (720-default, 480, etc). Defaults to 720.
            image_context_path (str, optional): Path to image file to use as image context. If None, uses random frame from video. Will be ignored and use input video if context_frame_idx is provided.
            keep_input_resolution (bool, optional): Whether to keep the exact dimension of the. Defaults to True.
            negative_prompt (str, optional): Negative prompt for classifier-free guidance. Defaults to None.
            max_frames (int, optional): Maximum number of frames to read from the video. Defaults to None. 1 for image.
            context_frame_idx (int, optional): Frame index of the input video to use as image context. Defaults to None. In this case, can still use image_context_path to provide image context.
        Returns:
            torch.Tensor: The generated video tensor (B, C, T, H, W) in the range [-1, 1].
            dict[str, torch.Tensor]: Dictionary mapping hint key to the corresponding control input video tensor.
            int: Frames per second of the original input video.
            tuple[int, int]: Original height and width of the input video.

        Raises:
            ValueError: If the input video is empty or invalid.
        """
        # --------Input processing--------
        # Process input video and get meta info.
        log.debug("Stage | video_load")
        # aspect_ratio is width / height
        # input_frames is (C, T, H, W)
        input_frames, fps, aspect_ratio, original_hw = read_and_process_video(
            video_path, resolution=resolution, max_frames=max_frames
        )
        if input_frames.shape[1] == 0:
            raise ValueError("Input video is empty")
        log.debug(f"Stage | video_load done ({input_frames.shape[1]} frames, {fps}fps)")
        self._log_gpu_memory("after_input_video_load")

        hdmap_frames = None
        if use_teacher_degradation_query and hdmap_path is None:
            raise ValueError("hdmap_path must be provided when use_teacher_degradation_query=True")
        if hdmap_path is not None:
            hdmap_frames, hdmap_fps, _, hdmap_original_hw = read_and_process_video(
                hdmap_path,
                resolution=resolution,
                max_frames=max_frames,
            )
            if hdmap_frames.shape != input_frames.shape:
                raise ValueError(
                    f"HD-map tensor {tuple(hdmap_frames.shape)} does not match input {tuple(input_frames.shape)}"
                )
            if tuple(hdmap_original_hw) != tuple(original_hw):
                raise ValueError(f"HD-map source size {hdmap_original_hw} does not match input {original_hw}")
            if abs(float(hdmap_fps) - float(fps)) > 1e-3:
                raise ValueError(f"HD-map FPS {hdmap_fps} does not match input FPS {fps}")
            log.debug(f"Stage | hdmap_load done ({hdmap_frames.shape[1]} frames, {hdmap_fps}fps)")

        if guided_generation_mask is not None:
            guided_generation_mask = self._read_guided_generation_mask(
                guided_generation_mask,
                h=input_frames.shape[2],
                w=input_frames.shape[3],
                foreground_labels=guided_generation_foreground_labels,
                resolution=resolution,
                max_frames=max_frames,
            ).squeeze(0)

        # Get text context embeddings
        log.debug("Stage | text_embeddings")
        with _maybe_get_timer(self.benchmark_timer, "get_text_embeddings"):
            self._maybe_move_text_encoder("cuda")
            if self.text_encoder_class == "T5":
                text_embeddings = get_t5_from_prompt(prompt, text_encoder_class="T5", cache_dir=self.cache_dir)
            else:
                text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    {"ai_caption": [prompt], "images": None}, input_caption_key="ai_caption"
                )
            if negative_prompt:
                log.debug("Stage | negative_text_embeddings")
                if self.text_encoder_class == "T5":
                    neg_text_embeddings = get_t5_from_prompt(
                        negative_prompt, text_encoder_class="T5", cache_dir=self.cache_dir
                    )
                else:
                    neg_text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                        {"ai_caption": [negative_prompt], "images": None}, input_caption_key="ai_caption"
                    )
                self.neg_t5_embeddings = neg_text_embeddings
            self._maybe_move_text_encoder("cpu")
            self._log_gpu_memory("after_text_embeddings")
        log.debug("Stage | text_embeddings done")

        # Process image context if provided; else will be None
        with _maybe_get_timer(self.benchmark_timer, "preprocessing"):
            if context_frame_idx is not None:
                image_context_path = video_path
            image_context = read_and_process_image_context(
                image_context_path,
                resolution=(VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]),
                resize=True,
                context_frame_idx=context_frame_idx,
            )
            if image_context is not None and reference_image_output_path is not None:
                self._save_processed_image_context(image_context, reference_image_output_path)
            # Load control inputs from paths, or optionally compute on-the-fly, and add to data batch.
            log.debug("Stage | control_inputs")
            control_input_dict, mask_video_dict = read_and_process_control_input(
                video_path=video_path,
                input_control_paths=input_control_video_paths,
                hint_key=hint_key,
                resolution=resolution,
                seg_control_prompt=seg_control_prompt,
                max_frames=max_frames,
            )
            log.debug(f"Stage | control_inputs done (modalities={','.join(hint_key)})")
            self._log_gpu_memory("after_control_input_prep")

            # -------- Stuff to handle chunk-wise long video generation --------
            num_total_frames, num_chunks, num_frames_per_chunk = self._get_num_chunks(
                input_frames, num_video_frames_per_chunk, num_conditional_frames
            )
            # Pad input frames if total frames is less than chunk size
            input_frames = self._pad_input_frames(input_frames, num_total_frames, num_video_frames_per_chunk)
            if guided_generation_mask is not None:
                guided_generation_mask = self._pad_input_frames(
                    guided_generation_mask, num_total_frames, num_video_frames_per_chunk
                )
            all_chunks, time_per_chunk = [], []
            # Initialize control_video_dict to accumulate control inputs across chunks
            control_video_dict = {}
            all_control_chunks = {key: [] for key in hint_key}
            all_hdmap_chunks = []
            all_prg_label_chunks = []
            all_prg_control_chunks = []
            # For first chunk, use zeros as input (after normalization it is 0)
            prev_output = torch.zeros_like(input_frames[:, :num_video_frames_per_chunk]).to(torch.uint8).cuda()[None]

        # --------Start of chunk-wise long video generation--------
        self.model.eval()
        for chunk_id in range(num_chunks):
            stage = f"chunk_{chunk_id + 1}/{num_chunks}"
            log.debug(f"Stage | {stage} | encode")
            with _maybe_get_timer(self.benchmark_timer, "generate_chunk"):
                start_time = time.perf_counter()

                # Calculate start frame for this chunk
                chunk_start_frame = chunk_id * num_frames_per_chunk
                # Auxiliary media only contains the original frames. Padding is
                # applied independently after slicing, so never expose padded
                # input indices to PRG labels, HD-map, or public controls.
                chunk_end_frame = min(chunk_start_frame + num_video_frames_per_chunk, num_total_frames)

                x0_clean = None
                x0_spatial_condition = None
                if input_frames is not None:
                    cur_input_frames = input_frames[:, chunk_start_frame:chunk_end_frame]
                    cur_input_frames = self._pad_input_frames(
                        cur_input_frames, cur_input_frames.shape[1], num_video_frames_per_chunk
                    )
                    if sigma_max is not None or guided_generation_mask is not None:
                        self._maybe_move_tokenizer(encoder_device="cuda")
                        x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None].cuda(
                            non_blocking=True
                        )
                        x0 = self.model.encode(x0).contiguous()
                        self._maybe_move_tokenizer(encoder_device="cpu")
                        if sigma_max is not None:
                            x0_clean = x0

                        if guided_generation_mask is not None:
                            _, C, T, H, W = x0.shape
                            cur_guided_generation_mask = guided_generation_mask[:, chunk_start_frame:chunk_end_frame]
                            cur_guided_generation_mask = self._pad_input_frames(
                                cur_guided_generation_mask,
                                cur_guided_generation_mask.shape[1],
                                num_video_frames_per_chunk,
                            )
                            x_sigma_mask = self.construct_latent_weight_map(
                                cur_guided_generation_mask.unsqueeze(0), h=H, w=W, c=C
                            ).cuda(non_blocking=True)
                            x0_spatial_condition = {
                                "x0": x0,
                                "x_sigma_mask": x_sigma_mask,
                                "step_threshold": guided_generation_step_threshold,
                            }

                if isinstance(text_embeddings, list):
                    text_emb_idx = min(chunk_id, len(text_embeddings) - 1)
                    text_embedding = text_embeddings[text_emb_idx]
                else:
                    text_embedding = text_embeddings

                # Prepare the data batch with current input. Note: this doesn't include control inputs yet.
                data_batch = self._get_data_batch_input(
                    cur_input_frames,
                    prev_output,
                    text_embedding,
                    fps,
                    negative_prompt=negative_prompt,
                    control_weight=control_weight,
                    image_context=image_context,
                )

                # Process control inputs as specified in the hint_key list.
                # If pre-computed control inputs are provided, load them into the data batch.
                for k, v in control_input_dict.items():
                    cur_control_input = v[:, chunk_start_frame:chunk_end_frame]
                    cur_control_input = self._pad_input_frames(
                        cur_control_input, cur_control_input.shape[1], num_video_frames_per_chunk
                    )
                    data_batch[k] = cur_control_input
                    if k == "control_input_inpaint_mask":
                        data_batch["control_input_inpaint"] = cur_input_frames
                cur_hdmap = None
                if hdmap_frames is not None:
                    cur_hdmap = hdmap_frames[:, chunk_start_frame:chunk_end_frame]
                    cur_hdmap = self._pad_input_frames(
                        cur_hdmap,
                        cur_hdmap.shape[1],
                        num_video_frames_per_chunk,
                    )
                    data_batch["control_input_seg"] = cur_hdmap.unsqueeze(0)
                    hdmap_control = uint8_to_normalized_float(cur_hdmap, dtype=torch.float32).unsqueeze(0)
                    if chunk_id == 0:
                        all_hdmap_chunks.append(hdmap_control)
                    else:
                        all_hdmap_chunks.append(hdmap_control[:, :, num_conditional_frames:, :, :])
                # Otherwise, compute control inputs on-the-fly via the augmentor（applicable to edge and vis).
                data_batch = get_augmentor_for_eval(
                    data_dict=data_batch,
                    input_keys=["input_video"],
                    output_keys=hint_key,
                    preset_edge_threshold=preset_edge_threshold,
                    preset_blur_strength=preset_blur_strength,
                )
                if use_teacher_degradation_query:
                    if prg_aux_dir is None:
                        raise ValueError("prg_aux_dir must be provided when use_teacher_degradation_query=True")
                    prg_label_video = self._add_teacher_degradation_queries(
                        data_batch,
                        prg_aux_dir=prg_aux_dir,
                        chunk_start_frame=chunk_start_frame,
                        chunk_end_frame=chunk_end_frame,
                        num_video_frames_per_chunk=num_video_frames_per_chunk,
                        scale=degradation_query_scale,
                        disabled_query_classes=disabled_prg_query_classes,
                        query_intervention=prg_query_intervention,
                    )
                    if chunk_id == 0:
                        all_prg_label_chunks.append(prg_label_video)
                    else:
                        all_prg_label_chunks.append(prg_label_video[:, :, num_conditional_frames:, :, :])
                    prg_control = uint8_to_normalized_float(data_batch["control_input_prg"], dtype=torch.float32)
                    if chunk_id == 0:
                        all_prg_control_chunks.append(prg_control)
                    else:
                        all_prg_control_chunks.append(prg_control[:, :, num_conditional_frames:, :, :])
                elif use_prg_dense_gate:
                    if prg_aux_dir is None:
                        raise ValueError("prg_aux_dir must be provided when use_prg_dense_gate=True")
                    prg_label_video = self._add_prg_dense_gate(
                        data_batch,
                        prg_aux_dir=prg_aux_dir,
                        chunk_start_frame=chunk_start_frame,
                        chunk_end_frame=chunk_end_frame,
                        num_video_frames_per_chunk=num_video_frames_per_chunk,
                        scale=degradation_query_scale,
                        hdmap=cur_hdmap,
                        hdmap_lane_as_generate=prg_dense_gate_hdmap_lane_as_generate,
                        hdmap_lane_color_threshold=prg_dense_gate_hdmap_lane_color_threshold,
                        hdmap_lane_dilation=prg_dense_gate_hdmap_lane_dilation,
                    )
                    if chunk_id == 0:
                        all_prg_label_chunks.append(prg_label_video)
                    else:
                        all_prg_label_chunks.append(prg_label_video[:, :, num_conditional_frames:, :, :])
                self._log_gpu_memory(f"chunk_{chunk_id + 1}_after_augmentor")

                if chunk_id == 0:
                    data_batch[NUM_CONDITIONAL_FRAMES_KEY] = 0
                else:
                    data_batch[NUM_CONDITIONAL_FRAMES_KEY] = (
                        1 + (num_conditional_frames - 1) // 4
                    )  # tokenizer temporal compression is 4x
                if guided_generation_mask is not None:
                    data_batch["x0_spatial_condition"] = x0_spatial_condition

                random.seed(seed)
                seed = random.randint(0, 1000000)
                log.debug(f"Seed: {seed}")
                self._log_runtime_state(f"chunk_{chunk_id + 1}_before_sampling")

                self._maybe_move_conditioner("cuda")
                self._maybe_move_image_context_encoder("cuda")
                self._maybe_move_tokenizer(encoder_device="cuda")
                prepared = self.model.prepare_generation_from_batch(
                    data_batch,
                    n_sample=1,
                    guidance=guidance,
                    seed=seed,
                    is_negative_prompt=negative_prompt is not None,
                    x0_clean=x0_clean,
                    sigma_max=sigma_max,
                    num_steps=num_steps,
                )
                self._log_gpu_memory(f"chunk_{chunk_id + 1}_after_prepare")
                # Guided-generation latents are only needed during prepare.
                data_batch.pop("x0_spatial_condition", None)
                x0_spatial_condition = None
                x0_clean = None
                self._maybe_move_tokenizer(encoder_device="cpu")
                self._maybe_move_image_context_encoder("cpu")
                self._maybe_move_conditioner("cpu")
                self._maybe_move_diffusion("cuda")
                log.debug(f"Stage | {stage} | sample ({num_steps} steps)")
                sample = self.model.generate_samples_from_prepared(prepared)
                del prepared
                torch.cuda.empty_cache()
                self._log_gpu_memory(f"chunk_{chunk_id + 1}_after_sampling")
                self._maybe_move_diffusion("cpu")
                self._maybe_move_image_context_encoder("cpu")
                self._maybe_move_tokenizer(decoder_device="cuda")
                log.debug(f"Stage | {stage} | decode")
                video = self.model.decode(sample).cpu()  # Shape: (1, C, T, H, W)
                del sample
                torch.cuda.empty_cache()
                self._log_gpu_memory(f"chunk_{chunk_id + 1}_after_decode")
                self._maybe_move_tokenizer(decoder_device="cpu")

                # For visualization: concatenate condition and input videos with generated video
                video_cat = video
                conditions = []
                if show_input and input_frames is not None:
                    x0 = uint8_to_normalized_float(cur_input_frames, dtype=torch.bfloat16)[None]
                    x0 = x0.to(device=video_cat.device)
                    video_cat = torch.cat([x0, video_cat], dim=-1)

                # Accumulate control inputs for each chunk
                for key in hint_key:
                    control_input = data_batch["control_input_" + key]
                    if f"control_input_{key}_mask" in data_batch:
                        mask = data_batch[f"control_input_{key}_mask"].to(device=control_input.device)
                        control_input = (control_input + 1) / 2 * mask * 2 - 1

                    control_input = control_input.cpu()

                    # Store control input for this chunk
                    if chunk_id == 0:
                        all_control_chunks[key].append(control_input)
                    else:
                        # For subsequent chunks, only append the non-overlapping frames
                        all_control_chunks[key].append(control_input[:, :, num_conditional_frames:, :, :])

                    if show_control_condition:
                        conditions += [control_input.to(device=video_cat.device)]

                if show_control_condition:
                    video_cat = torch.cat([*conditions, video_cat], dim=-1)

                if chunk_id == 0:
                    all_chunks.append(video_cat)
                else:
                    # For subsequent chunks, only append the non-overlapping frames
                    all_chunks.append(video_cat[:, :, num_conditional_frames:, :, :])

                # For next chunk, use last conditional_frames as input
                if chunk_id < num_chunks - 1:  # Don't need to prepare next input for last chunk
                    last_frames = video[
                        :, :, video.shape[2] - num_conditional_frames :, :, :
                    ]  # (1, C, num_conditional_frames, H, W)
                    # Convert to uint8 [0, 255]
                    last_frames_uint8 = normalized_float_to_uint8(last_frames)
                    # Create blank frames for the rest
                    blank_frames = torch.zeros(
                        (
                            1,
                            3,
                            num_video_frames_per_chunk - num_conditional_frames,
                            video.shape[-2],
                            video.shape[-1],
                        ),
                        dtype=torch.uint8,
                        device=video.device,
                    )
                    prev_output = torch.cat([last_frames_uint8, blank_frames], dim=2)
                    del last_frames, last_frames_uint8, blank_frames

                del data_batch, cur_input_frames, video_cat, conditions, video
                torch.cuda.empty_cache()
                self._log_gpu_memory(f"chunk_{chunk_id + 1}_after_cleanup")
                end_time = time.perf_counter()
                time_per_chunk.append(end_time - start_time)
                log.info(f"chunk {chunk_id + 1}/{num_chunks} done ({end_time - start_time:.1f}s)")

        log.debug("Stage | postprocess")
        with _maybe_get_timer(self.benchmark_timer, "postprocessing"):
            # Concatenate all chunks along time
            full_video = torch.cat(all_chunks, dim=2)  # (1, C, T, H, W)
            # Keep only the original number of frames
            full_video = full_video[:, :, :num_total_frames, :, :]

            full_video = full_video.cpu()
            # Concatenate all control chunks and trim to original frames
            for key in hint_key:
                if all_control_chunks[key]:
                    control_video_dict[key] = torch.cat(all_control_chunks[key], dim=2)  # (1, C, T, H, W)
                    # Keep only the original number of frames
                    control_video_dict[key] = control_video_dict[key][:, :, :num_total_frames, :, :]
            if all_hdmap_chunks:
                control_video_dict["hdmap"] = torch.cat(all_hdmap_chunks, dim=2)[:, :, :num_total_frames, :, :]
            if all_prg_control_chunks:
                control_video_dict["prg"] = torch.cat(all_prg_control_chunks, dim=2)[:, :, :num_total_frames, :, :]

            if keep_input_resolution:
                # reshape output video to match the input video resolution
                full_video = reshape_output_video_to_input_resolution(
                    full_video, hint_key, show_control_condition, show_input, original_hw
                )
                # Also resize control videos to match input resolution
                for key, control_video in control_video_dict.items():
                    control_video_dict[key] = reshape_output_video_to_input_resolution(
                        control_video, [key], False, False, original_hw
                    )
            if all_prg_label_chunks:
                mask_video_dict["prg_label"] = torch.cat(all_prg_label_chunks, dim=2)[0, :, :num_total_frames, :, :]
        log.debug("Stage | postprocess done")

        if guided_generation_mask is not None:
            if guided_generation_mask.ndim == 3:
                guided_generation_mask = guided_generation_mask.unsqueeze(0)
            mask_video_dict["guided_generation"] = guided_generation_mask
        return full_video, control_video_dict, mask_video_dict, fps, original_hw
