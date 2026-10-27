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

import os
from typing import Callable, Dict, Tuple

import attrs
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import tqdm
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.tensor import DTensor, distribute_tensor

from cosmos_transfer2._src.imaginaire.checkpointer.s3_filesystem import S3StorageReader
from cosmos_transfer2._src.imaginaire.flags import INTERNAL
from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.imaginaire.utils import log, misc
from cosmos_transfer2._src.imaginaire.utils.context_parallel import (
    broadcast_split_tensor,
    cat_outputs_cp,
    find_split,
)
from cosmos_transfer2._src.imaginaire.utils.distributed import get_rank, get_world_size
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.predict2.checkpointer.dcp import (
    ModelWrapper,
    load_model_state_dict_subset_from_dcp,
    map_multi_branch_key_to_single_branch,
)
from cosmos_transfer2._src.predict2.conditioner import DataType
from cosmos_transfer2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.conditioner import ControlVideo2WorldCondition
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import CTRL_HINT_KEYS
from cosmos_transfer2._src.transfer2.models.control_weights import (
    masked_latent_mse,
    resize_control_weight_maps,
    resize_video_mask_to_latent,
)
from cosmos_transfer2._src.transfer2.models.prg_losses import (
    build_hdmap_lane_valid_mask,
    build_prg_valid_token_mask,
    build_seg_only_control_routing,
    class_balanced_regional_flow_loss,
    disable_global_prg_queries,
    downsample_hdmap_lane_mask_to_latent,
    extract_hdmap_lane_mask,
    hdmap_lane_fidelity_losses,
    should_update_hdmap_lane,
)
from cosmos_transfer2._src.transfer2.networks.degradation_query import build_degradation_route_probs

IS_PREPROCESSED_KEY = "is_preprocessed"


@attrs.define(slots=False)
class ControlVideo2WorldRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    base_load_from: LazyDict = None
    min_num_conditional_frames: int = 0  # Minimum number of latent conditional frames
    max_num_conditional_frames: int = 2  # Maximum number of latent conditional frames
    copy_weight_strategy: str = (
        "first_n"  # How to copy weights from base model to control branch. "first_n" or "spaced_n"
    )
    hint_keys: str = "_".join([key.replace("control_input_", "") for key in CTRL_HINT_KEYS.keys()])
    use_reference_image: bool = False  # Whether to use reference image as control input
    train_reference_image_modules: bool = True
    use_prg_teacher_degradation_query: bool = False
    use_prg_dense_gate: bool = False
    use_prg_adaptive_dense_gate: bool = False
    control_branch_checkpoint_paths: tuple[str, ...] = ()
    control_branch_overlay_checkpoint_path: str | None = None
    control_branch_overlay_index: int = 1
    control_branch_overlay_indices: tuple[int, ...] = ()
    control_branch_overlay_from_single_branch: bool = False
    trainable_control_branch_indices: tuple[int, ...] = ()
    use_control_masks_during_training: bool = False
    apply_control_masks_to_residuals: bool = True
    normalize_control_weight_maps: bool = True
    freeze_base_model_for_prg_teacher: bool = False
    prg_base_flow_weight: float = 1.0
    prg_regional_flow_weight: float = 0.0
    prg_class_flow_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    prg_min_region_mass: float = 1.0
    prg_region_epsilon: float = 1e-6
    prg_disable_global_query: bool = False
    hdmap_lane_rf_weight: float = 0.0
    hdmap_lane_edge_weight: float = 0.0
    hdmap_lane_color_threshold: float = 60.0
    hdmap_lane_dilation: int = 9
    hdmap_lane_charbonnier_epsilon: float = 1e-3
    hdmap_lane_min_mass: float = 1.0
    hdmap_lane_epsilon: float = 1e-6
    hdmap_lane_seg_only: bool = False
    hdmap_lane_update_interval: int = 1


class ControlVideo2WorldModelRectifiedFlow(Video2WorldModelRectifiedFlow):
    """
    ImaginaireModel instance of the VACE-styled controlnet for training.
    """

    def __init__(self, config: ControlVideo2WorldRectifiedFlowConfig, *args, **kwargs):
        self.is_new_training = True
        self.copy_weight_strategy = config.copy_weight_strategy
        self.hint_keys = ["control_input_" + key for key in config.hint_keys.split("_")]
        super().__init__(config, *args, **kwargs)
        if config.trainable_control_branch_indices:
            if config.freeze_base_model_for_prg_teacher:
                raise ValueError("Selective control-branch training cannot be combined with PRG teacher freezing")
            self._freeze_for_control_branch_training(config.trainable_control_branch_indices)
        if config.use_prg_teacher_degradation_query:
            if not getattr(self.net, "use_vis_prg_teacher_query", False):
                raise ValueError("PRG teacher training requires net.use_vis_prg_teacher_query=True")
            if config.freeze_base_model_for_prg_teacher:
                for name, param in self.named_parameters():
                    train_query = name.startswith("net.prg_teacher_query_branch.") or (
                        name.startswith("net.") and ".routed_query_adapter." in name
                    )
                    param.requires_grad = train_query
                log.info("Frozen base model parameters for PRG teacher training", rank0_only=True)
            log.info(
                "Enabled Render-VIS + target-HD-map-SEG + privileged-PRG-SEG teacher query training",
                rank0_only=True,
            )
        if config.use_prg_dense_gate or config.use_prg_adaptive_dense_gate:
            if config.use_prg_dense_gate and config.use_prg_adaptive_dense_gate:
                raise ValueError("Fixed and adaptive PRG dense gates are mutually exclusive")
            if config.use_prg_teacher_degradation_query:
                raise ValueError("Pure PRG dense gating cannot be combined with PRG teacher Query mode")
            net_gate_enabled = getattr(
                self.net,
                "use_prg_adaptive_dense_gate" if config.use_prg_adaptive_dense_gate else "use_prg_dense_gate",
                False,
            )
            if not net_gate_enabled:
                raise ValueError("Model and network PRG dense-gate modes must match")
            mode = "adaptive" if config.use_prg_adaptive_dense_gate else "fixed"
            log.info(f"Enabled lossless-PRG {mode} dense VIS gating without Query/adapters", rank0_only=True)
        if config.hdmap_lane_seg_only:
            if config.hdmap_lane_rf_weight <= 0 and config.hdmap_lane_edge_weight <= 0:
                raise ValueError("SEG-only lane gradient routing requires a positive HD-map lane loss weight")
            if self.net.num_control_branches < 2:
                raise ValueError("SEG-only lane gradient routing requires separate VIS and SEG branches")
            if config.hdmap_lane_update_interval < 1:
                raise ValueError("HD-map lane update interval must be positive")
            log.info(
                f"Enabled SEG-only second-pass HD-map lane supervision every "
                f"{config.hdmap_lane_update_interval} optimizer steps",
                rank0_only=True,
            )
        log.info(self.net, rank0_only=True)

    def _freeze_for_control_branch_training(self, branch_indices: tuple[int, ...]) -> None:
        indices = tuple(dict.fromkeys(branch_indices))
        if len(indices) != len(branch_indices):
            raise ValueError(f"Duplicate trainable control branch indices: {branch_indices}")
        invalid = [index for index in indices if index < 0 or index >= self.net.num_control_branches]
        if invalid:
            raise ValueError(
                f"Invalid trainable control branch indices {invalid}; model has {self.net.num_control_branches} branches"
            )
        if hasattr(self.net, "after_proj"):
            raise ValueError("Selective control-branch training requires branch-local after_proj modules")

        self.net.requires_grad_(False)
        for index in indices:
            self.net.control_embedder[index].requires_grad_(True)
            getattr(self.net, f"control_blocks_{index}").requires_grad_(True)
        if getattr(self.net, "use_prg_adaptive_dense_gate", False):
            self.net.prg_adaptive_vis_gate.requires_grad_(True)
        trainable = [name for name, param in self.net.named_parameters() if param.requires_grad]
        expected_prefixes = tuple(
            prefix for index in indices for prefix in (f"control_embedder.{index}.", f"control_blocks_{index}.")
        )
        if getattr(self.net, "use_prg_adaptive_dense_gate", False):
            expected_prefixes += ("prg_adaptive_vis_gate.",)
        if not trainable or any(not name.startswith(expected_prefixes) for name in trainable):
            raise RuntimeError(f"Unexpected selective control-branch trainable set: {trainable[:20]}")
        log.info(
            f"Frozen base/control branches; trainable control branch indices={indices} tensors={len(trainable)}",
            rank0_only=True,
        )

    def _ensure_control_branches_loaded(self) -> None:
        checkpoint_paths = list(self.config.control_branch_checkpoint_paths)
        if not checkpoint_paths:
            return
        if getattr(self, "_control_branches_loaded", False):
            return
        if len(checkpoint_paths) != self.net.num_control_branches:
            raise RuntimeError(
                "Multi-branch training requires one pretrained checkpoint per control branch: "
                f"expected {self.net.num_control_branches}, got {len(checkpoint_paths)}"
            )
        self.load_multi_branch_checkpoints(checkpoint_paths)
        overlay_path = self.config.control_branch_overlay_checkpoint_path
        if overlay_path is not None:
            overlay_indices = self.config.control_branch_overlay_indices or (self.config.control_branch_overlay_index,)
            if len(set(overlay_indices)) != len(overlay_indices):
                raise ValueError(f"Duplicate control branch overlay indices: {overlay_indices}")
            if self.config.control_branch_overlay_from_single_branch and len(overlay_indices) != 1:
                raise ValueError("A single-branch checkpoint overlay requires exactly one target branch")
            invalid_indices = [
                index for index in overlay_indices if index < 0 or index >= self.net.num_control_branches
            ]
            if invalid_indices:
                raise ValueError(
                    f"Invalid control branch overlay indices {invalid_indices}; "
                    f"model has {self.net.num_control_branches} branches"
                )
            loaded_keys = load_model_state_dict_subset_from_dcp(
                self,
                overlay_path,
                lambda key: any(
                    key.startswith(f"net.control_embedder.{index}.") or key.startswith(f"net.control_blocks_{index}.")
                    for index in overlay_indices
                ),
                load_ema_to_reg=True,
                model_to_checkpoint_key=(
                    (lambda key: map_multi_branch_key_to_single_branch(key, overlay_indices[0]))
                    if self.config.control_branch_overlay_from_single_branch
                    else None
                ),
            )
            log.info(
                f"Training control branch overlay | branches={overlay_indices} | path={overlay_path} | "
                f"tensors={len(loaded_keys)}",
                rank0_only=True,
            )
        if self.config.ema.enabled:
            self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)
        self._control_branches_loaded = True

    def _prepare_vis_seg_prg_teacher_inputs(self, data_batch: dict[str, torch.Tensor]) -> None:
        if "prg_label" not in data_batch:
            raise KeyError("PRG teacher training requires data_batch['prg_label']")
        if "control_input_vis" not in data_batch:
            raise KeyError("PRG teacher training requires data_batch['control_input_vis'] as render input")
        if "control_input_seg" not in data_batch:
            raise KeyError("PRG teacher requires data_batch['control_input_seg'] as target-view HD-map input")
        if "control_input_prg" not in data_batch:
            raise KeyError("PRG teacher requires data_batch['control_input_prg'] as privileged PRG semantic input")
        data_batch["degradation_query_scale"] = data_batch.get("degradation_query_scale", 1.0)

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        super().on_train_start(memory_format)
        self._ensure_control_branches_loaded()

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        interval = self.config.hdmap_lane_update_interval
        self._hdmap_lane_update_active = not self.config.hdmap_lane_seg_only or should_update_hdmap_lane(
            iteration, interval
        )
        try:
            return super().training_step(data_batch, iteration)
        finally:
            del self._hdmap_lane_update_active

    def forward(self, data_batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if self.config.use_prg_teacher_degradation_query:
            if not getattr(self, "_control_branches_loaded", False):
                raise RuntimeError("VIS and SEG control branches must be loaded before PRG teacher training")
            self._prepare_vis_seg_prg_teacher_inputs(data_batch)

        output_batch, loss = super().forward(data_batch)
        lane_update_active = getattr(
            self,
            "_hdmap_lane_update_active",
            self.config.hdmap_lane_update_interval == 1,
        )
        if self.config.hdmap_lane_seg_only and lane_update_active:
            loss, lane_output = self._add_seg_only_lane_loss(output_batch, loss)
            output_batch.update(lane_output)
        if self.config.hdmap_lane_seg_only:
            output_batch["hdmap_lane_update_active"] = loss.new_tensor(float(lane_update_active))
        if self.config.use_prg_teacher_degradation_query:
            query_log_info = getattr(self.net, "prg_query_log_info", None)
            if query_log_info is not None:
                output_batch.update(
                    {
                        "prg_teacher_query_norm": query_log_info["query_norm"],
                        "prg_teacher_class_presence": query_log_info["class_presence"],
                    }
                )
        if self.config.use_prg_adaptive_dense_gate:
            gate_log_info = getattr(self.net, "prg_adaptive_gate_log_info", None)
            if gate_log_info is not None:
                output_batch.update({f"prg_adaptive_gate_{key}": value for key, value in gate_log_info.items()})
        return output_batch, loss

    def _add_seg_only_lane_loss(
        self,
        output_batch: dict[str, torch.Tensor],
        global_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = getattr(self, "_seg_only_lane_target", None)
        if target is None:
            raise RuntimeError("SEG-only lane target was not captured during the primary loss computation")
        del self._seg_only_lane_target
        condition = output_batch["condition"]
        seg_index = self.net.prg_teacher_hdmap_branch_index
        branch_weights, detached_indices = build_seg_only_control_routing(self.net.num_control_branches, seg_index)
        seg_condition = condition.set_control_condition(condition.latent_control_input, branch_weights)
        seg_condition = seg_condition.set_detached_control_branches(detached_indices)
        noise = target + output_batch["x0"]
        seg_prediction = self.denoise(
            noise=noise,
            xt_B_C_T_H_W=output_batch["xt"].to(**self.tensor_kwargs),
            timesteps_B_T=output_batch["timesteps"],
            condition=seg_condition,
        )
        lane_valid = build_hdmap_lane_valid_mask(
            condition.hdmap_lane_probs,
            padding_mask=condition.padding_mask,
            conditional_frame_mask=condition.condition_video_input_mask_B_C_T_H_W,
        )
        lane = hdmap_lane_fidelity_losses(
            seg_prediction,
            target,
            condition.hdmap_lane_clean_latent,
            condition.hdmap_lane_probs,
            output_batch["sigma"].reshape(-1),
            valid_mask=lane_valid,
            charbonnier_epsilon=self.config.hdmap_lane_charbonnier_epsilon,
            min_region_mass=self.config.hdmap_lane_min_mass,
            epsilon=self.config.hdmap_lane_epsilon,
            compute_edge=self.config.hdmap_lane_edge_weight > 0,
        )
        lane_per_instance = self.config.hdmap_lane_rf_weight * lane.rf + self.config.hdmap_lane_edge_weight * lane.edge
        time_weights = self.rectified_flow.train_time_weight(
            output_batch["timesteps"], self.tensor_kwargs_fp32
        ).reshape(-1)
        update_scale = float(self.config.hdmap_lane_update_interval)
        loss = global_loss + update_scale * torch.mean(time_weights * lane_per_instance)
        return loss, {
            "edm_loss": loss,
            "per_instance_loss": output_batch["per_instance_loss"] + update_scale * lane_per_instance,
            "hdmap_lane_rf_loss": lane.rf.mean().detach(),
            "hdmap_lane_edge_loss": lane.edge.mean().detach(),
            "hdmap_lane_mass": lane.mass.mean().detach(),
            "hdmap_lane_active_count": lane.active.sum().detach(),
            "hdmap_lane_update_scale": loss.new_tensor(update_scale),
        }

    def compute_rectified_flow_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        condition: ControlVideo2WorldCondition,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        global_loss, global_per_instance, diagnostics = super().compute_rectified_flow_loss(
            prediction,
            target,
            timesteps,
            condition,
        )
        if condition.control_input_lidar_mask is not None:
            lidar_mask = resize_video_mask_to_latent(
                condition.control_input_lidar_mask,
                prediction,
                temporal_compression_factor=self.tokenizer.temporal_compression_factor,
            ).to(device=prediction.device)
            time_weights = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32).reshape(-1)
            diagnostics = {
                **diagnostics,
                "lidar_bottom_rf_loss": torch.mean(
                    time_weights * masked_latent_mse(prediction, target, lidar_mask)
                ).detach(),
                "lidar_top_rf_loss": torch.mean(
                    time_weights * masked_latent_mse(prediction, target, 1.0 - lidar_mask)
                ).detach(),
                "lidar_bottom_fraction": lidar_mask.float().mean().detach(),
            }
        generic_lane_enabled = self.config.hdmap_lane_rf_weight > 0 or self.config.hdmap_lane_edge_weight > 0
        if generic_lane_enabled and self.config.hdmap_lane_seg_only:
            if getattr(self, "_hdmap_lane_update_active", True):
                self._seg_only_lane_target = target
            generic_lane_enabled = False
        if self.config.prg_regional_flow_weight == 0 and not generic_lane_enabled:
            return global_loss, global_per_instance, diagnostics
        generic_lane = None
        if generic_lane_enabled:
            if condition.hdmap_lane_probs is None or condition.hdmap_lane_clean_latent is None:
                raise RuntimeError("Generic HD-map lane loss requires lane probabilities and clean latent")
            lane_valid = build_hdmap_lane_valid_mask(
                condition.hdmap_lane_probs,
                padding_mask=condition.padding_mask,
                conditional_frame_mask=condition.condition_video_input_mask_B_C_T_H_W,
            )
            sigmas = self.rectified_flow.get_sigmas(timesteps.reshape(-1), self.tensor_kwargs_fp32)
            generic_lane = hdmap_lane_fidelity_losses(
                prediction,
                target,
                condition.hdmap_lane_clean_latent,
                condition.hdmap_lane_probs,
                sigmas,
                valid_mask=lane_valid,
                charbonnier_epsilon=self.config.hdmap_lane_charbonnier_epsilon,
                min_region_mass=self.config.hdmap_lane_min_mass,
                epsilon=self.config.hdmap_lane_epsilon,
                compute_edge=self.config.hdmap_lane_edge_weight > 0,
            )
        if self.config.prg_regional_flow_weight == 0:
            combined_per_instance = global_per_instance
            if generic_lane is not None:
                combined_per_instance = (
                    combined_per_instance
                    + self.config.hdmap_lane_rf_weight * generic_lane.rf
                    + self.config.hdmap_lane_edge_weight * generic_lane.edge
                )
            time_weights = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32).reshape(-1)
            loss = torch.mean(time_weights * combined_per_instance)
            diagnostics = {
                **diagnostics,
                "hdmap_lane_rf_loss": generic_lane.rf.mean().detach(),
                "hdmap_lane_edge_loss": generic_lane.edge.mean().detach(),
                "hdmap_lane_mass": generic_lane.mass.mean().detach(),
                "hdmap_lane_active_count": generic_lane.active.sum().detach(),
            }
            return loss, combined_per_instance, diagnostics
        route_probs = condition.degradation_route_probs
        if route_probs is None:
            raise RuntimeError("PRG regional flow loss requires degradation route probabilities")

        valid_token_mask = build_prg_valid_token_mask(
            route_probs,
            padding_mask=condition.padding_mask,
            conditional_frame_mask=condition.condition_video_input_mask_B_C_T_H_W,
        )
        regional = class_balanced_regional_flow_loss(
            prediction,
            target,
            route_probs,
            valid_token_mask=valid_token_mask,
            class_weights=self.config.prg_class_flow_weights,
            min_region_mass=self.config.prg_min_region_mass,
            epsilon=self.config.prg_region_epsilon,
        )
        combined_per_instance = (
            self.config.prg_base_flow_weight * global_per_instance
            + self.config.prg_regional_flow_weight * regional.per_instance_loss
            + (
                self.config.hdmap_lane_rf_weight * generic_lane.rf
                + self.config.hdmap_lane_edge_weight * generic_lane.edge
                if generic_lane is not None
                else 0.0
            )
        )
        time_weights = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32).reshape(-1)
        loss = torch.mean(time_weights * combined_per_instance)

        active = regional.class_active.float()
        class_loss_mean = (regional.per_class_loss * active).sum(dim=0) / active.sum(dim=0).clamp_min(1.0)
        diagnostics = {
            **diagnostics,
            "prg_global_flow_loss": global_loss.detach(),
            "prg_regional_flow_loss": regional.per_instance_loss.mean().detach(),
            "prg_flow_preserve": class_loss_mean[0].detach(),
            "prg_flow_repair": class_loss_mean[1].detach(),
            "prg_flow_generate": class_loss_mean[2].detach(),
            "prg_region_mass_preserve": regional.region_mass[:, 0].mean().detach(),
            "prg_region_mass_repair": regional.region_mass[:, 1].mean().detach(),
            "prg_region_mass_generate": regional.region_mass[:, 2].mean().detach(),
            "hdmap_lane_rf_loss": generic_lane.rf.mean().detach()
            if generic_lane is not None
            else prediction.new_zeros(()),
            "hdmap_lane_edge_loss": generic_lane.edge.mean().detach()
            if generic_lane is not None
            else prediction.new_zeros(()),
            "hdmap_lane_mass": generic_lane.mass.mean().detach()
            if generic_lane is not None
            else prediction.new_zeros(()),
            "hdmap_lane_active_count": generic_lane.active.sum().detach()
            if generic_lane is not None
            else prediction.new_zeros(()),
        }
        return loss, combined_per_instance, diagnostics

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, ControlVideo2WorldCondition]:
        # Get base data and condition
        if self.input_data_key in data_batch and data_batch[self.input_data_key].shape[2] == 1:
            data_batch[self.input_image_key] = data_batch[self.input_data_key].squeeze(2)
            assert data_batch[self.input_image_key].dtype == torch.uint8, "Image data is not in uint8 format."
            data_batch[self.input_image_key] = data_batch[self.input_image_key].to(**self.tensor_kwargs) / 127.5 - 1.0
            del data_batch[self.input_data_key]
        hdmap_lane_mask = None
        if self.config.hdmap_lane_rf_weight > 0 or self.config.hdmap_lane_edge_weight > 0:
            if "control_input_seg" not in data_batch:
                raise KeyError("HD-map lane-weighted regional RF requires control_input_seg")
            hdmap_lane_mask = extract_hdmap_lane_mask(
                data_batch["control_input_seg"],
                color_threshold=self.config.hdmap_lane_color_threshold,
                dilation=self.config.hdmap_lane_dilation,
            )
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        if self.config.hdmap_lane_rf_weight > 0 or self.config.hdmap_lane_edge_weight > 0:
            if hdmap_lane_mask is None:
                raise RuntimeError("Generic HD-map lane supervision requires an extracted lane mask")
            lane_probs = downsample_hdmap_lane_mask_to_latent(
                hdmap_lane_mask,
                latent_shape=tuple(latent_state.shape[-3:]),
                temporal_compression_factor=self.tokenizer.temporal_compression_factor,
            ).to(device=latent_state.device, dtype=torch.float32)
            condition = condition.set_hdmap_lane_condition(lane_probs, latent_state)
        # Add control conditioning
        latent_control_input = []
        control_weight = data_batch.get("control_weight", [1.0] * len(self.hint_keys))
        if len(control_weight) == 1:
            control_weight = control_weight * len(self.hint_keys)
        if self.config.use_prg_teacher_degradation_query:
            if self.hint_keys != ["control_input_vis", "control_input_seg", "control_input_prg"]:
                raise RuntimeError("PRG teacher requires hint_keys='vis_seg_prg' in that order")
            # VIS and trained HD-map SEG are dense controls. Privileged PRG-SEG
            # supplies typed query semantics only and has no dense residual.
            control_weight = [1.0, 1.0, 0.0]
        control_weight_maps = [None] * len(self.hint_keys)  # spatio-temporal control weight
        for hi, hint_key in enumerate(self.hint_keys):
            control_input = getattr(condition, hint_key, None)
            control_input_mask = getattr(condition, hint_key + "_mask", None)
            latent_control_input += self.get_control_latent(latent_state, control_input, control_input_mask)
            use_control_mask = self.config.apply_control_masks_to_residuals and (
                self.config.use_control_masks_during_training or not torch.is_grad_enabled()
            )
            if use_control_mask and not self.net.vace_has_mask:
                if control_input is None:  # set control weight to 0 if no control input
                    if len(control_weight) == len(self.hint_keys):
                        control_weight[hi] = 0.0
                    else:
                        control_weight.insert(hi, 0.0)
                if (
                    control_input_mask is not None and (control_input_mask != 1).any()
                ):  # use control weight to implement masking operation
                    assert control_input_mask.shape[1] == 1, (
                        f"control_input_mask.shape[1] != 1: {control_input_mask.shape[1]}"
                    )
                    control_weight_maps[hi] = control_input_mask * control_weight[hi]
        # If any control mask exists, use spatio-temporal control weight instead of scalar control weight.
        if any(c is not None for c in control_weight_maps):
            for hi in range(len(self.hint_keys)):
                if control_weight_maps[hi] is None:  # convert scalar control weight to spatio-temporal control weight
                    control_weight_maps[hi] = control_weight[hi] * torch.ones_like(
                        next(c for c in control_weight_maps if c is not None)
                    )
            control_weight_maps = torch.stack(control_weight_maps)
            # resize spatio-temporal control weight to match latent_state shape
            control_weight = self.resize_control_weight(control_weight_maps, latent_state)

        # assert num_modalities > 0, "No control input found"
        latent_control_input = torch.cat(latent_control_input, dim=1)
        condition = condition.set_control_condition(
            latent_control_input=latent_control_input,
            control_weight=control_weight,
        )
        use_internal_teacher = self.config.use_prg_teacher_degradation_query
        use_dense_gate = self.config.use_prg_dense_gate or self.config.use_prg_adaptive_dense_gate
        if "degradation_queries" in data_batch or (
            (use_internal_teacher or use_dense_gate) and "prg_label" in data_batch
        ):
            valid_mask = None
            if condition.padding_mask is not None:
                valid_mask = 1.0 - condition.padding_mask.clamp(0.0, 1.0)
            route_probs = self.build_degradation_route_probs(
                data_batch["prg_label"],
                latent_state,
                valid_mask=valid_mask,
            )
            query_active = data_batch.get("degradation_query_active")
            if query_active is None and use_internal_teacher:
                class_presence = route_probs.sum(dim=(2, 3, 4)) > 0
                query_counts = tuple(self.net.routed_query_counts)
                query_active = torch.cat(
                    [
                        class_presence[:, class_index : class_index + 1].expand(-1, count)
                        for class_index, count in enumerate(query_counts[:3])
                    ]
                    + [
                        torch.ones(
                            route_probs.shape[0],
                            query_counts[3],
                            device=route_probs.device,
                            dtype=torch.bool,
                        )
                    ],
                    dim=1,
                )
                disabled_classes = set(data_batch.get("disabled_prg_query_classes", ()))
                class_ids = {"preserve": 0, "repair": 1, "generate": 2}
                starts = (0, query_counts[0], query_counts[0] + query_counts[1])
                for class_name in disabled_classes:
                    if class_name not in class_ids:
                        raise ValueError(f"Unknown disabled PRG query class: {class_name}")
                    class_index = class_ids[class_name]
                    query_active[:, starts[class_index] : starts[class_index] + query_counts[class_index]] = False
                if self.config.prg_disable_global_query:
                    query_active = disable_global_prg_queries(query_active, query_counts)
                self._prg_active_gradient_classes = tuple(
                    class_index
                    for class_index, (start, count) in enumerate(
                        zip((0, query_counts[0], query_counts[0] + query_counts[1]), query_counts[:3], strict=True)
                    )
                    if query_active[:, start : start + count].any()
                )
            condition = condition.set_degradation_query_condition(
                degradation_queries=data_batch.get("degradation_queries"),
                query_scale=data_batch.get("degradation_query_scale", 1.0),
                route_probs=route_probs,
                query_active=query_active,
                query_intervention=data_batch.get("prg_query_intervention", "none"),
            )

        return raw_state, latent_state, condition

    def build_degradation_route_probs(
        self,
        prg_labels: Tensor,
        latent_state: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Map video-frame PRG labels to the DiT token grid."""
        return build_degradation_route_probs(
            prg_labels,
            latent_shape=tuple(latent_state.shape),
            temporal_compression_factor=self.tokenizer.temporal_compression_factor,
            patch_temporal=self.net.patch_temporal,
            patch_spatial=self.net.patch_spatial,
            dtype=latent_state.dtype,
            device=latent_state.device,
            valid_mask=valid_mask,
        )

    def resize_control_weight(self, control_context_scale: Tensor, latent_state: Tensor) -> Tensor:
        return resize_control_weight_maps(
            control_context_scale,
            latent_state,
            temporal_compression_factor=self.tokenizer.temporal_compression_factor,
            patch_spatial=self.net.patch_spatial,
            normalize=self.config.normalize_control_weight_maps,
        )

    def get_control_latent(self, latent_state: Tensor, control_input: Tensor, control_input_mask: Tensor) -> Tensor:
        latent_control_input = []
        if control_input is not None and not (control_input == -1).all():
            if self.net.vace_has_mask:
                if control_input_mask is None or (control_input_mask == 0).all():
                    control_input_mask = torch.ones_like(control_input[:, :1])
                assert control_input_mask.shape[1] == 1, (
                    f"control_input_mask.shape[1] != 1: {control_input_mask.shape[1]}"
                )
                fg = (control_input + 1) / 2 * control_input_mask * 2 - 1
                latent_control_input.append(self.encode(fg).contiguous().to(**self.tensor_kwargs))

                # reshape 8x8 spatial patch to channel dimension
                ph = pw = self.tokenizer.spatial_compression_factor
                mask = rearrange(control_input_mask, "b c t (h ph) (w pw) -> b (c ph pw) t h w", ph=ph, pw=pw)
                if mask.shape[2] > 1:
                    # interpolate to t frames
                    t = self.config.state_t
                    assert control_input_mask.shape[2] == 4 * (t - 1) + 1, (
                        f"control_input_mask.shape[2] != 4 * (t - 1) + 1: {control_input_mask.shape[2]} != {4 * (t - 1) + 1}"
                    )
                    H, W = mask.shape[-2:]
                    mask = [
                        mask[:, :, :1],
                        nn.functional.interpolate(mask[:, :, 1:], size=(t - 1, H, W), mode="nearest-exact"),
                    ]
                    mask = torch.cat(mask, dim=2)
                latent_control_input.append(mask.contiguous().to(**self.tensor_kwargs))
            else:
                latent_control_input.append(self.encode(control_input).contiguous().to(**self.tensor_kwargs))
        else:
            if self.net.vace_has_mask:
                ch = latent_state.shape[1] + self.tokenizer.spatial_compression_factor**2
                zero_latent_state = (
                    torch.zeros_like(latent_state[:, :1]).repeat(1, ch, 1, 1, 1).to(**self.tensor_kwargs)
                )
            else:
                zero_latent_state = torch.zeros_like(latent_state).to(**self.tensor_kwargs)
            latent_control_input.append(zero_latent_state)
        return latent_control_input

    def denoise(self, noise: torch.Tensor, xt_B_C_T_H_W: torch.Tensor, timesteps_B_T: torch.Tensor, condition):
        """
        Override denoise method for control branch support in rectified flow.
        """
        # Handle control conditioning
        if hasattr(condition, "latent_control_input"):
            # The control conditioning is already set in the condition object
            pass

        # Call parent's denoise method
        return super().denoise(noise, xt_B_C_T_H_W, timesteps_B_T, condition)

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `velocity_fn` based on the provided data batch and guidance factor for rectified flow.
        """
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
            log.info(
                f"num_conditional_frames: {num_conditional_frames} is set by data_batch[NUM_CONDITIONAL_FRAMES_KEY]"
            )
        else:
            num_conditional_frames = 0

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, control_condition = self.get_data_and_condition(data_batch)

        # Set video condition
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
        )

        # Set control condition
        latent_control_input = control_condition.latent_control_input
        control_weight = control_condition.control_context_scale
        condition = condition.set_control_condition(
            latent_control_input=latent_control_input, control_weight=control_weight
        )
        uncondition = uncondition.set_control_condition(
            latent_control_input=latent_control_input, control_weight=control_weight
        )
        if control_condition.degradation_queries is not None or control_condition.degradation_route_probs is not None:
            condition = condition.set_degradation_query_condition(
                degradation_queries=control_condition.degradation_queries,
                query_scale=control_condition.degradation_query_scale,
                route_probs=control_condition.degradation_route_probs,
                query_active=control_condition.degradation_query_active,
                query_intervention=control_condition.prg_query_intervention,
            )
            uncondition = uncondition.set_degradation_query_condition(
                degradation_queries=control_condition.degradation_queries,
                query_scale=control_condition.degradation_query_scale,
                route_probs=control_condition.degradation_route_probs,
                query_active=control_condition.degradation_query_active,
                query_intervention=control_condition.prg_query_intervention,
            )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        world_size = get_world_size()
        rank = get_rank()

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            noise_x = noise_x.to(**self.tensor_kwargs)
            """
            Use CFG parallel with 2 independent CP groups, each performing one denoising step.
            It allows for better scaling, as we get an additional 2x scaling increase. For example if standard
            inference of Cosmos-Transfer2 model hits a scaling wall at 32 GPUs with 8.4s inference time and using 64 GPUs
            provides no additional benefit, we can enable CFG-parallelism to scale the model to 64 GPUs with 4.2s inference time.
            """
            if getattr(self.net, "cfg_parallel", False):
                second_cp_start_rank = world_size // 2

                if rank < second_cp_start_rank:
                    cond_v = self.denoise(noise, noise_x, timestep, condition)
                else:
                    uncond_v = self.denoise(noise, noise_x, timestep, uncondition)

                rec_tensor = torch.empty_like(cond_v if rank < second_cp_start_rank else uncond_v)
                if rank < second_cp_start_rank:
                    torch.distributed.isend(cond_v, dst=second_cp_start_rank + rank)
                    res = torch.distributed.irecv(rec_tensor, src=second_cp_start_rank + rank)
                    res.wait()
                    uncond_v = rec_tensor

                else:
                    torch.distributed.irecv(rec_tensor, src=rank - second_cp_start_rank)
                    res = torch.distributed.isend(uncond_v, dst=rank - second_cp_start_rank)
                    res.wait()
                    cond_v = rec_tensor
            else:
                # Standard path without CFG parallelism
                cond_v = self.denoise(noise, noise_x, timestep, condition)
                uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn

    def prepare_generation_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> Dict:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )
        latents = noise
        x0_clean = kwargs.get("x0_clean")

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(
            num_steps,
            device=self.tensor_kwargs["device"],
            shift=shift,
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )

        timesteps = self.sample_scheduler.timesteps
        sigmas = self.sample_scheduler.sigmas
        if x0_clean is not None and kwargs.get("sigma_max") is not None:
            sigma_max_value = float(kwargs["sigma_max"])
            sigma_max_normalized = sigma_max_value / (1.0 + sigma_max_value)
            sigma_max_normalized = max(0.0, min(sigma_max_normalized, 1.0))
            x0_clean = x0_clean.to(device=self.tensor_kwargs["device"], dtype=noise.dtype).contiguous()
            latents = x0_clean * (1.0 - sigma_max_normalized) + noise * sigma_max_normalized

            if len(timesteps) > 0:
                schedule_sigmas = sigmas[:-1]
                start_index = int(torch.argmin(torch.abs(schedule_sigmas - sigma_max_normalized)).item())
                timesteps = timesteps[start_index:]
                sigmas = sigmas[start_index:]
        velocity_fn = self.get_velocity_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)
        x0_spatial_condition = data_batch.get("x0_spatial_condition", None)

        use_spatial_split = False
        if self.net.is_context_parallel_enabled:
            cp_size = len(torch.distributed.get_process_group_ranks(self.get_context_parallel_group()))
            n_views = noise.shape[2] // self.get_num_video_latent_frames()
            state_t = noise.shape[2] // n_views
            use_spatial_split = cp_size > state_t or state_t % cp_size != 0
            after_split_shape = None
            if use_spatial_split:
                after_split_shape = find_split(noise.shape, cp_size, view_factor=n_views)
                after_split_shape = torch.Size([after_split_shape[0] * n_views, *after_split_shape[1:]])
                noise = rearrange(noise, "b c t h w -> b c (t h w)")
                latents = rearrange(latents, "b c t h w -> b c (t h w)")
            noise = broadcast_split_tensor(tensor=noise, seq_dim=2, process_group=self.get_context_parallel_group())
            latents = broadcast_split_tensor(tensor=latents, seq_dim=2, process_group=self.get_context_parallel_group())
            if x0_spatial_condition is not None:
                x0 = x0_spatial_condition["x0"]
                x_sigma_mask = x0_spatial_condition["x_sigma_mask"]
                x0 = broadcast_split_tensor(x0, seq_dim=2, process_group=self.get_context_parallel_group())
                x_sigma_mask = broadcast_split_tensor(
                    x_sigma_mask, seq_dim=2, process_group=self.get_context_parallel_group()
                )
                x0_spatial_condition = {
                    **x0_spatial_condition,
                    "x0": x0,
                    "x_sigma_mask": x_sigma_mask,
                }
            if use_spatial_split:
                noise = rearrange(noise, "b c (t h w) -> b c t h w", t=after_split_shape[0], h=after_split_shape[1])
                latents = rearrange(latents, "b c (t h w) -> b c t h w", t=after_split_shape[0], h=after_split_shape[1])
                if x0_spatial_condition is not None:
                    x0_spatial_condition["x0"] = rearrange(
                        x0_spatial_condition["x0"],
                        "b c (t h w) -> b c t h w",
                        t=after_split_shape[0],
                        h=after_split_shape[1],
                    )
                    x0_spatial_condition["x_sigma_mask"] = rearrange(
                        x0_spatial_condition["x_sigma_mask"],
                        "b c (t h w) -> b c t h w",
                        t=after_split_shape[0],
                        h=after_split_shape[1],
                    )

        return {
            "latents": noise,
            "initial_latents": latents,
            "timesteps": timesteps,
            "sigmas": sigmas,
            "velocity_fn": velocity_fn,
            "x0_spatial_condition": x0_spatial_condition,
            "state_shape": state_shape,
            "seed_generator": seed_g,
            "use_spatial_split": use_spatial_split,
        }

    @torch.no_grad()
    def generate_samples_from_prepared(self, prepared: Dict) -> torch.Tensor:
        noise = prepared["latents"]
        latents = prepared.get("initial_latents", noise)
        timesteps = prepared["timesteps"]
        sigmas = prepared["sigmas"]
        velocity_fn = prepared["velocity_fn"]
        x0_spatial_condition = prepared.get("x0_spatial_condition")
        state_shape = prepared["state_shape"]
        seed_g = prepared["seed_generator"]
        use_spatial_split = prepared["use_spatial_split"]

        if x0_spatial_condition is not None:
            x0 = x0_spatial_condition["x0"]
            x_sigma_mask = x0_spatial_condition["x_sigma_mask"]
            step_threshold = x0_spatial_condition["step_threshold"]

        if INTERNAL:
            timesteps_iter = timesteps
        else:
            timesteps_iter = tqdm.tqdm(timesteps, desc="Generating samples", total=len(timesteps))

        for num_step, t in enumerate(timesteps_iter):
            latent_model_input = latents
            timestep = torch.stack([t])
            if x0_spatial_condition is not None and num_step <= step_threshold:
                sigma = sigmas[num_step]
                cur_x0 = x0 * (1 - sigma) + noise * sigma
                latent_model_input = cur_x0 * x_sigma_mask + latent_model_input * (1 - x_sigma_mask)
                if get_rank() == 0:
                    log.debug(f"denoising with guided generation at step {num_step}")

            velocity_pred = velocity_fn(noise, latent_model_input, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents = temp_x0.squeeze(0)

        if self.net.is_context_parallel_enabled:
            if use_spatial_split:
                latents = rearrange(latents, "b c t h w -> b c (t h w)")
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())
            if use_spatial_split:
                latents = rearrange(latents, "b c (t h w) -> b c t h w", t=state_shape[1], h=state_shape[2])

        return latents

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
        """
        prepared = self.prepare_generation_from_batch(
            data_batch,
            guidance=guidance,
            seed=seed,
            state_shape=state_shape,
            n_sample=n_sample,
            is_negative_prompt=is_negative_prompt,
            num_steps=num_steps,
            shift=shift,
            **kwargs,
        )
        return self.generate_samples_from_prepared(prepared)

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str | None = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        super()._normalize_video_databatch_inplace(data_batch, input_key)

        # Handle control_input if it exists
        for key in data_batch.keys():
            if key.startswith("control_input_") and data_batch[key] is not None:
                hint_key = key
                # Normalize control_input if not already normalized
                if data_batch[hint_key].dtype == torch.uint8:
                    data_batch[hint_key] = data_batch[hint_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                elif data_batch[hint_key].dtype == torch.bool:
                    data_batch[hint_key] = data_batch[hint_key].to(**self.tensor_kwargs)

                if data_batch[hint_key].dim() == 5 and data_batch[hint_key].shape[2] > 1:
                    expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
                    original_length = data_batch[hint_key].shape[2]
                    assert original_length == expected_length, (
                        "Input control_input length doesn't match expected length specified by state_t."
                    )

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        super()._augment_image_dim_inplace(data_batch, input_key)
        # Handle control_input if it exists
        for key in data_batch.keys():
            if key.startswith("control_input_") and data_batch[key] is not None and data_batch[key].dim() == 4:
                data_batch[key] = rearrange(data_batch[key], "b c h w -> b c 1 h w").contiguous()

    def copy_weights_to_control_branch(self) -> None:
        """
        VACE has the skip design of control blocks: control block i output modulates base block 2i
        In ControlNet training beginning, we copy base model weights to control branch. There are two strategies:
        1. copy base model's i-th block weight to control net's i-th block (more intuitive, the control blocks is a trainable
         copy of the first N layers of the base model)
        2. copy base model's 2i-th block weight to control net's i-th block (follow the correspondence of skip connection, \
           but the block-to-block connection in the control branch is weird.)
        Here we adopt the first strategy.
        """
        if self.is_new_training:
            control_blocks = (
                self.net.control_blocks if self.net.num_control_branches == 1 else self.net.control_blocks_0
            )
            if self.copy_weight_strategy == "first_n":
                # copy base model's i-th block weight to control net's i-th block
                control_to_base_layer_maping = {i: i for i in range(len(control_blocks))}
                assert len(control_to_base_layer_maping) == len(control_blocks)
            elif self.copy_weight_strategy == "spaced_n":
                # copy base model's 2i-th block weight to control net's i-th block
                control_to_base_layer_maping = {v: k for k, v in self.net.control_layers_mapping.items()}
                assert len(control_to_base_layer_maping) == len(control_blocks)
            else:
                raise ValueError("Other copy weight strategy doesn't seem to make sense.")

            # 1. First copy weights from base model to control net
            for control_layer_idx, base_layer_idx in control_to_base_layer_maping.items():
                log.info(
                    f"======Copying base model's {base_layer_idx}-th block weight to control net's {control_layer_idx}-th block"
                )

                if self.net.num_control_branches > 1:
                    for nc in range(self.net.num_control_branches):
                        missing_keys, unexpected_keys = getattr(self.net, f"control_blocks_{nc}")[
                            control_layer_idx
                        ].load_state_dict(self.net.blocks[base_layer_idx].state_dict(), strict=False)
                else:
                    missing_keys, unexpected_keys = self.net.control_blocks[control_layer_idx].load_state_dict(
                        self.net.blocks[base_layer_idx].state_dict(), strict=False
                    )
                assert len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
                assert set(missing_keys).issubset(
                    {
                        "before_proj.weight",
                        "before_proj.bias",
                        "after_proj.weight",
                        "after_proj.bias",
                        "_checkpoint_wrapped_module.before_proj.weight",
                        "_checkpoint_wrapped_module.before_proj.bias",
                        "_checkpoint_wrapped_module.after_proj.weight",
                        "_checkpoint_wrapped_module.after_proj.bias",
                    }
                ), f"missing_keys: {missing_keys}"

            if self.net.separate_embedders:
                self.net.t_embedder_for_control_branch.load_state_dict(self.net.t_embedder.state_dict(), strict=True)
                self.net.t_embedding_norm_for_control_branch.load_state_dict(
                    self.net.t_embedding_norm.state_dict(), strict=True
                )
                self.net.x_embedder_for_control_branch.load_state_dict(self.net.x_embedder.state_dict(), strict=True)

            self.is_new_training = False

    def freeze_base_model(self):
        log.info("\nFreezing base model\n")
        # 1. freeze everything
        for param in self.net.parameters():
            param.requires_grad = False

        # 2. unfreeze control-specific parameters: the blocks and patch embedding
        if self.net.num_control_branches > 1:
            for nc in range(self.net.num_control_branches):
                for param in getattr(self.net, f"control_blocks_{nc}").parameters():
                    param.requires_grad = True
            if hasattr(self.net, "after_proj"):
                for param in self.net.after_proj.parameters():
                    param.requires_grad = True
        else:
            for block in self.net.control_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        for param in self.net.control_embedder.parameters():
            param.requires_grad = True

        if self.net.separate_embedders:
            for param in self.net.t_embedder_for_control_branch.parameters():
                param.requires_grad = True
            for param in self.net.t_embedding_norm_for_control_branch.parameters():
                param.requires_grad = True
            for param in self.net.x_embedder_for_control_branch.parameters():
                param.requires_grad = True

        if self.net.use_input_hint_block:
            for param in self.net.input_hint_block.parameters():
                param.requires_grad = True

        # 3. unfreeze reference image weights if we use reference image control
        if self.config.use_reference_image and self.config.train_reference_image_modules:
            if hasattr(self.net, "img_context_proj"):
                for param in self.net.img_context_proj.parameters():
                    param.requires_grad = True
                log.info("✓ Unfroze img_context_proj")

            # 3.1 Unfreeze reference image weights in each ControlAwareDiTBlock
            if hasattr(self.net, "blocks"):
                for i, block in enumerate(self.net.blocks):
                    # Access the actual block inside CheckpointWrapper
                    actual_block = block._checkpoint_wrapped_module
                    cross_attn = actual_block.cross_attn

                    # Unfreeze k_img, v_img, k_img_norm
                    for param_key in ["k_img", "v_img", "k_img_norm", "q_img", "q_img_norm"]:
                        if hasattr(cross_attn, param_key):
                            for param in getattr(cross_attn, param_key).parameters():
                                param.requires_grad = True

                    log.info(f"✓ Unfroze reference image weights in ControlAwareDiTBlock {i}")

            # 3.2 Unfreeze reference image weights in each ControlEncoderDiTBlock
            if hasattr(self.net, "control_blocks"):
                for i, block in enumerate(self.net.control_blocks):
                    # Access the actual block inside CheckpointWrapper
                    actual_block = block._checkpoint_wrapped_module
                    cross_attn = actual_block.cross_attn

                    # Unfreeze k_img, v_img, k_img_norm
                    for param_key in ["k_img", "v_img", "k_img_norm", "q_img", "q_img_norm"]:
                        if hasattr(cross_attn, param_key):
                            for param in getattr(cross_attn, param_key).parameters():
                                param.requires_grad = True

                    log.info(f"✓ Unfroze reference image weights in ControlEncoderDiTBlock {i}")
        elif self.config.use_reference_image:
            log.info("Kept reference image projection and cross-attention modules frozen")

    def set_up_model(self):
        super().set_up_model()
        self.freeze_base_model()
        self.load_base_model()
        self.copy_weights_to_control_branch()

    def load_multi_branch_checkpoints(self, checkpoint_paths: list[str]):
        """
        Load control blocks from multiple checkpoint paths into control_blocks_0, control_blocks_1, etc.

        Args:
            checkpoint_paths (list[str]): List of checkpoint paths containing control blocks
        """
        if not checkpoint_paths:
            log.warning("No checkpoint paths provided for control branches")
            return

        # Use the same credentials as base model if available
        credential_path = "credentials/s3_checkpoint.secret"
        if hasattr(self.config, "base_load_from") and self.config.base_load_from is not None:
            credential_path = self.config.base_load_from.credentials

        load_planner = DefaultLoadPlanner(allow_partial_load=False)
        _model_wrapper = ModelWrapper(self)
        _state_dict = _model_wrapper.state_dict()

        # Filter out _extra_state entries to avoid metadata mismatch
        checkpoint_state_dict = {k: v for k, v in _state_dict.items() if "_extra_state" not in k}
        # Replace control_blocks_{nc} with control_blocks in the state dict
        for k in list(checkpoint_state_dict.keys()):
            for nc in range(self.net.num_control_branches):
                if f"control_blocks_{nc}" in k:
                    new_key = k.replace(f"control_blocks_{nc}", "control_blocks")
                    checkpoint_state_dict[new_key] = checkpoint_state_dict.pop(k)
                elif f"control_embedder.{nc}" in k:
                    new_key = k.replace(f"control_embedder.{nc}", "control_embedder")
                    checkpoint_state_dict[new_key] = checkpoint_state_dict.pop(k)

        for nc, checkpoint_path in enumerate(checkpoint_paths):
            if checkpoint_path is None:
                log.warning(f"No checkpoint path provided for control branch {nc}")
                continue

            if not INTERNAL:
                from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import download_checkpoint

                checkpoint_path = download_checkpoint(checkpoint_path)

            checkpoint_format = "pt" if checkpoint_path.endswith(".pt") else "dcp"
            # Handle checkpoint path with or without "model" suffix
            cur_key_ckpt_full_path = (
                checkpoint_path
                if checkpoint_path.endswith("model") or checkpoint_format == "pt"
                else os.path.join(checkpoint_path, "model")
            )
            log.critical(f"Start loading checkpoint for control branch {nc} from {checkpoint_path}")

            if "s3://" in checkpoint_path:
                storage_reader = S3StorageReader(
                    credential_path=credential_path,
                    path=cur_key_ckpt_full_path,
                )
            else:
                storage_reader = FileSystemReader(cur_key_ckpt_full_path)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if checkpoint_format == "dcp":  # load dcp checkpoint
                dcp.load(
                    checkpoint_state_dict,
                    storage_reader=storage_reader,
                    planner=load_planner,
                )
            else:
                # load pytorch checkpoint appending all keys to checkpoint_to_model_keys
                checkpoint_state_dict = torch.load(checkpoint_path)

            # Create mapping from checkpoint keys to model keys
            # Checkpoint has "control_blocks" but we want to load into "control_blocks_{nc}"
            checkpoint_to_model_keys = {}
            for k, v in checkpoint_state_dict.items():
                if "control_blocks." in k:
                    # Replace "control_blocks" with "control_blocks_{nc}" in the key
                    new_key = k.replace("control_blocks", f"control_blocks_{nc}")
                    checkpoint_to_model_keys[new_key] = v
                elif "control_embedder" in k:
                    new_key = k.replace("control_embedder", f"control_embedder.{nc}")
                    checkpoint_to_model_keys[new_key] = v

            assert checkpoint_to_model_keys, f"No control branch keys found in checkpoint for branch {nc}"

            model_state = nn.Module.state_dict(self, keep_vars=True)
            canonical_to_target = {
                key.replace("._checkpoint_wrapped_module", ""): key for key in model_state if "_extra_state" not in key
            }
            resolved_checkpoint_state = {}
            skipped_keys = []
            for key, source in checkpoint_to_model_keys.items():
                if "_extra_state" in key:
                    continue
                target_key = canonical_to_target.get(key.replace("._checkpoint_wrapped_module", ""))
                if target_key is None:
                    skipped_keys.append(key)
                    continue
                if tuple(model_state[target_key].shape) != tuple(source.shape):
                    raise ValueError(
                        f"Control branch {nc} shape mismatch for {target_key}: "
                        f"model={tuple(model_state[target_key].shape)}, checkpoint={tuple(source.shape)}"
                    )
                resolved_checkpoint_state[target_key] = source
            if not resolved_checkpoint_state:
                raise RuntimeError(f"No compatible control tensors found for branch {nc}")
            if skipped_keys:
                log.warning(
                    f"Skipped {len(skipped_keys)} optional control tensors absent from branch {nc}: {skipped_keys[:5]}"
                )
            with torch.no_grad():
                for key, source in resolved_checkpoint_state.items():
                    target = model_state[key]
                    if isinstance(target, DTensor):
                        source = distribute_tensor(
                            source.to(device=target.to_local().device),
                            device_mesh=target.device_mesh,
                            placements=target.placements,
                        )
                    else:
                        source = source.to(device=target.device)
                    target.copy_(source.to(dtype=target.dtype))
            log.info(f"Loaded {len(resolved_checkpoint_state)} tensors into control branch {nc}")
            log.info(f"Done loading the control branch {nc} checkpoint.")

    def load_base_model(self, load_ema_to_reg: bool = False) -> None:
        config = self.config
        if config.base_load_from is not None:
            checkpoint_path = config.base_load_from["load_path"]
        else:
            checkpoint_path = None
        if checkpoint_path is not None:
            load_planner = DefaultLoadPlanner(allow_partial_load=True)
            if config.base_load_from.get("credentials", None):
                cur_key_ckpt_full_path = os.path.join("s3://", checkpoint_path, "model")
                if INTERNAL:
                    storage_reader = S3StorageReader(
                        credential_path=config.base_load_from.credentials,
                        path=cur_key_ckpt_full_path,
                    )
                else:
                    from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path

                    checkpoint_path = get_checkpoint_path(cur_key_ckpt_full_path)
            else:
                storage_reader = FileSystemReader(checkpoint_path)

            log.critical(f"Start loading checkpoint for base model from {checkpoint_path}")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            checkpoint_format = "pt" if checkpoint_path.endswith(".pt") else "dcp"

            _model_wrapper = ModelWrapper(
                self, load_ema_to_reg=load_ema_to_reg if checkpoint_format == "dcp" else False
            )
            _state_dict = _model_wrapper.state_dict()

            # Filter out _extra_state entries to avoid metadata mismatch
            filtered_state_dict = {k: v for k, v in _state_dict.items() if "_extra_state" not in k}

            # Copy EMA weights to regular weights
            all_keys = list(filtered_state_dict.keys())
            # log.info(f"All keys: {all_keys}")
            for k in all_keys:
                if k.startswith("net.") and k.replace("net.", "net_ema.") in filtered_state_dict:
                    filtered_state_dict[k] = filtered_state_dict[k.replace("net.", "net_ema.")]
            self.load_state_dict(easy_io.load(checkpoint_path), strict=False)
        log.info("Done loading the base model checkpoint.")

    def get_x_from_clean(
        self,
        in_clean_img: torch.Tensor,
        sigma_max: float | None,
        seed: int = 1,
    ) -> Tensor:
        """
        in_clean_img (torch.Tensor): input clean image for image-to-image/video-to-video by adding noise then denoising
        sigma_max (float): maximum sigma applied to in_clean_image for image-to-image/video-to-video
        """
        if in_clean_img is None:
            return None
        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)
        noise = torch.randn(*in_clean_img.shape, **self.tensor_kwargs, generator=generator)
        if sigma_max is None:
            sigma_max = self.sde.sigma_max
        x_sigma_max = in_clean_img + noise * sigma_max
        return x_sigma_max
