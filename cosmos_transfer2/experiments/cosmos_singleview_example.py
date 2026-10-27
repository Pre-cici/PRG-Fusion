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

"""
Post-training configurations for Cosmos-Transfer2 with local single-view datasets.

These configurations demonstrate how to post-train Transfer2 models on your own data.
They inherit from the base 2B control experiment and override dataset and checkpoint settings.

Example usage:
    torchrun --nproc_per_node=8 -m scripts.train \\
        --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \\
        -- experiment=transfer2_singleview_posttrain_edge_example \\
        dataloader_train.dataset.dataset_dir=/path/to/your/dataset
"""

import os

from hydra.core.config_store import ConfigStore

from cosmos_transfer2.configs.common import DEFAULT_BASE_EXPERIMENT, MODEL_CHECKPOINTS, ModelKey, ModelVariant

# Default model keys and checkpoints for each control type
DEFAULT_EDGE_MODEL_KEY = ModelKey(variant=ModelVariant.EDGE)
DEFAULT_DEPTH_MODEL_KEY = ModelKey(variant=ModelVariant.DEPTH)
DEFAULT_SEG_MODEL_KEY = ModelKey(variant=ModelVariant.SEG)
DEFAULT_VIS_MODEL_KEY = ModelKey(variant=ModelVariant.VIS)

EDGE_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_EDGE_MODEL_KEY]
DEPTH_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_DEPTH_MODEL_KEY]
SEG_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_SEG_MODEL_KEY]
VIS_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_VIS_MODEL_KEY]


# =============================================================================
# Post-training with Edge Control (2B Model)
# =============================================================================

transfer2_singleview_posttrain_edge_example = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_edge"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view",
        name="transfer2_singleview_posttrain_edge_example",
    ),
    checkpoint=dict(
        save_iter=1000,
        load_path=EDGE_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="edge",
            base_load_from=None,  # Disable base model loading (already loading from checkpoint.load_path)
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="edge",
        ),
    ),
    trainer=dict(
        max_iter=5000,
        straggler_detection=dict(enabled=False),  # Disable for local training
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=200),
            every_n_sample_ema=dict(save_s3=False, every_n=200),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[5000],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


# =============================================================================
# Post-training with Depth Control (2B Model)
# =============================================================================

transfer2_singleview_posttrain_depth_example = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_depth"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view",
        name="transfer2_singleview_posttrain_depth_example",
    ),
    checkpoint=dict(
        save_iter=1000,
        load_path=DEPTH_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="depth",
            base_load_from=None,  # Disable base model loading (already loading from checkpoint.load_path)
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="depth",
        ),
    ),
    trainer=dict(
        max_iter=5000,
        straggler_detection=dict(enabled=False),  # Disable for local training
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=200),
            every_n_sample_ema=dict(save_s3=False, every_n=200),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[5000],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


# =============================================================================
# Post-training with Segmentation Control (2B Model)
# =============================================================================

transfer2_singleview_posttrain_seg_example = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_seg"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view",
        name="transfer2_singleview_posttrain_seg_example",
    ),
    checkpoint=dict(
        save_iter=1000,
        load_path=SEG_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="seg",
            base_load_from=None,  # Disable base model loading (already loading from checkpoint.load_path)
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="segcolor",
        ),
    ),
    trainer=dict(
        max_iter=5000,
        straggler_detection=dict(enabled=False),  # Disable for local training
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=200),
            every_n_sample_ema=dict(save_s3=False, every_n=200),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[5000],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


# =============================================================================
# Post-training with Vis (Blur) Control (2B Model)
# =============================================================================

transfer2_singleview_posttrain_vis_example = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_vis"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view",
        name="transfer2_singleview_posttrain_vis_example",
    ),
    checkpoint=dict(
        save_iter=1000,
        load_path=VIS_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="vis",
            base_load_from=None,  # Disable base model loading (already loading from checkpoint.load_path)
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="vis",
        ),
    ),
    trainer=dict(
        max_iter=5000,
        straggler_detection=dict(enabled=False),  # Disable for local training
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=200),
            every_n_sample_ema=dict(save_s3=False, every_n=200),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[5000],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


transfer2_singleview_vis_hdmap_seg_branch = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_prg_teacher_vis"},
    ],
    job=dict(
        project="cosmos_transfer2_seg_branch",
        group="waymo_vis_hdmap_seg",
        name="transfer2_singleview_vis_hdmap_seg_branch",
    ),
    checkpoint=dict(
        save_iter=500,
        load_path=VIS_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="vis_seg",
            base_load_from=None,
            control_branch_checkpoint_paths=(VIS_CHECKPOINT.s3.uri, SEG_CHECKPOINT.s3.uri),
            trainable_control_branch_indices=(1,),
            use_prg_teacher_degradation_query=False,
            prg_regional_flow_weight=0.0,
            net=dict(
                num_control_branches=2,
                use_after_proj_for_multi_branch=False,
                use_spatial_degradation_query=False,
                use_vis_prg_teacher_query=False,
            ),
        ),
    ),
    dataloader_train=dict(dataset=dict(control_input_type="vis")),
    trainer=dict(
        max_iter=1000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=1_000_000),
            every_n_sample_ema=dict(save_s3=False, every_n=1_000_000),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(cycle_lengths=[1000]),
    model_parallel=dict(context_parallel_size=1),
)


transfer2_singleview_vis_lidar_branch = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_prg_teacher_vis"},
    ],
    job=dict(
        project="cosmos_transfer2_vis_lidar_v2",
        group="waymo_3cams_v2",
        name="transfer2_singleview_vis_lidar_branch",
    ),
    checkpoint=dict(
        save_iter=100,
        load_path=VIS_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="vis_lidar",
            base_load_from=None,
            control_branch_checkpoint_paths=(VIS_CHECKPOINT.s3.uri, EDGE_CHECKPOINT.s3.uri),
            trainable_control_branch_indices=(1,),
            use_control_masks_during_training=True,
            normalize_control_weight_maps=False,
            use_prg_teacher_degradation_query=False,
            prg_regional_flow_weight=0.0,
            hdmap_lane_rf_weight=0.0,
            hdmap_lane_edge_weight=0.0,
            net=dict(
                num_control_branches=2,
                use_after_proj_for_multi_branch=False,
                use_spatial_degradation_query=False,
                use_vis_prg_teacher_query=False,
                use_prg_dense_gate=False,
                use_prg_adaptive_dense_gate=False,
            ),
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="vis",
            require_hdmap=False,
            require_lidar=True,
            lidar_mask_top_rows=256,
            load_prg_labels=False,
        )
    ),
    trainer=dict(
        max_iter=1000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=1_000_000),
            every_n_sample_ema=dict(save_s3=False, every_n=1_000_000),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(cycle_lengths=[1000]),
    model_parallel=dict(context_parallel_size=1),
)


transfer2_singleview_lidar_image_context = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_prg_teacher_vis"},
        {"override /conditioner": "video_prediction_control_conditioner_image_context"},
    ],
    job=dict(
        project="cosmos_transfer2_vis_lidar_v4",
        group="waymo_3cams_v4",
        name="transfer2_singleview_lidar_image_context",
    ),
    checkpoint=dict(
        save_iter=500,
        load_path=EDGE_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="lidar",
            base_load_from=None,
            use_reference_image=True,
            use_control_masks_during_training=True,
            normalize_control_weight_maps=False,
            use_prg_teacher_degradation_query=False,
            prg_regional_flow_weight=0.0,
            hdmap_lane_rf_weight=0.0,
            hdmap_lane_edge_weight=0.0,
            net=dict(
                num_control_branches=1,
                extra_image_context_dim=1152,
                share_q_in_i2v_cross_attn=False,
                img_context_deep_proj=False,
                use_spatial_degradation_query=False,
                use_vis_prg_teacher_query=False,
                use_prg_dense_gate=False,
                use_prg_adaptive_dense_gate=False,
            ),
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="vis",
            require_hdmap=False,
            require_lidar=True,
            lidar_mask_top_rows=256,
            load_prg_labels=False,
            use_reference_image=True,
        )
    ),
    trainer=dict(
        max_iter=1000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=1_000_000),
            every_n_sample_ema=dict(save_s3=False, every_n=1_000_000),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(cycle_lengths=[1000]),
    model_parallel=dict(context_parallel_size=1),
)


transfer2_singleview_vis_lidar_full_canvas = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_prg_teacher_vis"},
    ],
    job=dict(
        project="cosmos_transfer2_vis_lidar_v4",
        group="waymo_3cams_v4",
        name="transfer2_singleview_vis_lidar_full_canvas",
    ),
    checkpoint=dict(
        save_iter=500,
        load_path=VIS_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="vis_lidar",
            base_load_from=None,
            use_reference_image=False,
            control_branch_checkpoint_paths=(VIS_CHECKPOINT.s3.uri, EDGE_CHECKPOINT.s3.uri),
            control_branch_overlay_checkpoint_path=None,
            control_branch_overlay_indices=(1,),
            control_branch_overlay_from_single_branch=True,
            trainable_control_branch_indices=(0,),
            use_control_masks_during_training=True,
            apply_control_masks_to_residuals=False,
            normalize_control_weight_maps=False,
            use_prg_teacher_degradation_query=False,
            prg_regional_flow_weight=0.0,
            hdmap_lane_rf_weight=0.0,
            hdmap_lane_edge_weight=0.0,
            net=dict(
                num_control_branches=2,
                use_after_proj_for_multi_branch=False,
                use_spatial_degradation_query=False,
                use_vis_prg_teacher_query=False,
                use_prg_dense_gate=False,
                use_prg_adaptive_dense_gate=False,
            ),
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="vis",
            require_hdmap=False,
            require_lidar=True,
            lidar_mask_top_rows=256,
            load_prg_labels=False,
            use_reference_image=False,
        )
    ),
    trainer=dict(
        max_iter=500,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=1_000_000),
            every_n_sample_ema=dict(save_s3=False, every_n=1_000_000),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(cycle_lengths=[500]),
    model_parallel=dict(context_parallel_size=1),
)


# =============================================================================
# Local 5090 smoke-test experiments
# =============================================================================

transfer2_singleview_posttrain_edge_local_5090_smoke = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_edge"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view_smoke",
        name="transfer2_singleview_posttrain_edge_local_5090_smoke",
    ),
    checkpoint=dict(
        save_iter=1,
        load_path=EDGE_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="edge",
            fsdp_shard_size=1,
            resolution="480",
            state_t=16,
            base_load_from=None,
            net=dict(
                num_blocks=4,
                vace_block_every_n=7,
                rope_enable_fps_modulation=False,
                rope_h_extrapolation_ratio=1.0,
                rope_w_extrapolation_ratio=1.0,
                rope_t_extrapolation_ratio=1.0,
                timestep_scale=0.001,
                sac_config=dict(mode="mm_only"),
                use_crossattn_projection=True,
                crossattn_proj_in_channels=100352,
                crossattn_emb_channels=1024,
                use_wan_fp32_strategy=True,
            ),
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="edge",
            num_frames=61,
            video_size=(480, 832),
            resolution="480",
        ),
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=None,
    ),
    trainer=dict(
        max_iter=10,
        logging_iter=1,
        validation_iter=1000,
        run_validation=False,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=1_000_000),
            every_n_sample_ema=dict(save_s3=False, every_n=1_000_000),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[10],
        warm_up_steps=[1],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)

transfer2_singleview_posttrain_depth_local_5090_smoke = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_depth"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view_smoke",
        name="transfer2_singleview_posttrain_depth_local_5090_smoke",
    ),
    checkpoint=dict(
        save_iter=1,
        load_path=DEPTH_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="depth",
            fsdp_shard_size=1,
            resolution="480",
            state_t=16,
            base_load_from=None,
            net=dict(
                num_blocks=4,
                vace_block_every_n=7,
                rope_enable_fps_modulation=False,
                rope_h_extrapolation_ratio=1.0,
                rope_w_extrapolation_ratio=1.0,
                rope_t_extrapolation_ratio=1.0,
                timestep_scale=0.001,
                sac_config=dict(mode="mm_only"),
                use_crossattn_projection=True,
                crossattn_proj_in_channels=100352,
                crossattn_emb_channels=1024,
                use_wan_fp32_strategy=True,
            ),
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="depth",
            num_frames=61,
            video_size=(480, 832),
            resolution="480",
        ),
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=None,
    ),
    trainer=dict(
        max_iter=10,
        logging_iter=1,
        validation_iter=1000,
        run_validation=False,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=1_000_000),
            every_n_sample_ema=dict(save_s3=False, every_n=1_000_000),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[10],
        warm_up_steps=[1],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


transfer2_singleview_posttrain_edge_local_5090_observe = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_edge"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view_observe",
        name="transfer2_singleview_posttrain_edge_local_5090_observe",
    ),
    checkpoint=dict(
        save_iter=10,
        load_path=EDGE_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="edge",
            fsdp_shard_size=1,
            resolution="480",
            state_t=16,
            base_load_from=None,
            net=dict(
                num_blocks=4,
                vace_block_every_n=7,
                rope_enable_fps_modulation=False,
                rope_h_extrapolation_ratio=1.0,
                rope_w_extrapolation_ratio=1.0,
                rope_t_extrapolation_ratio=1.0,
                timestep_scale=0.001,
                sac_config=dict(mode="mm_only"),
                use_crossattn_projection=True,
                crossattn_proj_in_channels=100352,
                crossattn_emb_channels=1024,
                use_wan_fp32_strategy=True,
            ),
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="edge",
            num_frames=61,
            video_size=(480, 832),
            resolution="480",
        ),
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=None,
    ),
    trainer=dict(
        max_iter=50,
        logging_iter=1,
        validation_iter=1000,
        run_validation=False,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False, every_n=10),
            iter_speed=dict(save_s3=False, every_n=1, hit_thres=3),
            device_monitor=dict(save_s3=False, every_n=1),
            every_n_sample_reg=dict(save_s3=False, every_n=1_000_000),
            every_n_sample_ema=dict(save_s3=False, every_n=1_000_000),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False, every_n=1),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[50],
        warm_up_steps=[2],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


transfer2_singleview_posttrain_edge_local_5090_train = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_edge"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view_train",
        name="transfer2_singleview_posttrain_edge_local_5090_train",
    ),
    checkpoint=dict(
        save_iter=50,
        load_path=EDGE_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="edge",
            fsdp_shard_size=1,
            resolution="480",
            state_t=16,
            base_load_from=None,
            net=dict(
                num_blocks=4,
                vace_block_every_n=7,
                rope_enable_fps_modulation=False,
                rope_h_extrapolation_ratio=1.0,
                rope_w_extrapolation_ratio=1.0,
                rope_t_extrapolation_ratio=1.0,
                timestep_scale=0.001,
                sac_config=dict(mode="mm_only"),
                use_crossattn_projection=True,
                crossattn_proj_in_channels=100352,
                crossattn_emb_channels=1024,
                use_wan_fp32_strategy=True,
            ),
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="edge",
            num_frames=61,
            video_size=(480, 832),
            resolution="480",
        ),
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        prefetch_factor=None,
    ),
    trainer=dict(
        max_iter=500,
        logging_iter=10,
        validation_iter=1000,
        run_validation=False,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False, every_n=10),
            iter_speed=dict(save_s3=False, every_n=10, hit_thres=3),
            device_monitor=dict(save_s3=False, every_n=10),
            every_n_sample_reg=dict(save_s3=False, every_n=1_000_000),
            every_n_sample_ema=dict(save_s3=False, every_n=1_000_000),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False, every_n=10),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[500],
        warm_up_steps=[10],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


# =============================================================================
# Register all experiments
# =============================================================================

cs = ConfigStore.instance()

for _item in [
    transfer2_singleview_posttrain_edge_example,
    transfer2_singleview_posttrain_depth_example,
    transfer2_singleview_posttrain_seg_example,
    transfer2_singleview_posttrain_vis_example,
    transfer2_singleview_vis_hdmap_seg_branch,
    transfer2_singleview_vis_lidar_branch,
    transfer2_singleview_lidar_image_context,
    transfer2_singleview_vis_lidar_full_canvas,
    transfer2_singleview_posttrain_edge_local_5090_smoke,
    transfer2_singleview_posttrain_depth_local_5090_smoke,
    transfer2_singleview_posttrain_edge_local_5090_observe,
    transfer2_singleview_posttrain_edge_local_5090_train,
]:
    _name: str = _item["job"]["name"]  # pyrefly: ignore
    cs.store(
        group="experiment",
        package="_global_",
        name=_name,
        node=_item,
    )
