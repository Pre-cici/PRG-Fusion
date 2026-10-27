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

import copy
import math
import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
import wandb
from einops import rearrange, repeat
from megatron.core import parallel_state
from torch.utils.data._utils.collate import default_collate

from cosmos_transfer2._src.imaginaire.callbacks.every_n import EveryN
from cosmos_transfer2._src.imaginaire.model import ImaginaireModel
from cosmos_transfer2._src.imaginaire.utils import distributed, log, misc
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_transfer2._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import CTRL_HINT_KEYS
from cosmos_transfer2._src.transfer2.datasets.local_datasets.singleview_dataset import SingleViewTransferDataset


# use first two rank to generate some images for visualization
def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


class EveryNDrawSample(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        fix_batch_fp: Optional[str] = None,
        n_x0_level: int = 4,
        n_viz_sample: int = 3,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        guidance: List[float] = [3.0, 7.0, 9.0, 13.0],
        is_x0: bool = True,
        is_sample: bool = True,
        save_s3: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        show_all_frames: bool = False,
        fps: int = 16,
        sample_dataset_dir: Optional[str] = None,
        sample_dataset_index: int = 0,
        sample_num_frames: Optional[int] = None,
        sample_video_size: Optional[List[int]] = None,
        sample_resolution: Optional[str] = None,
        sample_hint_key: Optional[str] = None,
        generation_types: Optional[List[str]] = None,
    ):
        super().__init__(every_n, step_size)
        self.fix_batch = fix_batch_fp
        self.n_x0_level = n_x0_level
        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.is_x0 = is_x0
        self.is_sample = is_sample
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.show_all_frames = show_all_frames
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.fps = fps
        self.sample_dataset_dir = sample_dataset_dir
        self.sample_dataset_index = sample_dataset_index
        self.sample_num_frames = sample_num_frames
        self.sample_video_size = tuple(sample_video_size) if sample_video_size is not None else None
        self.sample_resolution = sample_resolution
        self.sample_hint_key = sample_hint_key
        self.generation_types = generation_types or ["t2v", "i2v", "v2v"]

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        if self.fix_batch is not None:
            with misc.timer(f"loading fix_batch {self.fix_batch}"):
                self.fix_batch = misc.co(easy_io.load(self.fix_batch), "cpu")
        elif self.sample_dataset_dir is not None:
            with misc.timer(f"loading sample batch from {self.sample_dataset_dir}"):
                self.fix_batch = self._load_fixed_batch_from_dataset()

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

        if self.use_negative_prompt:
            negative_prompt_path = "s3://bucket/edify_video/v4/validation/item_dataset/negative_prompt/000000.pkl"
            self.negative_prompt_data = easy_io.load(negative_prompt_path)

    def _get_dataset_hint_key(self) -> str:
        if self.sample_hint_key is not None:
            return self.sample_hint_key

        dataset_config = getattr(self.config.dataloader_train, "dataset", None)
        hint_key = getattr(dataset_config, "hint_key", None)
        if hint_key is not None:
            return str(hint_key)

        control_input_type = getattr(dataset_config, "control_input_type", None)
        if control_input_type is not None:
            return f"control_input_{control_input_type}"

        model_hint_keys = getattr(self.config.model.config, "hint_keys", None)
        if isinstance(model_hint_keys, str):
            return f"control_input_{model_hint_keys}"

        raise ValueError("Unable to infer sample hint_key for EveryNDrawSample")

    def _load_fixed_batch_from_dataset(self) -> dict:
        dataset_dir = Path(self.sample_dataset_dir).expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Sample dataset directory does not exist: {dataset_dir}")

        dataset_config = getattr(self.config.dataloader_train, "dataset", None)
        num_frames = self.sample_num_frames or getattr(dataset_config, "num_frames", None)
        video_size = self.sample_video_size or getattr(dataset_config, "video_size", None)
        resolution = self.sample_resolution or getattr(dataset_config, "resolution", None)
        hint_key = self._get_dataset_hint_key()

        if num_frames is None or video_size is None or resolution is None:
            raise ValueError("Missing sample dataset settings: num_frames, video_size, and resolution are required")

        dataset = SingleViewTransferDataset(
            dataset_dir=str(dataset_dir),
            num_frames=int(num_frames),
            video_size=tuple(video_size),
            resolution=str(resolution),
            hint_key=hint_key,
            is_train=False,
        )
        if len(dataset) == 0:
            raise ValueError(f"Sample dataset is empty: {dataset_dir}")

        sample_index = self.sample_dataset_index % len(dataset)
        sample = dataset[sample_index]
        log.info(
            f"Loaded fixed sample batch from {dataset_dir} at index {sample_index} "
            f"for callback {self.__class__.__name__}"
        )
        return default_collate([sample])

    def _clone_batch(self, data):
        if isinstance(data, torch.Tensor):
            return data.clone()
        if isinstance(data, dict):
            return type(data)({key: self._clone_batch(value) for key, value in data.items()})
        if isinstance(data, list):
            return [self._clone_batch(value) for value in data]
        if isinstance(data, tuple):
            return tuple(self._clone_batch(value) for value in data)
        return copy.deepcopy(data)

    def _cast_floating_tensors(self, data, dtype):
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype) if torch.is_floating_point(data) else data
        if isinstance(data, dict):
            return type(data)({key: self._cast_floating_tensors(value, dtype) for key, value in data.items()})
        if isinstance(data, list):
            return [self._cast_floating_tensors(value, dtype) for value in data]
        if isinstance(data, tuple):
            return tuple(self._cast_floating_tensors(value, dtype) for value in data)
        return data

    def _prepare_fixed_batch(self, model) -> dict:
        data_batch = misc.to(self._clone_batch(self.fix_batch), device=model.tensor_kwargs["device"])
        text_encoder_config = getattr(model.config, "text_encoder_config", None)
        if text_encoder_config is not None and text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0],
                text_embeddings.shape[1],
                device=model.tensor_kwargs["device"],
            )
        return self._cast_floating_tensors(data_batch, model.tensor_kwargs["dtype"])

    @misc.timer("EveryNDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.fix_batch is not None:
            data_batch = self._prepare_fixed_batch(model)
        tag = "ema" if self.is_ema else "reg"

        log.debug("starting data and condition model", rank0_only=False)

        raw_data, x0, condition = model.get_data_and_condition(data_batch)
        _, condition, x0, _ = model.broadcast_split_for_model_parallelsim(None, condition, x0, None)

        log.debug("done data and condition model", rank0_only=False)
        batch_size = x0.shape[0]
        sigmas = np.exp(
            np.linspace(math.log(model.sde.sigma_min), math.log(model.sde.sigma_max), self.n_x0_level + 1)[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        random_noise = torch.randn(*x0.shape, generator=generator, **model.tensor_kwargs)
        _ones = torch.ones(batch_size, **model.tensor_kwargs)
        mse_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            log.debug(f"starting denoising {sigma}", rank0_only=False)
            sample = model.denoise(x_sigma, _ones * sigma, condition).x0
            log.debug(f"done denoising {sigma}", rank0_only=False)
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            mse_loss_list.append(mse_loss)

            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )
        # check if it has cotrol input
        if "hint_key" in data_batch:
            hint = data_batch[data_batch["hint_key"]]
            to_show.append(hint.float().cpu())
        else:
            for key in CTRL_HINT_KEYS.keys():
                if key in data_batch and data_batch[key] is not None:
                    hint = data_batch[key]
                    to_show.append(hint.float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_x0_Iter{iteration:09d}"

        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if is_tp_cp_pp_rank0():
            if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"s3://rundir/{self.name}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        log.debug("entering, every_n_impl", rank0_only=False)
        with context():
            log.debug("entering, ema", rank0_only=False)
            # we only use rank0 and rank to generate images and save
            # other rank run forward pass to make sure it works for FSDP
            log.debug("entering, fsdp", rank0_only=False)
            if self.is_x0:
                log.debug("entering, x0_pred", rank0_only=False)
                x0_img_fp, mse_loss, sigmas = self.x0_pred(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                log.debug("done, x0_pred", rank0_only=False)
                if self.save_s3 and self.rank == 0:
                    easy_io.dump(
                        {
                            "mse_loss": mse_loss.tolist(),
                            "sigmas": sigmas.tolist(),
                            "iteration": iteration,
                        },
                        f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                    )
            if self.is_sample:
                log.debug("entering, sample", rank0_only=False)
                sample_img_fp = self.sample(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                log.debug("done, sample", rank0_only=False)
            if self.fix_batch is not None:
                misc.to(self.fix_batch, "cpu")

            log.debug("waiting for all ranks to finish", rank0_only=False)
            dist.barrier()
        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }
            if self.is_x0:
                info[f"{self.name}/{tag}_x0"] = wandb.Image(x0_img_fp, caption=f"{sample_counter}")
                # convert mse_loss to a dict
                mse_loss = mse_loss.tolist()
                info.update({f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i] for i in range(len(mse_loss))})

            if self.is_sample:
                info[f"{self.name}/{tag}_sample"] = wandb.Image(sample_img_fp, caption=f"{sample_counter}")
            wandb.log(
                info,
                step=iteration,
            )
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        """
        Args:
            skip_save: to make sure FSDP can work, we run forward pass on all ranks even though we only save on rank 0 and 1
        """
        if self.fix_batch is not None:
            analysis_batch = self._prepare_fixed_batch(model)
        else:
            analysis_batch = data_batch

        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, condition = model.get_data_and_condition(analysis_batch)
        if self.use_negative_prompt:
            batch_size = x0.shape[0]
            if self.negative_prompt_data["t5_text_embeddings"].shape != analysis_batch["t5_text_embeddings"].shape:
                neg_t5_text_embeddings = misc.to(
                    repeat(
                        self.negative_prompt_data["t5_text_embeddings"],
                        "... -> b ...",
                        b=batch_size,
                    ),
                    **model.tensor_kwargs,
                )
            else:
                neg_t5_text_embeddings = misc.to(
                    self.negative_prompt_data["t5_text_embeddings"],
                    **model.tensor_kwargs,
                )

        num_conditional_frames_options = {
            "t2v": 0,
            "i2v": 1,
            "v2v": 2,
        }
        local_path_dict = {}

        for generation_type in self.generation_types:
            if generation_type not in num_conditional_frames_options:
                raise ValueError(
                    f"Unsupported generation type: {generation_type}. "
                    f"Expected one of {list(num_conditional_frames_options.keys())}"
                )
            num_conditional_frames = num_conditional_frames_options[generation_type]
            to_show = []
            for guidance in self.guidance:
                if self.fix_batch is not None:
                    sample_batch = self._prepare_fixed_batch(model)
                else:
                    sample_batch = data_batch
                sample_batch[NUM_CONDITIONAL_FRAMES_KEY] = num_conditional_frames
                if self.use_negative_prompt:
                    sample_batch["neg_t5_text_embeddings"] = neg_t5_text_embeddings
                    assert sample_batch["neg_t5_text_embeddings"].shape == sample_batch["t5_text_embeddings"].shape, (
                        f"{sample_batch['neg_t5_text_embeddings'].shape} != {sample_batch['t5_text_embeddings'].shape}"
                    )
                    sample_batch["neg_t5_text_mask"] = sample_batch["t5_text_mask"]
                sample = model.generate_samples_from_batch(
                    sample_batch,
                    guidance=guidance,
                    # make sure no mismatch and also works for cp
                    state_shape=x0.shape[1:],
                    n_sample=x0.shape[0],
                    num_steps=self.num_sampling_step,
                    is_negative_prompt=True if self.use_negative_prompt else False,
                )
                if hasattr(model, "decode"):
                    sample = model.decode(sample)
                to_show.append(sample.float().cpu())

            to_show.append(raw_data.float().cpu())

            # visualize input video
            if "hint_key" in analysis_batch:
                hint = analysis_batch[analysis_batch["hint_key"]]
                for idx in range(0, hint.size(1), 3):
                    x_rgb = hint[:, idx : idx + 3]
                    to_show.append(x_rgb.float().cpu())
            else:
                for key in CTRL_HINT_KEYS.keys():
                    if key in analysis_batch and analysis_batch[key] is not None:
                        hint = analysis_batch[key]
                        to_show.append(hint.float().cpu())

            base_fp_wo_ext = (
                f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}_{generation_type}"
            )

            batch_size = output_batch["x0"].shape[0]
            if is_tp_cp_pp_rank0():
                local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
                local_path_dict[generation_type] = local_path

            base_fp_wo_ext = (
                f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}_{generation_type}_single"
            )
            batch_size = output_batch["x0"].shape[0]
            if is_tp_cp_pp_rank0():
                _ = self.run_save(to_show[2:3], batch_size, base_fp_wo_ext)

        # to be determined: which one to return
        if len(local_path_dict) != 0:
            if "i2v" in local_path_dict:
                return local_path_dict["i2v"]
            return next(iter(local_path_dict.values()))

        return None

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = min(self.n_viz_sample, batch_size)

        # ! we only save first n_sample_to_save video!
        if self.save_s3 and self.data_parallel_id < self.n_sample_to_save:
            save_img_or_video(
                rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
                f"s3://rundir/{self.name}/{base_fp_wo_ext}",
                fps=self.fps,
            )

        file_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        local_path = f"{self.local_dir}/{file_base_fp}"

        if self.rank == 0 and wandb.run:
            if is_single_frame:  # image case
                to_show = rearrange(
                    to_show[:, :n_viz_sample],
                    "n b c t h w -> t c (n h) (b w)",
                )
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                # resize so that wandb can handle it
                torchvision.utils.save_image(resize_image(image_grid, 1024), local_path, nrow=1, scale_each=True)
            else:
                to_show = to_show[:, :n_viz_sample]  # [n, b, c, 3, h, w]
                if not self.show_all_frames:
                    # resize 3 frames frames so that we can display them on wandb
                    _T = to_show.shape[3]
                    three_frames_list = [0, _T // 2, _T - 1]
                    to_show = to_show[:, :, :, three_frames_list]
                    log_image_size = 1024
                else:
                    log_image_size = 512 * to_show.shape[3]
                to_show = rearrange(
                    to_show,
                    "n b c t h w -> 1 c (n h) (b t w)",
                )

                # resize so that wandb can handle it
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                torchvision.utils.save_image(
                    resize_image(image_grid, log_image_size), local_path, nrow=1, scale_each=True
                )

            return local_path
        return None
