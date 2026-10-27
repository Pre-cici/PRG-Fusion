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

"""Training runtime execution for YAML-driven train runs."""

from __future__ import annotations

import os

from loguru import logger as logging

from cosmos_transfer2.configs import TrainConfigFile
from cosmos_transfer2.runtime.resolve_config import ResolvedTrainConfig, resolve_train_config
from cosmos_transfer2.runtime.save_config import save_final_config
from cosmos_transfer2.runtime.validate_config import validate_train_config
from cosmos_transfer2.utils import log_reproducible_setup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _log_train_summary(resolved: ResolvedTrainConfig) -> None:
    config = resolved.final_config
    dataset_dir = getattr(config.dataloader_train.dataset, "dataset_dir", "<unknown>")
    batch_size = getattr(config.dataloader_train, "batch_size", "<unknown>")
    num_workers = getattr(config.dataloader_train, "num_workers", "<unknown>")
    hint_keys = getattr(config.model.config, "hint_keys", "<unknown>")
    resolution = getattr(config.model.config, "resolution", "<unknown>")
    text_encoder = getattr(config.model.config, "text_encoder_class", "<unknown>")
    run_label = "/".join(part for part in (config.job.group, config.job.name) if part)
    logging.info(
        "Train setup | "
        f"experiment={resolved.args.setup.experiment} | "
        f"run={run_label or '<unnamed>'} | "
        f"output={resolved.output_dir}"
    )
    logging.info(f"Train model | controls={hint_keys} | resolution={resolution} | text_encoder={text_encoder}")
    logging.info(
        "Train data | "
        f"dataset={dataset_dir} | batch_size={batch_size} | workers={num_workers} | max_iter={config.trainer.max_iter}"
    )
    logging.info(
        "Train runtime | "
        f"checkpoint_load={config.checkpoint.load_path} | save_every={config.checkpoint.save_iter} | "
        f"wandb={config.job.wandb_mode}"
    )


@logging.catch(reraise=True)
def launch_train(resolved: ResolvedTrainConfig) -> None:
    from cosmos_transfer2._src.imaginaire.flags import INTERNAL
    from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
    from cosmos_transfer2._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp

    config = resolved.final_config
    config.validate()
    config.freeze()  # type: ignore[arg-type]
    _log_train_summary(resolved)
    trainer = config.trainer.type(config)
    log_reproducible_setup(config, config_path=resolved.args.config_path, overrides=resolved.overrides)

    load_path = config.checkpoint.load_path
    if load_path:
        logging.info(f"Model load | checkpoint={load_path}")
    if not INTERNAL:
        from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import download_checkpoint

        if load_path:
            load_path = download_checkpoint(load_path)
    if load_path:
        logging.info(f"Model ready | checkpoint_local={load_path}")
    if isinstance(load_path, str) and load_path.endswith(".pt"):
        logging.info("Model init | loading consolidated checkpoint weights")
        model = create_model_from_consolidated_checkpoint_with_fsdp(config)
        logging.info("Model ready | weights loaded")
    else:
        if load_path:
            logging.info("Model init | checkpoint will be loaded by checkpointer")
        model = instantiate(config.model)

    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val)
    trainer.train(model, dataloader_train, dataloader_val)


def run_training(config: TrainConfigFile) -> None:
    from cosmos_oss.init import init_output_dir, is_rank0

    from cosmos_transfer2._src.imaginaire.config import pretty_print_overrides

    resolved = resolve_train_config(config)
    validate_train_config(resolved)
    config_path = resolved.output_dir / "final_config.yaml"

    if is_rank0():
        init_output_dir(resolved.output_dir, profile=config.setup.profile)
        config_path = save_final_config(resolved.final_config, resolved.output_dir)

    if config.setup.dryrun:
        logging.info(
            "Config:\n"
            + resolved.final_config.pretty_print(use_color=True)
            + "\n"
            + pretty_print_overrides(resolved.overrides, use_color=True)
        )
        os.makedirs(resolved.output_dir, exist_ok=True)
        print(str(config_path))
        return

    launch_train(resolved)
