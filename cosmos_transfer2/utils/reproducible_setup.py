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

"""Structured reproducible setup logging for train runtime."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import wandb
from omegaconf import OmegaConf

from cosmos_transfer2._src.imaginaire.utils import distributed, log
from cosmos_transfer2._src.imaginaire.utils.cluster_env import get_cluster_env
from cosmos_transfer2._src.imaginaire.utils.easy_io import easy_io
from cosmos_transfer2._src.imaginaire.utils.env_parsers.cred_env_parser import CRED_ENVS

if TYPE_CHECKING:
    from cosmos_transfer2._src.imaginaire.config import Config

S3_READY = False


def log_reproducible_setup(
    config: "Config",
    *,
    config_path: Path | None,
    overrides: list[str],
) -> None:
    """Save launch metadata without relying on argparse namespaces."""
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    time_tensor = torch.ByteTensor(bytearray(run_timestamp, "utf-8")).cuda()
    distributed.broadcast(time_tensor, 0)
    run_timestamp = time_tensor.cpu().numpy().tobytes().decode("utf-8")

    global S3_READY
    if os.path.exists(config.checkpoint.save_to_object_store.credentials) or CRED_ENVS.APP_ENV in [
        "prod",
        "dev",
        "stg",
    ]:
        easy_io.set_s3_backend(
            backend_args={
                "backend": "s3",
                "path_mapping": {
                    "s3://timestamps_rundir/": f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/job_runs/{run_timestamp}/",
                    "s3://rundir/": f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/",
                },
                "s3_credential_path": config.checkpoint.save_to_object_store.credentials,
            }
        )
        S3_READY = True
    else:
        log.warning("S3 credentials not found. Skipping easy_io S3 setup.")

    log.info(f"Run path | remote={config.job.path}")
    job_info = get_cluster_env()
    if distributed.get_rank() == 0:
        job_local_path = config.job.path_local
        log.info(f"Run path | local={job_local_path}")
        os.makedirs(config.job.path_local, exist_ok=True)
        launch_info = {
            "cmd": " ".join(sys.argv),
            "args_cfg_path": None if config_path is None else str(config_path),
            "args_override": overrides,
        }

        job_info["job_local_path"] = str(job_local_path)
        job_info["s3"] = f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/"
        if os.path.exists("git_commit.txt"):
            with open("git_commit.txt", "r") as f:
                job_info["commit_id"] = f.read().strip()
                log.info(f"Source | commit={job_info['commit_id']}")
        if os.path.exists("git_branch.txt"):
            with open("git_branch.txt", "r") as f:
                job_info["git_branch"] = f.read().strip()
                log.info(f"Source | branch={job_info['git_branch']}")

        with open(f"{job_local_path}/job_env.yaml", "w") as f:
            OmegaConf.save(job_info, f)
        with open(f"{job_local_path}/launch_info.yaml", "w") as f:
            OmegaConf.save(launch_info, f)
        if wandb.run:
            wandb.run.config.update({f"JOB_INFO/{k}": v for k, v in job_info.items()}, allow_val_change=True)

        if config.upload_reproducible_setup:
            if S3_READY:
                log.info(
                    "Run metadata upload | "
                    f"destination=s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/job_runs/{run_timestamp}/"
                )

                config_pkl_save_fp = f"{config.job.path_local}/config.pkl"
                easy_io.copyfile_from_local(
                    config_pkl_save_fp, f"s3://timestamps_rundir/{config_pkl_save_fp.split('/')[-1]}"
                )
                config_yaml_save_fp = config_pkl_save_fp.replace(".pkl", ".yaml")
                easy_io.copyfile_from_local(
                    config_yaml_save_fp, f"s3://timestamps_rundir/{config_yaml_save_fp.split('/')[-1]}"
                )
                easy_io.copyfile_from_local(f"{job_local_path}/job_env.yaml", "s3://timestamps_rundir/job_env.yaml")
                easy_io.copyfile_from_local(
                    f"{job_local_path}/launch_info.yaml", "s3://timestamps_rundir/launch_info.yaml"
                )
                if os.path.exists("codebase.zip"):
                    easy_io.copyfile_from_local("codebase.zip", "s3://timestamps_rundir/codebase.zip")
                if os.path.exists("code.tar.gz"):
                    easy_io.copyfile_from_local("code.tar.gz", "s3://timestamps_rundir/code.tar.gz")
                if os.path.exists("git_diff.txt"):
                    easy_io.copyfile_from_local("git_diff.txt", "s3://timestamps_rundir/git_diff.txt")
                if easy_io.exists("s3://rundir/job_history.yaml"):
                    job_history = easy_io.load("s3://rundir/job_history.yaml")
                else:
                    job_history = {}
                job_history[len(job_history)] = {
                    "timestamp": run_timestamp,
                    "reproduce_dir": f"s3://{config.checkpoint.save_to_object_store.bucket}/{config.job.path}/job_runs/{run_timestamp}/",
                    **launch_info,
                }
                print(job_history)
                easy_io.dump(job_history, "s3://rundir/job_history.yaml")
            else:
                log.warning("S3 credentials not found. Skipping upload of reproducible setup.")

    if config.upload_reproducible_setup and S3_READY:
        easy_io.dump(job_info, f"s3://timestamps_rundir/cluster_env/RANK_{distributed.get_rank():06d}.yaml")
