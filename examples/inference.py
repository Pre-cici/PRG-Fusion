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

import argparse
from pathlib import Path

from cosmos_transfer2.runtime.resolve_config import load_inference_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Singleview inference")
    parser.add_argument("--config", required=True, type=Path, help="YAML config file for inference.")
    parser.add_argument("--sample", action="append", help="Run only the named sample; may be repeated.")
    parser.add_argument("--output-dir", type=Path, help="Override setup.output_dir.")
    parser.add_argument(
        "--control-branch-checkpoint-path",
        type=Path,
        help="Override setup.control_branch_checkpoint_path.",
    )
    parser.add_argument("--vis-control-weight", type=float, help="Override every sample's VIS control weight.")
    parser.add_argument("--lidar-control-weight", type=float, help="Override every sample's LiDAR control weight.")
    parser.add_argument(
        "--degradation-query-scale",
        type=float,
        help="Override every sample's PRG gate strength; zero is the strict no-gate baseline.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_inference_config(args.config.resolve())
    if args.sample:
        requested = set(args.sample)
        available = {str(sample.get("name")) for sample in config.inference}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Unknown inference samples {missing}; available={sorted(available)}")
        config.inference = [sample for sample in config.inference if sample.get("name") in requested]
    if args.output_dir is not None:
        config.setup["output_dir"] = str(args.output_dir.resolve())
    if args.control_branch_checkpoint_path is not None:
        config.setup["control_branch_checkpoint_path"] = str(args.control_branch_checkpoint_path.resolve())
    for control_name, control_weight in (
        ("vis", args.vis_control_weight),
        ("lidar", args.lidar_control_weight),
    ):
        if control_weight is None:
            continue
        if not 0.0 <= control_weight <= 1.0:
            raise ValueError(f"{control_name} control weight must be in [0, 1], got {control_weight}")
        for sample in config.inference:
            if control_name not in sample:
                raise ValueError(f"Sample {sample.get('name')!r} has no {control_name} control")
            sample[control_name]["control_weight"] = control_weight
    if args.degradation_query_scale is not None:
        if args.degradation_query_scale < 0.0:
            raise ValueError(f"Degradation query scale must be nonnegative, got {args.degradation_query_scale}")
        for sample in config.inference:
            sample["degradation_query_scale"] = args.degradation_query_scale
    from cosmos_oss.init import cleanup_environment, init_environment

    from cosmos_transfer2.runtime.inference_runner import run_inference

    init_environment()
    run_inference(config)
    cleanup_environment()


if __name__ == "__main__":
    main()
