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

"""Persist resolved runtime configs and snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from cosmos_transfer2._src.imaginaire.config import Config


def save_final_config(config: "Config | Any", output_dir: Path, filename: str = "final_config.yaml") -> Path:
    from cosmos_transfer2._src.imaginaire.lazy_config.lazy import LazyConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / filename
    if hasattr(config, "job") and hasattr(config, "trainer"):
        LazyConfig.save_yaml(config, str(config_path))
        return config_path

    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)
    return config_path
