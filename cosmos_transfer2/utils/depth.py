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

"""Depth estimation utility using Video-Depth-Anything."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cosmos_transfer2._src.transfer2.auxiliary.depth_anything.video_depth_model import VideoDepthAnythingModel


def compute_depth_video(video_path: Path, output_path: Path | None = None, encoder: str = "vits") -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_depth.mp4")

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    video = np.array(frames)

    model = VideoDepthAnythingModel(encoder=encoder)
    model.setup()

    depth_maps = model.generate(video)

    depth_range = depth_maps.max() - depth_maps.min()
    if depth_range == 0:
        depth_normalized = np.zeros_like(depth_maps, dtype=np.uint8)
    else:
        depth_normalized = ((depth_maps - depth_maps.min()) / depth_range * 255).astype(np.uint8)

    T, H, W = depth_maps.shape
    depth_3ch = np.stack([depth_normalized] * 3, axis=-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    for frame in depth_3ch:
        out.write(frame)
    out.release()

    return output_path
