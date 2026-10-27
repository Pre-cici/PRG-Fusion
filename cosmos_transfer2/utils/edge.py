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

"""Edge extraction utility using OpenCV Canny with configurable threshold presets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2

EdgePreset = Literal["very_low", "low", "medium", "high", "very_high"]

PRESET_CANNY_THRESHOLDS: dict[EdgePreset, tuple[int, int]] = {
    "very_low": (20, 50),
    "low": (50, 100),
    "medium": (100, 200),
    "high": (200, 300),
    "very_high": (300, 400),
}


def compute_edge_video(
    video_path: Path,
    output_path: Path | None = None,
    preset: EdgePreset = "medium",
    contrast: float = 1.5,
    brightness: int = 0,
) -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_edge.mp4")

    t_lower, t_upper = PRESET_CANNY_THRESHOLDS[preset]

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Could not open video: {video_path}"
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), isColor=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if contrast != 1.0 or brightness != 0:
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, t_lower, t_upper)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out.write(edges_bgr)

    cap.release()
    out.release()

    return output_path
