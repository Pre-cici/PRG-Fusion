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

"""Segmentation utility using SAM2 + GroundingDINO."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def compute_segmentation_video(
    video_path: Path,
    prompt: str,
    output_path: Path | None = None,
    box_threshold: float = 0.3,
    binarize_video: bool = False,
) -> Path:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_seg.mp4")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    from cosmos_transfer2._src.transfer2.auxiliary.sam2.sam2_model import VideoSegmentationModel

    model = VideoSegmentationModel()
    _patch_box_threshold(model, box_threshold)
    try:
        model(
            input_video=str(video_path),
            output_video=str(output_path),
            prompt=prompt,
            binarize_video=binarize_video,
        )
    except IndexError:
        _write_empty_segmentation_video(video_path, output_path)

    return output_path


def _patch_box_threshold(model, threshold):
    original = model.get_boxes_from_text

    def patched(image_path, text_prompt):
        result = original(image_path, text_prompt)
        boxes = result["boxes"]
        scores = result["scores"]
        if len(boxes) == 0:
            return result
        keep = scores >= threshold
        result["boxes"] = boxes[keep]
        result["scores"] = scores[keep]
        if result.get("labels") is not None:
            result["labels"] = np.array(result["labels"])[keep]
        return result

    model.get_boxes_from_text = patched


def _write_empty_segmentation_video(video_path: Path, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=True)
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        out.write(black_frame)

    cap.release()
    out.release()
