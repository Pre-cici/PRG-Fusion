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
Single-view Transfer dataset for local video files with control inputs.

This dataset loader is designed for post-training Cosmos-Transfer2 models with local data.
It uses the full Transfer2 augmentor pipeline including:
- Randomized edge detection thresholds for training diversity
- Automatic resizing with aspect ratio preservation
- Reflection padding
- Text transforms for caption handling
- Control input generation (edge, depth, seg, blur, etc.)

Example usage:
    dataset = SingleViewTransferDataset(
        dataset_dir="datasets/example",
        num_frames=93,
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_edge",
        is_train=True,  # Enable augmentations for training
    )
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.datasets.augmentor_provider import get_video_augmentor_v2_with_control
from cosmos_transfer2._src.transfer2.datasets.prg import (
    PRG_REGIME_TO_ID,
    load_prg_label_frames,
    load_prg_manifest,
    prg_labels_to_rgb,
    resize_and_pad_prg_labels,
)
from cosmos_transfer2._src.transfer2.utils.input_handling import detect_aspect_ratio


# Mock URL object for augmentor compatibility
class MockUrlMeta:
    """Mock metadata object for WebDataset compatibility."""

    def __init__(self):
        self.opts = {}


class MockUrl:
    """Mock URL object that augmentors expect from WebDataset."""

    def __init__(self, url: str):
        self._url = url
        self.meta = MockUrlMeta()

    def __str__(self) -> str:
        return self._url

    def __repr__(self) -> str:
        return f"MockUrl({self._url})"


# Mappings between control types and corresponding sub-folder names in the data folder
CTRL_TYPE_INFO = {
    "keypoint": {"folder": "keypoint", "format": "pickle", "data_dict_key": "keypoint"},
    "depth": {"folder": "depth", "format": "mp4", "data_dict_key": "depth"},
    "seg": {"folder": "seg", "format": "mp4", "data_dict_key": "segmentation"},
    "edge": {"folder": "edge", "format": "mp4", "data_dict_key": "control_input_edge", "optional": True},
    "vis": {"folder": "vis", "format": "mp4", "data_dict_key": "vis", "optional": True},
}


class SingleViewTransferDataset(Dataset):
    """Dataset class for loading single-view video-to-video generation data with control inputs.

    This dataset is designed for post-training Cosmos-Transfer2 models with local video files.
    It supports various control modalities including depth, segmentation, edge, and blur.

    Dataset structure:
        dataset_dir/
        ├── videos/
        │   ├── video1.mp4
        │   └── video2.mp4
        ├── captions/
        │   ├── video1.json  ({"caption": "text description"})
        │   └── video2.json
        └── <control_type>/  (optional for depth/seg, computed on-the-fly for edge/vis)
            ├── video1.mp4  (for depth)
            └── video1.pickle  (for seg/keypoint)

    Args:
        dataset_dir: Base path to the dataset directory
        num_frames: Number of frames to load per sequence
        video_size: Target size (H, W) for video frames
        resolution: Resolution key for augmentor (e.g., "720", "1080")
        hint_key: Control input type (e.g., "control_input_edge", "control_input_depth")
        is_train: Whether this is for training (affects sampling)
        caption_type: Type of caption to load (default: "t2w_qwen2p5_7b")
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        resolution: str = "720",
        hint_key: str = "control_input_edge",
        is_train: bool = True,
        caption_type: str = "t2w_qwen2p5_7b",  # Use Qwen2.5-7B caption type
        **kwargs,  # Accept extra params for config compatibility (like MultiviewTransferDataset)
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.video_size = video_size
        self.resolution = resolution
        self.is_train = is_train
        self.caption_type = caption_type

        # Parse control type from hint_key
        self.hint_key = hint_key
        self.ctrl_type = hint_key.replace("control_input_", "")
        if self.ctrl_type not in CTRL_TYPE_INFO:
            raise ValueError(
                f"Unsupported control type: {self.ctrl_type}. Supported types: {list(CTRL_TYPE_INFO.keys())}"
            )
        self.ctrl_config = CTRL_TYPE_INFO[self.ctrl_type]

        # Set up directories
        video_dirs = [os.path.join(self.dataset_dir, name) for name in ("videos", "video")]
        video_dir = next((path for path in video_dirs if os.path.isdir(path)), None)
        if video_dir is None:
            raise FileNotFoundError(f"Neither videos/ nor video/ exists under {self.dataset_dir}")
        self.video_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")])

        # Support both "captions/" and "metas/" directories
        self.caption_dir = os.path.join(self.dataset_dir, "captions")
        if not os.path.exists(self.caption_dir):
            self.caption_dir = os.path.join(self.dataset_dir, "metas")

        # Note: We no longer load T5 embeddings - captions are encoded on-the-fly
        # by the model's text encoder (Qwen2.5-VL-7B / reason1p1_7B)
        self.num_failed_loads = 0
        self.bad_video_indices = set()  # Track videos that fail to load (too short, corrupted, etc.)

        # Use proper augmentor pipeline for training quality
        # This includes randomized edge detection, reflection padding, and text transforms
        # Pass embedding_type=None since we're handling T5 embeddings ourselves
        # (if embedding_type is set, the function returns early with only video_parsing)
        augmentor_config = get_video_augmentor_v2_with_control(
            resolution=resolution,
            caption_type=caption_type,
            embedding_type=None,  # We handle embeddings ourselves, get full augmentor pipeline
            control_input_type=self.ctrl_type,
            use_random=is_train,  # Enable random augmentations for training
        )

        # Filter out augmentors that don't apply to local datasets
        # The augmentor pipeline includes augmentors designed for S3/WebDataset that need to be skipped:
        # - video_parsing: Decodes video bytes from S3 → we already load tensors from local MP4 files
        # - depth_parsing: Decodes depth bytes from S3 key "depth_pervideo_video_depth_anything" → we load from local depth/ folder
        # - seg_parsing: Decodes seg bytes from S3 key "segmentation_sam2_color_video_v2" → we load from local seg/ folder
        # - merge_datadict: Merges multiple WebDataset shards → not needed for single local dataset
        # - text_transform: Loads pre-computed T5 embeddings → we pass raw captions for on-the-fly encoding
        skip_augmentors = ["video_parsing", "merge_datadict", "text_transform", "depth_parsing", "seg_parsing"]
        augmentor_config = {k: v for k, v in augmentor_config.items() if k not in skip_augmentors}

        log.info(f"Filtered augmentors: {list(augmentor_config.keys())}")

        # Instantiate augmentors
        self.augmentor = {k: instantiate(v) for k, v in augmentor_config.items()}

        # Double-check text_transform is not present
        if "text_transform" in self.augmentor:
            raise RuntimeError("text_transform should have been filtered out but is still present!")

        log.info(f"Initialized SingleViewTransferDataset with {len(self.video_paths)} videos")
        log.info(f"  Dataset dir: {self.dataset_dir}")
        log.info(f"  Control type: {self.ctrl_type}")
        log.info(f"  Resolution: {resolution}, Video size: {video_size}")
        log.info(f"  Required frames: {self.sequence_length}")

        # Quick validation: check for obviously bad videos (optional, can be slow for large datasets)
        # self._validate_videos()  # Uncomment to pre-filter bad videos at initialization

    def __str__(self) -> str:
        return f"SingleViewTransferDataset: {len(self.video_paths)} videos from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _validate_videos(self) -> None:
        """Validate all videos and pre-mark bad ones (too short, corrupted, etc.).

        This is optional and can be slow for large datasets, but helps identify
        problematic videos upfront. Call this in __init__ if you want pre-filtering.
        """
        log.info("Validating videos for minimum frame count...")
        bad_count = 0

        for idx, video_path in enumerate(self.video_paths):
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                del vr

                if total_frames < self.sequence_length:
                    self.bad_video_indices.add(idx)
                    bad_count += 1
                    log.debug(
                        f"Marking video {idx} as bad: {os.path.basename(video_path)} "
                        f"has only {total_frames} frames (need {self.sequence_length})"
                    )
            except Exception as e:
                self.bad_video_indices.add(idx)
                bad_count += 1
                log.debug(f"Marking video {idx} as bad: {os.path.basename(video_path)} - {e}")

        valid_count = len(self.video_paths) - bad_count
        log.info(
            f"Video validation complete: {valid_count} valid, {bad_count} bad "
            f"({bad_count / len(self.video_paths) * 100:.1f}% filtered)"
        )

    def _load_video(self, video_path: str, frame_ids: list[int] | None = None) -> tuple[np.ndarray, float, list[int]]:
        """Load video frames from file.

        Args:
            video_path: Path to video file
            frame_ids: Specific frame indices to load. If None, randomly samples frames.

        Returns:
            Tuple of (frames, fps, frame_ids)
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)

        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # Sample frames if not provided
        if frame_ids is None:
            max_start_idx = total_frames - self.sequence_length
            start_frame = np.random.randint(0, max_start_idx + 1) if self.is_train else 0
            frame_ids = list(range(start_frame, start_frame + self.sequence_length))

        # Load frames
        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # Reset video reader

        # Debug: Log frame loading
        log.info(
            f"Loaded video {os.path.basename(video_path)}: "
            f"total_frames={total_frames}, "
            f"requested={len(frame_ids)}, "
            f"loaded={frame_data.shape[0]}"
        )

        try:
            fps = vr.get_avg_fps()
        except Exception:
            fps = 24  # Default FPS

        del vr
        return frame_data, fps, frame_ids

    def _load_caption(self, video_name: str) -> str:
        """Load caption from JSON or text file.

        Args:
            video_name: Video name without extension

        Returns:
            Caption text
        """
        # Try JSON first (Transfer2 format)
        json_path = Path(self.caption_dir) / f"{video_name}.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    # Support various JSON formats
                    if isinstance(data, dict):
                        return data.get("caption", data.get("text", data.get("prompt", "")))
                    return str(data)
            except Exception as e:
                log.warning(f"Failed to load caption from {json_path}: {e}")

        # Fall back to text file (Transfer1 format)
        txt_path = Path(self.caption_dir) / f"{video_name}.txt"
        if txt_path.exists():
            try:
                return txt_path.read_text().strip()
            except Exception as e:
                log.warning(f"Failed to load caption from {txt_path}: {e}")

        log.debug(f"No caption found for {video_name}, using generic caption")
        return "a video"  # Generic fallback caption

    # Captions are now encoded on-the-fly by the model's text encoder (Qwen/reason1p1_7B)

    def _load_control_data(self, video_name: str, frame_ids: list[int]) -> dict[str, Any] | None:
        """Load control input data (depth, segmentation, etc.).

        For vis, returns None (computed on-the-fly by augmentor).
        For edge, prefers a precomputed edge video if present, otherwise falls back
        to on-the-fly Canny generation by the augmentor.
        For depth, loads video frames.
        For seg/keypoint, loads pickle data.

        Args:
            video_name: Video name without extension
            frame_ids: Frame indices to load

        Returns:
            Dictionary with control data or None if computed on-the-fly
        """
        if self.ctrl_config["folder"] is None:
            return None

        ctrl_folder = os.path.join(self.dataset_dir, self.ctrl_config["folder"])
        ctrl_format = self.ctrl_config["format"]
        ctrl_path = os.path.join(ctrl_folder, f"{video_name}.{ctrl_format}")

        if not os.path.exists(ctrl_path):
            if self.ctrl_config.get("optional", False):
                log.info(
                    f"Optional control input missing for {video_name}, using on-the-fly {self.ctrl_type}: {ctrl_path}"
                )
                return None
            raise FileNotFoundError(f"Control input file not found: {ctrl_path}")

        data_dict = {}

        try:
            if self.ctrl_type == "seg":
                # Load segmentation video (same format as depth)
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                if len(vr) < frame_ids[-1] + 1:
                    raise ValueError(f"Seg video has fewer frames than RGB video: {ctrl_path}")

                seg_frames = vr.get_batch(frame_ids).asnumpy()  # [T, H, W, C]
                seg_frames = seg_frames.astype(np.uint8)
                # Convert to tensor - augmentor will handle resizing to match video
                seg_t = torch.from_numpy(seg_frames).permute(0, 3, 1, 2)  # (T, C, H, W) uint8
                seg_video = seg_t.permute(1, 0, 2, 3)  # (C, T, H, W) uint8

                # Store with the key expected by AddControlInputSeg augmentor
                data_dict["segmentation"] = seg_video
                del vr

            elif self.ctrl_type == "keypoint":
                # Load keypoint pickle
                with open(ctrl_path, "rb") as f:
                    keypoint_data = pickle.load(f)
                data_dict["keypoint"] = keypoint_data

            elif self.ctrl_type == "depth":
                # Load depth video
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                if len(vr) < frame_ids[-1] + 1:
                    raise ValueError(f"Depth video has fewer frames than RGB video: {ctrl_path}")

                depth_frames = vr.get_batch(frame_ids).asnumpy()  # [T, H, W, C]
                depth_frames = depth_frames.astype(np.uint8)
                # Convert to tensor - augmentor will handle resizing to match video
                depth_t = torch.from_numpy(depth_frames).permute(0, 3, 1, 2)  # (T, C, H, W) uint8
                depth_video = depth_t.permute(1, 0, 2, 3)  # (C, T, H, W) uint8

                # Store with the key expected by AddControlInputDepth augmentor
                data_dict["depth"] = depth_video
                del vr

            elif self.ctrl_type == "edge":
                # Load precomputed edge video as an intermediate tensor.
                # The edge augmentor will align it to the final video size and
                # produce control_input_edge, matching the depth code path.
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                if len(vr) < frame_ids[-1] + 1:
                    raise ValueError(f"Edge video has fewer frames than RGB video: {ctrl_path}")

                edge_frames = vr.get_batch(frame_ids).asnumpy()  # [T, H, W, C]
                edge_frames = edge_frames.astype(np.uint8)
                edge_t = torch.from_numpy(edge_frames).permute(0, 3, 1, 2)  # (T, C, H, W) uint8
                edge_video = edge_t.permute(1, 0, 2, 3)  # (C, T, H, W) uint8

                data_dict["edge"] = edge_video
                del vr

            elif self.ctrl_type == "vis":
                vr = VideoReader(ctrl_path, ctx=cpu(0))
                if len(vr) < frame_ids[-1] + 1:
                    raise ValueError(f"Vis video has fewer frames than RGB video: {ctrl_path}")

                vis_frames = vr.get_batch(frame_ids).asnumpy()
                vis_frames = vis_frames.astype(np.uint8)
                vis_t = torch.from_numpy(vis_frames).permute(0, 3, 1, 2)
                vis_video = vis_t.permute(1, 0, 2, 3)

                data_dict["vis"] = vis_video
                del vr

        except Exception as e:
            log.warning(f"Failed to load control data from {ctrl_path}: {e}")
            return None

        return data_dict

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single training sample with full augmentation.

        Returns:
            Dictionary with:
                - video: RGB video tensor [C=3, T, H, W] dtype=uint8, resized and padded by augmentor
                - control_input_<type>: Control input tensor [C=3, T, H, W] dtype=uint8, same shape as video
                  (generated with randomized parameters if is_train=True)
                - fps: Video frame rate (float)
                - aspect_ratio: Aspect ratio string (e.g., "16:9")
                - image_size: Image dimensions [H, W, H, W] tensor (after augmentation)
                - padding_mask: Padding mask [1, H, W] tensor (marks valid vs. padded regions)
                - num_frames: Number of frames (int)
                - chunk_index: Chunk index (int, typically 0)
                - __url__: Dataset directory (str)
                - __key__: Video name (str)

        Note:
            - Augmentor applies randomized edge detection thresholds during training for diversity
            - Video dimensions may differ from input video_size due to augmentor's padding/resizing
            - Model expects uint8 format and performs normalization internally:
              uint8 [0, 255] → float32 [-1, 1]
        """
        max_retries = 10  # Try up to 10 different videos
        original_index = index

        for retry in range(max_retries):
            # Skip known bad videos
            if index in self.bad_video_indices:
                index = (index + 1) % len(self.video_paths)
                continue

            try:
                video_path = self.video_paths[index]
                video_name = os.path.basename(video_path).replace(".mp4", "")

                # Load video frames
                frames, fps, frame_ids = self._load_video(video_path)
                frames = frames.astype(np.uint8)

                # Convert to tensor - augmentor will handle resizing and padding
                # frames: numpy (T, H, W, C) uint8
                frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W) uint8
                # Permute to (C, T, H, W) format expected by augmentors
                video = frames_t.permute(1, 0, 2, 3)  # (C, T, H, W) uint8
                aspect_ratio = detect_aspect_ratio((video.shape[3], video.shape[2]))  # (W, H)

                # Build data dictionary
                data = {
                    "video": video,
                    "aspect_ratio": aspect_ratio,
                    "fps": fps,
                    "frame_start": frame_ids[0],
                    "frame_end": frame_ids[-1] + 1,
                    "num_frames": self.sequence_length,
                    "chunk_index": 0,
                    "frame_indices": frame_ids,
                    "n_orig_video_frames": len(frame_ids),
                }

                # Load caption
                caption = self._load_caption(video_name)
                data[self.caption_type] = caption

                # Create metadata structure for augmentor compatibility
                # The augmentor expects "metas" with window information
                # Map caption_type to the expected caption key in window_data
                if self.caption_type == "t2w_qwen2p5_7b":
                    caption_key_in_window = "qwen2p5_7b_caption"  # gitleaks:allow
                else:
                    caption_key_in_window = self.caption_type

                window_data = {
                    "start_frame": frame_ids[0],
                    "end_frame": frame_ids[-1] + 1,
                    caption_key_in_window: caption,
                }
                data["metas"] = {
                    "framerate": fps,
                    "nb_frames": len(frame_ids),
                    # Create a single window spanning the entire video segment
                    # Include both windows and t2w_windows for different caption types
                    "windows": [window_data],
                    "t2w_windows": [window_data],
                    "i2w_windows_later_frames": [window_data],
                }

                # Pass raw caption for on-the-fly encoding by model's text encoder
                # (Like multiview dataset - model will encode with Qwen/reason1 encoder)
                data["ai_caption"] = caption

                # Add URL and key for logging (used by augmentors and training)
                # Use MockUrl object for augmentor compatibility (augmentors expect __url__.meta.opts)
                data["__url__"] = MockUrl(str(self.dataset_dir))
                data["__key__"] = video_name

                # Load control input data (if pre-computed)
                ctrl_data = self._load_control_data(video_name, frame_ids)
                if ctrl_data is not None:
                    data.update(ctrl_data)

                # Apply augmentation pipeline
                # This includes: resizing, padding, text transform, and control input generation
                # The augmentor will handle edge detection with randomized thresholds for training
                for aug_name, aug_fn in self.augmentor.items():
                    result = aug_fn(data)
                    # Check if augmentor returned None (e.g., filtering)
                    if result is None:
                        raise ValueError(f"Augmentor {aug_name} filtered out the sample")
                    data = result

                # Convert MockUrl back to string for DataLoader collate compatibility
                # (PyTorch's collate function can't handle custom objects)
                if isinstance(data.get("__url__"), MockUrl):
                    data["__url__"] = str(data["__url__"])

                # Add final metadata (after augmentation)
                c, t, h, w = data["video"].shape
                if "image_size" not in data:
                    data["image_size"] = torch.tensor([h, w, h, w])
                if "padding_mask" not in data:
                    data["padding_mask"] = torch.zeros(1, h, w)  # Zero denotes valid content.

                # Validate output format after augmentation
                assert data["video"].dtype == torch.uint8, f"Video dtype is {data['video'].dtype}, expected uint8"
                assert data["video"].shape[0] == 3, f"Video should have 3 channels, got {data['video'].shape[0]}"
                assert data["video"].shape[1] == self.sequence_length, (
                    f"Video should have {self.sequence_length} frames, got {data['video'].shape[1]}"
                )

                # Check control input exists and has correct format
                ctrl_key = f"control_input_{self.ctrl_type}"
                assert ctrl_key in data, f"Control input key '{ctrl_key}' not found in data"
                assert data[ctrl_key].dtype == torch.uint8, (
                    f"Control input dtype is {data[ctrl_key].dtype}, expected uint8"
                )
                assert data[ctrl_key].shape == data["video"].shape, (
                    f"Control input shape {data[ctrl_key].shape} doesn't match video shape {data['video'].shape}"
                )

                log.debug(
                    f"Dataset sample ready: video={data['video'].shape} {data['video'].dtype}, "
                    f"{ctrl_key}={data[ctrl_key].shape} {data[ctrl_key].dtype}, "
                )

                return data

            except Exception as e:
                self.num_failed_loads += 1
                # Mark this video as bad so we skip it in the future
                self.bad_video_indices.add(index)

                log.warning(
                    f"Failed to load video {self.video_paths[index]} (index {index}): {e}. "
                    f"Marking as bad and trying next video. "
                    f"(attempt {retry + 1}/{max_retries}, total bad videos: {len(self.bad_video_indices)})",
                    rank0_only=False,
                )

                if retry == max_retries - 1:
                    log.error(
                        f"Failed to load data after {max_retries} attempts starting from index {original_index}. "
                        f"Total bad videos: {len(self.bad_video_indices)}/{len(self.video_paths)}"
                    )
                    raise RuntimeError(
                        f"Failed to load data after {max_retries} attempts. "
                        f"Original index: {original_index}, last tried: {video_path}"
                    )

                # Try the next video in sequence (wraps around at end)
                index = (index + 1) % len(self.video_paths)

        raise RuntimeError("Should not reach here")


class PRGSingleViewTransferDataset(SingleViewTransferDataset):
    """Load Render, target HD-map, and privileged PRG semantics as distinct inputs."""

    def __init__(
        self,
        *args,
        default_caption: str = "a realistic driving video",
        require_hdmap: bool = False,
        require_lidar: bool = False,
        lidar_mask_top_rows: int = 0,
        load_prg_labels: bool = True,
        use_reference_image: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.ctrl_type != "vis":
            raise ValueError("PRGSingleViewTransferDataset currently expects hint_key='control_input_vis'")

        manifest = load_prg_manifest(self.dataset_dir)
        self.prg_samples = manifest.get("samples", [])
        if not self.prg_samples:
            raise ValueError(f"No PRG samples found in {Path(self.dataset_dir) / 'manifest.json'}")

        self.video_paths = [str(Path(self.dataset_dir) / sample["shared_video"]) for sample in self.prg_samples]
        self.prg_samples_by_name = {sample["name"]: sample for sample in self.prg_samples}
        if len(self.prg_samples_by_name) != len(self.prg_samples):
            raise ValueError(f"Duplicate PRG sample names found in {Path(self.dataset_dir) / 'manifest.json'}")
        for sample, video_path in zip(self.prg_samples, self.video_paths):
            if Path(video_path).stem != sample["name"]:
                raise ValueError(f"PRG sample name {sample['name']!r} does not match video {video_path}")
            hdmap_path = sample.get("condition_videos", {}).get("hdmap")
            if require_hdmap and not hdmap_path:
                raise ValueError(f"PRG sample {sample['name']!r} is missing condition_videos.hdmap")
            lidar_path = sample.get("condition_videos", {}).get("lidar")
            if require_lidar and not lidar_path:
                raise ValueError(f"PRG sample {sample['name']!r} is missing condition_videos.lidar")
        self.require_hdmap = require_hdmap
        self.require_lidar = require_lidar
        self.lidar_mask_top_rows = lidar_mask_top_rows
        self.load_prg_labels = load_prg_labels
        self.use_reference_image = use_reference_image

        # Apply the same aspect-preserving resize and reflection padding to the
        # Render and HD-map controls use the same geometry as GT. PRG labels
        # replay it below with nearest-neighbor interpolation.
        geometry_keys = ["vis"]
        if require_hdmap:
            geometry_keys.append("hdmap")
        if require_lidar:
            geometry_keys.append("lidar")
        for augmentor_name in ("resize_largest_side_aspect_ratio_preserving", "reflection_padding"):
            augmentor = self.augmentor[augmentor_name]
            for key in geometry_keys:
                if key in augmentor.input_keys:
                    continue
                augmentor.input_keys.append(key)
                if augmentor.output_keys is not None:
                    augmentor.output_keys.append(key)

        self.default_caption = default_caption
        log.info(f"Initialized PRGSingleViewTransferDataset with {len(self.video_paths)} manifest samples")

    def _load_caption(self, video_name: str) -> str:
        caption = super()._load_caption(video_name)
        return self.default_caption if caption == "a video" else caption

    def _load_control_data(self, video_name: str, frame_ids: list[int]) -> dict[str, Any] | None:
        try:
            sample = self.prg_samples_by_name[video_name]
        except KeyError as error:
            raise KeyError(f"No PRG manifest sample matches video {video_name!r}") from error
        render_path = Path(self.dataset_dir) / sample["condition_videos"]["render"]
        render_reader = VideoReader(str(render_path), ctx=cpu(0), num_threads=2)
        if len(render_reader) < frame_ids[-1] + 1:
            raise ValueError(f"Render video has fewer frames than RGB video: {render_path}")

        render_frames = render_reader.get_batch(frame_ids).asnumpy().astype(np.uint8)
        result = {
            "vis": torch.from_numpy(render_frames).permute(3, 0, 1, 2),
            "_prg_render_source_hw": render_frames.shape[1:3],
        }

        lidar_relative_path = sample.get("condition_videos", {}).get("lidar")
        if lidar_relative_path is not None:
            lidar_path = Path(self.dataset_dir) / lidar_relative_path
            lidar_reader = VideoReader(str(lidar_path), ctx=cpu(0), num_threads=2)
            if len(lidar_reader) != len(render_reader):
                raise ValueError(
                    f"LiDAR and Render frame counts differ for {video_name}: "
                    f"{len(lidar_reader)} != {len(render_reader)}"
                )
            if abs(float(lidar_reader.get_avg_fps()) - float(render_reader.get_avg_fps())) > 1e-3:
                raise ValueError(
                    f"LiDAR and Render FPS differ for {video_name}: "
                    f"{lidar_reader.get_avg_fps()} != {render_reader.get_avg_fps()}"
                )
            if len(lidar_reader) < frame_ids[-1] + 1:
                raise ValueError(f"LiDAR video has fewer frames than RGB video: {lidar_path}")
            lidar_frames = lidar_reader.get_batch(frame_ids).asnumpy().astype(np.uint8)
            if lidar_frames.shape[1:3] != render_frames.shape[1:3]:
                raise ValueError(
                    f"LiDAR and Render sizes differ for {video_name}: "
                    f"{lidar_frames.shape[1:3]} != {render_frames.shape[1:3]}"
                )
            result["lidar"] = torch.from_numpy(lidar_frames).permute(3, 0, 1, 2)
            del lidar_reader
        elif self.require_lidar:
            raise ValueError(f"PRG sample {video_name!r} is missing condition_videos.lidar")

        hdmap_relative_path = sample.get("condition_videos", {}).get("hdmap")
        if hdmap_relative_path is None:
            if self.require_hdmap:
                raise ValueError(f"PRG sample {video_name!r} is missing condition_videos.hdmap")
            del render_reader
            return result

        hdmap_path = Path(self.dataset_dir) / hdmap_relative_path
        hdmap_reader = VideoReader(str(hdmap_path), ctx=cpu(0), num_threads=2)
        if len(hdmap_reader) != len(render_reader):
            raise ValueError(
                f"HD-map and Render frame counts differ for {video_name}: {len(hdmap_reader)} != {len(render_reader)}"
            )
        if abs(float(hdmap_reader.get_avg_fps()) - float(render_reader.get_avg_fps())) > 1e-3:
            raise ValueError(
                f"HD-map and Render FPS differ for {video_name}: "
                f"{hdmap_reader.get_avg_fps()} != {render_reader.get_avg_fps()}"
            )
        if len(hdmap_reader) < frame_ids[-1] + 1:
            raise ValueError(f"HD-map video has fewer frames than RGB video: {hdmap_path}")
        hdmap_frames = hdmap_reader.get_batch(frame_ids).asnumpy().astype(np.uint8)
        if hdmap_frames.shape[1:3] != render_frames.shape[1:3]:
            raise ValueError(
                f"HD-map and Render sizes differ for {video_name}: "
                f"{hdmap_frames.shape[1:3]} != {render_frames.shape[1:3]}"
            )
        result["hdmap"] = torch.from_numpy(hdmap_frames).permute(3, 0, 1, 2)
        del render_reader, hdmap_reader
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = super().__getitem__(index)
        video_name = data["__key__"]
        sample = self.prg_samples_by_name[video_name]
        render_source_hw = tuple(data.pop("_prg_render_source_hw"))
        if "hdmap" in data:
            data["control_input_seg"] = data.pop("hdmap")
        if self.require_hdmap and "control_input_seg" not in data:
            raise ValueError(f"PRG sample {video_name!r} did not produce an aligned HD-map SEG input")
        if "control_input_seg" in data:
            if data["control_input_seg"].dtype != torch.uint8:
                raise TypeError(f"HD-map dtype must be uint8, got {data['control_input_seg'].dtype}")
            if data["control_input_seg"].shape != data["video"].shape:
                raise ValueError(
                    f"HD-map shape {tuple(data['control_input_seg'].shape)} does not match "
                    f"video {tuple(data['video'].shape)}"
                )
        if "lidar" in data:
            data["control_input_lidar"] = data.pop("lidar")
        require_lidar = getattr(self, "require_lidar", False)
        if require_lidar and "control_input_lidar" not in data:
            raise ValueError(f"PRG sample {video_name!r} did not produce an aligned LiDAR input")
        if "control_input_lidar" in data:
            lidar = data["control_input_lidar"]
            if lidar.dtype != torch.uint8:
                raise TypeError(f"LiDAR dtype must be uint8, got {lidar.dtype}")
            if lidar.shape != data["video"].shape:
                raise ValueError(f"LiDAR shape {tuple(lidar.shape)} does not match video {tuple(data['video'].shape)}")
            height = lidar.shape[-2]
            lidar_mask_top_rows = getattr(self, "lidar_mask_top_rows", 0)
            if not 0 <= lidar_mask_top_rows <= height:
                raise ValueError(f"lidar_mask_top_rows must be in [0, {height}], got {lidar_mask_top_rows}")
            lidar_mask = torch.ones((1, lidar.shape[1], height, lidar.shape[-1]), dtype=torch.bool)
            lidar_mask[:, :, :lidar_mask_top_rows] = False
            lidar[:, :, :lidar_mask_top_rows] = 0
            data["control_input_lidar_mask"] = lidar_mask
        if getattr(self, "use_reference_image", False):
            frame_index = np.random.randint(data["video"].shape[1]) if self.is_train else 0
            reference = data["video"][:, frame_index : frame_index + 1].float()
            data["image_context"] = reference.div(127.5).sub(1.0)
        if not self.load_prg_labels:
            return data

        prg_aux_dir = Path(self.dataset_dir) / sample["prg_auxiliary"]
        frame_indices = data.get("frame_indices")
        if frame_indices is None:
            frame_indices = list(range(data["video"].shape[1]))
        prg_labels = load_prg_label_frames(prg_aux_dir, frame_ids=list(frame_indices))
        if tuple(prg_labels.shape[-2:]) != render_source_hw:
            raise ValueError(
                f"PRG labels {tuple(prg_labels.shape[-2:])} do not match render {render_source_hw} for {video_name}"
            )
        data["prg_label"] = resize_and_pad_prg_labels(prg_labels, data["image_size"])
        if tuple(data["prg_label"].shape[-2:]) != tuple(data["video"].shape[-2:]):
            raise ValueError(
                f"Transformed PRG labels {tuple(data['prg_label'].shape)} do not match video {tuple(data['video'].shape)}"
            )
        data["control_input_prg"] = prg_labels_to_rgb(data["prg_label"])
        if data["control_input_prg"].shape != data["video"].shape:
            raise ValueError(
                f"PRG semantic control {tuple(data['control_input_prg'].shape)} does not match "
                f"video {tuple(data['video'].shape)}"
            )
        data["prg_aux_dir"] = str(prg_aux_dir)
        data["prg_sample_name"] = video_name
        data["prg_regime"] = torch.tensor(PRG_REGIME_TO_ID[sample["regime"]], dtype=torch.long)
        return data


if __name__ == "__main__":
    """
    Sanity check for the dataset.

    Usage:
        PYTHONPATH=. python cosmos_transfer2/_src/transfer2/datasets/local_datasets/single_view_dataset.py
    """
    import sys

    # Example dataset with edge control (computed on-the-fly)
    dataset = SingleViewTransferDataset(
        dataset_dir="datasets/hdvila",
        num_frames=93,
        video_size=(704, 1280),
        resolution="720",
        hint_key="control_input_edge",
        is_train=True,
    )

    log.info(f"Dataset: {dataset}")
    log.info(f"Number of videos: {len(dataset)}")

    # Test loading a few samples
    indices = [0] if len(dataset) > 0 else []
    for idx in indices:
        log.info(f"\nTesting sample {idx}:")
        try:
            data = dataset[idx]
            log.info(f"  Video shape: {data['video'].shape}")
            log.info(f"  Control input shape: {data['control_input_edge'].shape}")
            log.info(f"  Caption: {data.get('ai_caption', 'N/A')[:100]}...")
            log.info(f"  FPS: {data['fps']}")
            log.info(f"  Aspect ratio: {data['aspect_ratio']}")
            log.info("  ✅ Sample loaded successfully")
        except Exception as e:
            log.error(f"  ❌ Failed to load sample: {e}")
            sys.exit(1)

    log.info("\n✅ All tests passed!")
