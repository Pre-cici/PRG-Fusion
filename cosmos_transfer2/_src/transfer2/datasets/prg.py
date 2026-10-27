# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image

PRG_CLASS_TO_ID = {
    "preserve": 0,
    "repair": 1,
    "generate": 2,
}

PRG_ID_TO_CLASS = {idx: name for name, idx in PRG_CLASS_TO_ID.items()}

PRG_REGIME_REPAIR = "repair"
PRG_REGIME_GENERATE = "generate"
PRG_REGIME_REGION_STUDENT = "region_student"
PRG_REGIME_REPAIR_ID = 0
PRG_REGIME_GENERATE_ID = 1
PRG_REGIME_REGION_STUDENT_ID = 2
PRG_REGIME_TO_ID = {
    PRG_REGIME_REPAIR: PRG_REGIME_REPAIR_ID,
    PRG_REGIME_GENERATE: PRG_REGIME_GENERATE_ID,
    PRG_REGIME_REGION_STUDENT: PRG_REGIME_REGION_STUDENT_ID,
}
PRG_ID_TO_REGIME = {idx: name for name, idx in PRG_REGIME_TO_ID.items()}

# Immutable v1 manifests used variant names as regime names.
PRG_LEGACY_REGIME_ALIASES = {
    "stride2": PRG_REGIME_REPAIR,
    "crossref_key3": PRG_REGIME_GENERATE,
}

# Clean PNG label colors emitted under prg_aux/*/label_frames.
PRG_LABEL_COLORS = {
    PRG_CLASS_TO_ID["preserve"]: (60, 180, 75),
    PRG_CLASS_TO_ID["repair"]: (255, 165, 0),
    PRG_CLASS_TO_ID["generate"]: (220, 60, 60),
}


def prg_labels_to_rgb(labels: torch.Tensor) -> torch.Tensor:
    """Convert class IDs ``[T,H,W]`` to the canonical uint8 RGB control ``[3,T,H,W]``."""
    if labels.ndim != 3:
        raise ValueError(f"Expected PRG labels [T,H,W], got {tuple(labels.shape)}")
    labels = labels.long()
    if ((labels < 0) | (labels >= len(PRG_LABEL_COLORS))).any():
        raise ValueError("PRG labels contain class IDs outside Preserve/Repair/Generate")
    palette = torch.tensor(
        [PRG_LABEL_COLORS[class_id] for class_id in range(len(PRG_LABEL_COLORS))],
        dtype=torch.uint8,
        device=labels.device,
    )
    return palette[labels].permute(3, 0, 1, 2).contiguous()


def normalize_prg_regime(regime: str) -> str:
    """Return a canonical PRG regime, accepting only known legacy aliases."""
    if not isinstance(regime, str):
        raise TypeError(f"PRG regime must be a string, got {type(regime).__name__}")
    if regime in PRG_REGIME_TO_ID:
        return regime
    try:
        return PRG_LEGACY_REGIME_ALIASES[regime]
    except KeyError as error:
        allowed = sorted([*PRG_REGIME_TO_ID, *PRG_LEGACY_REGIME_ALIASES])
        raise ValueError(f"Unknown PRG regime {regime!r}; expected one of {allowed}") from error


def _frame_number(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if match is None:
        raise ValueError(f"Could not parse frame number from {path}")
    return int(match.group(1))


def list_prg_label_frames(prg_aux_dir: str | Path) -> list[Path]:
    """Return PRG label PNGs sorted by numeric frame index."""
    prg_aux_dir = Path(prg_aux_dir)
    label_dirs = (prg_aux_dir / "class_labels", prg_aux_dir / "label_frames")
    label_dir = next((path for path in label_dirs if path.is_dir()), None)
    if label_dir is None:
        names = ", ".join(str(path) for path in label_dirs)
        raise FileNotFoundError(f"PRG lossless label directory not found; checked: {names}")

    frames = sorted(label_dir.glob("*.png"), key=_frame_number)
    if not frames:
        raise FileNotFoundError(f"No PRG label PNG files found in {label_dir}")
    return frames


def _load_prg_label_video(prg_aux_dir: str | Path, frame_ids: list[int] | None = None) -> torch.Tensor:
    """Load a legacy PRG label video as class IDs [T, H, W]."""
    video_path = Path(prg_aux_dir) / "prg_label.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"PRG label video not found: {video_path}")

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    total_frames = len(vr)
    if frame_ids is None:
        selected_frame_ids = list(range(total_frames))
    else:
        selected_frame_ids = frame_ids
        for frame_id in selected_frame_ids:
            if frame_id < 0 or frame_id >= total_frames:
                raise IndexError(f"PRG frame id {frame_id} out of range for {video_path} ({total_frames} frames)")

    frames = vr.get_batch(selected_frame_ids).asnumpy()
    del vr
    return torch.stack([decode_prg_label_image(frame, allow_nearest=True) for frame in frames], dim=0)


def decode_prg_label_image(image: np.ndarray | Image.Image, *, allow_nearest: bool = False) -> torch.Tensor:
    """Decode an RGB PRG label image into class IDs [H, W].

    Lossless label frames must contain exact palette colors. Nearest-color
    decoding is reserved for legacy video labels that may contain codec drift.
    """
    if isinstance(image, Image.Image):
        image_np = np.asarray(image)
    else:
        image_np = np.asarray(image)

    if image_np.ndim == 2 or (image_np.ndim == 3 and image_np.shape[-1] == 1):
        class_ids = image_np if image_np.ndim == 2 else image_np[..., 0]
        unique_ids = set(np.unique(class_ids).tolist())
        valid_ids = set(PRG_ID_TO_CLASS)
        if not unique_ids.issubset(valid_ids):
            raise ValueError(f"PRG class-ID image contains invalid IDs: {sorted(unique_ids - valid_ids)}")
        return torch.from_numpy(class_ids.astype(np.int64, copy=False))

    if image_np.ndim != 3 or image_np.shape[-1] < 3:
        raise ValueError(f"Expected class IDs [H,W] or RGB image [H,W,3], got {image_np.shape}")

    rgb = image_np[..., :3].astype(np.int32, copy=False)
    labels = np.full(rgb.shape[:2], -1, dtype=np.int64)
    for class_id, color in PRG_LABEL_COLORS.items():
        labels[np.all(rgb == np.asarray(color, dtype=np.int32), axis=-1)] = class_id

    unknown = labels < 0
    if not unknown.any():
        return torch.from_numpy(labels)

    if not allow_nearest:
        unknown_colors, unknown_counts = np.unique(rgb[unknown], axis=0, return_counts=True)
        examples = [
            f"{tuple(color.tolist())} ({int(count)} pixels)"
            for color, count in zip(unknown_colors[:5], unknown_counts[:5])
        ]
        raise ValueError(
            f"Lossless PRG label contains {int(unknown.sum())} pixels outside the declared palette: "
            + ", ".join(examples)
        )

    colors = np.asarray([PRG_LABEL_COLORS[i] for i in range(len(PRG_LABEL_COLORS))], dtype=np.int32)
    distances = ((rgb[unknown][:, None, :] - colors[None, :, :]) ** 2).sum(axis=-1)
    labels[unknown] = distances.argmin(axis=-1).astype(np.int64)
    return torch.from_numpy(labels)


def _load_prg_label_path(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return decode_prg_label_image(image)


def load_prg_label_frames(
    prg_aux_dir: str | Path,
    frame_ids: list[int] | None = None,
    resize_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Load PRG labels as a LongTensor with shape [T, H, W].

    Prefer lossless PNG labels under ``class_labels`` or ``label_frames`` and
    require exact class IDs or palette colors. If they are absent, fall back to legacy
    ``prg_label.mp4`` using int32 nearest-color decoding.
    """
    try:
        frame_paths = list_prg_label_frames(prg_aux_dir)
    except FileNotFoundError:
        labels = _load_prg_label_video(prg_aux_dir, frame_ids=frame_ids)
    else:
        if frame_ids is None:
            selected_paths = frame_paths
        else:
            selected_paths = []
            for frame_id in frame_ids:
                if frame_id < 0 or frame_id >= len(frame_paths):
                    raise IndexError(
                        f"PRG frame id {frame_id} out of range for {prg_aux_dir} ({len(frame_paths)} frames)"
                    )
                selected_paths.append(frame_paths[frame_id])
        labels = torch.stack([_load_prg_label_path(path) for path in selected_paths], dim=0)

    if resize_hw is not None and tuple(labels.shape[-2:]) != tuple(resize_hw):
        labels = F.interpolate(labels[:, None].float(), size=resize_hw, mode="nearest")[:, 0].long()
    return labels


def resize_and_pad_prg_labels(labels: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    """Replay the local dataset's aspect resize and reflection padding on labels.

    ``image_size`` follows ``ReflectionPadding`` and stores
    ``[target_h, target_w, resized_h, resized_w]``.
    """
    if labels.ndim != 3:
        raise ValueError(f"Expected PRG labels [T,H,W], got {tuple(labels.shape)}")
    if image_size.numel() != 4:
        raise ValueError(f"Expected image_size with four values, got {image_size}")

    target_h, target_w, resized_h, resized_w = (int(value) for value in image_size.tolist())
    if tuple(labels.shape[-2:]) != (resized_h, resized_w):
        labels = F.interpolate(labels[:, None].float(), size=(resized_h, resized_w), mode="nearest")[:, 0].long()

    pad_left = (target_w - resized_w) // 2
    pad_right = target_w - resized_w - pad_left
    pad_top = (target_h - resized_h) // 2
    pad_bottom = target_h - resized_h - pad_top
    if min(pad_left, pad_right, pad_top, pad_bottom) < 0:
        raise ValueError(f"Resized PRG label {(resized_h, resized_w)} exceeds target size {(target_h, target_w)}")
    if max(pad_left, pad_right, pad_top, pad_bottom) == 0:
        return labels

    padding_mode = "reflect"
    if max(pad_left, pad_right) >= resized_w or max(pad_top, pad_bottom) >= resized_h:
        padding_mode = "replicate"
    padded = F.pad(
        labels[None].float(),
        (pad_left, pad_right, pad_top, pad_bottom),
        mode=padding_mode,
    )
    return padded[0].long()


def load_prg_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(dataset_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"PRG manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for index, sample in enumerate(manifest.get("samples", [])):
        regime = sample.get("regime", sample.get("variant"))
        if regime is None:
            raise ValueError(f"PRG manifest sample {index} has neither 'regime' nor 'variant': {manifest_path}")
        try:
            sample["regime"] = normalize_prg_regime(regime)
        except (TypeError, ValueError) as error:
            name = sample.get("name", index)
            raise ValueError(f"Invalid PRG regime for sample {name!r} in {manifest_path}: {error}") from error
        hdmap_path = sample.get("condition_videos", {}).get("hdmap")
        if hdmap_path is not None and (not isinstance(hdmap_path, str) or not hdmap_path.strip()):
            name = sample.get("name", index)
            raise ValueError(f"Invalid HD-map path for PRG sample {name!r} in {manifest_path}")
    return manifest


def get_prg_sample(dataset_dir: str | Path, index: int = 0, name: str | None = None) -> dict[str, Any]:
    """Resolve one manifest sample and its absolute video/label paths."""
    root = Path(dataset_dir)
    manifest = load_prg_manifest(root)
    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in PRG manifest: {root / 'manifest.json'}")

    if name is None:
        sample = samples[index]
    else:
        matches = [sample for sample in samples if sample.get("name") == name]
        if not matches:
            raise KeyError(f"PRG sample {name!r} not found in {root / 'manifest.json'}")
        sample = matches[0]

    condition_videos = sample.get("condition_videos", {})
    resolved = {
        **sample,
        "video_path": root / sample["shared_video"],
        "render_path": root / condition_videos["render"],
        "prg_video_path": root / condition_videos["prg"],
        "prg_aux_dir": root / sample["prg_auxiliary"],
    }
    if "hdmap" in condition_videos:
        resolved["hdmap_path"] = root / condition_videos["hdmap"]
    return resolved


def load_video_tensor(video_path: str | Path, num_frames: int | None = None) -> torch.Tensor:
    """Load a local RGB video as float tensor [C, T, H, W] in [0, 1]."""
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    total_frames = len(vr)
    if num_frames is None:
        frame_ids = list(range(total_frames))
    else:
        if total_frames < num_frames:
            raise ValueError(f"Video {video_path} has {total_frames} frames, requested {num_frames}")
        frame_ids = list(range(num_frames))
    frames = vr.get_batch(frame_ids).asnumpy().astype(np.float32) / 255.0
    del vr
    return torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
