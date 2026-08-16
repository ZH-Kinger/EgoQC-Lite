from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import av
import numpy as np
from PIL import Image, ImageDraw

from .report import write_json


APPLE_EGODEX_UPSTREAM = {
    "repository": "https://github.com/apple/ml-egodex",
    "commit": "7a1801597844bc7712b179aac83a48b6c4335f3a",
    "projection": "inverse(camera_world) @ joint_world; pinhole K; zero distortion",
}

FINGERS = {
    "little": ["LittleFingerMetacarpal", "LittleFingerKnuckle", "LittleFingerIntermediateBase", "LittleFingerIntermediateTip", "LittleFingerTip"],
    "ring": ["RingFingerMetacarpal", "RingFingerKnuckle", "RingFingerIntermediateBase", "RingFingerIntermediateTip", "RingFingerTip"],
    "middle": ["MiddleFingerMetacarpal", "MiddleFingerKnuckle", "MiddleFingerIntermediateBase", "MiddleFingerIntermediateTip", "MiddleFingerTip"],
    "index": ["IndexFingerMetacarpal", "IndexFingerKnuckle", "IndexFingerIntermediateBase", "IndexFingerIntermediateTip", "IndexFingerTip"],
    "thumb": ["ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip"],
}
COLORS = {
    "little": (0, 152, 191), "ring": (173, 255, 47), "middle": (230, 245, 250),
    "index": (255, 99, 71), "thumb": (238, 130, 238),
}


def _project(point: np.ndarray, intrinsic: np.ndarray) -> Tuple[float, float] | None:
    if not np.isfinite(point).all() or point[2] <= 1e-6:
        return None
    pixel = intrinsic @ point
    return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])


def _draw_chain(
    draw: ImageDraw.ImageDraw,
    names: Sequence[str],
    transforms: Dict[str, np.ndarray],
    camera_inverse: np.ndarray,
    intrinsic: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    points: List[Tuple[float, float] | None] = []
    for name in names:
        camera_transform = camera_inverse @ transforms[name]
        points.append(_project(camera_transform[:3, 3], intrinsic))
    for start, end in zip(points, points[1:]):
        if start is None or end is None:
            continue
        draw.line((start, end), fill=color, width=5)
        radius = 5
        for x, y in (start, end):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def render_egodex_overlay(
    dataset: Path,
    episode: str,
    output: Path,
    *,
    start_frame: int = 0,
    max_frames: int = 300,
    stride: int = 1,
) -> Dict[str, Any]:
    if start_frame < 0 or max_frames < 1 or stride < 1:
        raise ValueError("start_frame>=0, max_frames>=1, stride>=1")
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("EgoDex overlay requires: pip install -e '.[egodex]'") from error

    base = dataset.expanduser().resolve() / episode
    hdf5_path = base.with_suffix(".hdf5")
    video_path = base.with_suffix(".mp4")
    if not hdf5_path.is_file() or not video_path.is_file():
        raise FileNotFoundError(f"EgoDex pair missing: {hdf5_path}, {video_path}")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as handle:
        frame_count = int(handle["transforms/camera"].shape[0])
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        stop = min(frame_count, start_frame + max_frames * stride)
        indices = list(range(start_frame, stop, stride))
        names = [
            f"{side}{suffix}" for side in ("left", "right")
            for suffixes in FINGERS.values() for suffix in suffixes
        ] + ["leftHand", "rightHand", "leftForearm", "rightForearm"]
        sampled = {
            name: np.asarray(handle[f"transforms/{name}"][indices], dtype=np.float64)
            for name in names
        }
        cameras = np.asarray(handle["transforms/camera"][indices], dtype=np.float64)
        confidences = {}
        if "confidences" in handle:
            for side in ("left", "right"):
                confidences[side] = np.asarray(handle[f"confidences/{side}Hand"][indices])

    input_container = av.open(str(video_path))
    input_stream = input_container.streams.video[0]
    source_fps = float(input_stream.average_rate or 30)
    output_rate = Fraction(input_stream.average_rate or Fraction(30, 1)) / stride
    output_fps = float(output_rate)
    output_container = av.open(str(output), mode="w")
    output_stream = output_container.add_stream("libx264", rate=output_rate)
    output_stream.width = input_stream.codec_context.width
    output_stream.height = input_stream.codec_context.height
    output_stream.pix_fmt = "yuv420p"
    output_stream.options = {"crf": "18", "preset": "medium"}

    selected_position = 0
    selected_lookup = set(indices)
    try:
        for frame_index, frame in enumerate(input_container.decode(input_stream)):
            if frame_index >= stop or selected_position >= len(indices):
                break
            if frame_index not in selected_lookup:
                continue
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            draw = ImageDraw.Draw(image)
            camera_inverse = np.linalg.inv(cameras[selected_position])
            frame_transforms = {name: values[selected_position] for name, values in sampled.items()}
            for side in ("left", "right"):
                for finger, suffixes in FINGERS.items():
                    _draw_chain(
                        draw, [f"{side}Hand"] + [f"{side}{suffix}" for suffix in suffixes],
                        frame_transforms, camera_inverse, intrinsic, COLORS[finger],
                    )
                _draw_chain(
                    draw, [f"{side}Forearm", f"{side}Hand"], frame_transforms,
                    camera_inverse, intrinsic, COLORS["middle"],
                )
            confidence_text = " ".join(
                f"{side[0].upper()}={float(values[selected_position]):.2f}"
                for side, values in confidences.items()
            )
            draw.rectangle((12, 12, 390, 52), fill=(0, 0, 0))
            draw.text((20, 20), f"frame={frame_index} {confidence_text}", fill=(255, 255, 255))
            encoded_frame = av.VideoFrame.from_image(image)
            for packet in output_stream.encode(encoded_frame):
                output_container.mux(packet)
            selected_position += 1
        for packet in output_stream.encode():
            output_container.mux(packet)
    finally:
        input_container.close()
        output_container.close()

    report = {
        "schema_version": "egoqc-egodex-overlay-v1",
        "episode": episode,
        "source_video": str(video_path),
        "source_hdf5": str(hdf5_path),
        "output_video": str(output),
        "rendered_frames": selected_position,
        "start_frame": start_frame,
        "stride": stride,
        "output_fps": output_fps,
        "distortion_status": "not_required_no_distortion_model_provided",
        "geometry_warning": "Apple documents perspective mismatch from Vision Pro multi-camera RGB synthesis",
        "upstream": APPLE_EGODEX_UPSTREAM,
        "raw_immutable": True,
    }
    write_json(output.with_suffix(".json"), report)
    return report
