from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import av


def fixed_list(values: np.ndarray) -> pa.Array:
    return pa.array(values.tolist())


def create_fixture(root: Path, frames: int = 12, episodes: int = 1) -> Path:
    total_frames = frames * episodes
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "videos" / "observation.images.ego" / "chunk-000").mkdir(parents=True, exist_ok=True)
    features = {
        "action": {"dtype": "float64", "shape": [102]},
        "observation.state": {"dtype": "float64", "shape": [122]},
        "state_mask": {"dtype": "bool", "shape": [2]},
        "fov": {"dtype": "float64", "shape": [2]},
        "intrinsics": {"dtype": "float64", "shape": [9]},
        "extrinsics_w2c": {"dtype": "float64", "shape": [16]},
        "main_type": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "observation.images.ego": {
            "dtype": "video",
            "shape": [90, 160, 3],
            "info": {
                "video.fps": 30.0,
                "video.height": 90,
                "video.width": 160,
                "video.channel": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
    }
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": 30.0, "features": features}), encoding="utf-8"
    )
    pq.write_table(
        pa.table({"task": ["fixture task"], "task_index": [0]}),
        root / "meta" / "tasks.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": np.arange(episodes),
                "length": np.full(episodes, frames),
                "tasks": [["fixture task"] for _ in range(episodes)],
                "dataset_from_index": np.arange(episodes) * frames,
                "dataset_to_index": (np.arange(episodes) + 1) * frames,
                "data/chunk_index": np.zeros(episodes, dtype=np.int64),
                "data/file_index": np.zeros(episodes, dtype=np.int64),
                "videos/observation.images.ego/chunk_index": np.zeros(episodes, dtype=np.int64),
                "videos/observation.images.ego/file_index": np.zeros(episodes, dtype=np.int64),
                "videos/observation.images.ego/from_timestamp": np.arange(episodes) * frames / 30.0,
                "videos/observation.images.ego/to_timestamp": (np.arange(episodes) + 1) * frames / 30.0,
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    eye3 = np.broadcast_to(np.eye(3), (total_frames, 3, 3)).copy()
    eye15 = np.broadcast_to(np.eye(3), (total_frames, 15, 3, 3)).copy()
    extr = np.broadcast_to(np.eye(4), (total_frames, 4, 4)).copy()
    local_x = np.tile(np.linspace(0, 0.1, frames), episodes)
    left = np.column_stack([local_x, np.zeros(total_frames), np.ones(total_frames)])
    right = left + np.array([0.2, 0.0, 0.0])
    state = np.zeros((total_frames, 122))
    state[:, 0:3] = left
    state[:, 61:64] = right
    table = pa.table(
        {
            "index": np.arange(total_frames),
            "frame_index": np.tile(np.arange(frames), episodes),
            "episode_index": np.repeat(np.arange(episodes), frames),
            "task_index": np.zeros(total_frames, dtype=np.int64),
            "main_type": -np.ones(total_frames, dtype=np.int64),
            "timestamp": np.tile(np.arange(frames) / 30.0, episodes),
            "state_mask": fixed_list(np.ones((total_frames, 2), dtype=bool)),
            "observation.state": fixed_list(state),
            "fov": fixed_list(np.broadcast_to([1.0, 0.7], (total_frames, 2))),
            "intrinsics": fixed_list(np.broadcast_to(np.eye(3).reshape(9), (total_frames, 9))),
            "extrinsics_w2c": fixed_list(extr.reshape(total_frames, 16)),
            "left_transl_world": fixed_list(left),
            "left_orient_world": fixed_list(eye3.reshape(total_frames, 9)),
            "left_hand_pose": fixed_list(eye15.reshape(total_frames, 135)),
            "left_kept": np.ones(total_frames, dtype=bool),
            "left_seg_start": -np.ones(total_frames, dtype=np.int64),
            "left_seg_end": -np.ones(total_frames, dtype=np.int64),
            "right_transl_world": fixed_list(right),
            "right_orient_world": fixed_list(eye3.reshape(total_frames, 9)),
            "right_hand_pose": fixed_list(eye15.reshape(total_frames, 135)),
            "right_kept": np.ones(total_frames, dtype=bool),
            "right_seg_start": -np.ones(total_frames, dtype=np.int64),
            "right_seg_end": -np.ones(total_frames, dtype=np.int64),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    video_path = root / "videos" / "observation.images.ego" / "chunk-000" / "file-000.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("h264", rate=30)
        stream.width = 160
        stream.height = 90
        stream.pix_fmt = "yuv420p"
        for index in range(total_frames):
            pixels = np.zeros((90, 160, 3), dtype=np.uint8)
            pixels[..., 0] = (index * 10) % 256
            start = (index * 4) % max(1, pixels.shape[1] - 12)
            pixels[:, start : start + 12, 1] = 220
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return root
