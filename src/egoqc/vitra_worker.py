from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one official Microsoft VITRA undistortion task")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vitra-root", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2_000_000_000)
    parser.add_argument("--crf", type=int, default=22)
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    preprocessing = args.vitra_root / "data" / "preprocessing"
    sys.path.insert(0, str(preprocessing))
    args.save_root.mkdir(parents=True, exist_ok=True)

    if task["dataset_kind"] == "ego4d":
        module = _load_module(preprocessing / "undistort_video.py", "vitra_ego4d_undistort")
        module.process_single_video(
            task["video_id"], str(Path(task["source_video"]).parent),
            str(Path(task["intrinsics_file"]).parent), str(args.save_root),
            args.batch_size, args.crf,
        )
    elif task["dataset_kind"] == "egoexo4d":
        if not task.get("aria_name"):
            raise ValueError("aria_name missing from task")
        module = _load_module(preprocessing / "undistort_video_egoexo4d.py", "vitra_egoexo4d_undistort")
        # source: <video_root>/takes/<take>/frame_aligned_videos/<aria>_214-1.mp4
        source = Path(task["source_video"])
        video_root = source.parents[3]
        module.process_single_video(
            task["video_id"], task["aria_name"], str(video_root),
            str(Path(task["intrinsics_file"]).parent), str(args.save_root),
            args.batch_size, args.crf,
        )
    else:
        raise ValueError(f"unsupported dataset_kind: {task['dataset_kind']}")


if __name__ == "__main__":
    main()
