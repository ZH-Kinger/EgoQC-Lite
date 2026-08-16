from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional

from .provenance import code_version
from .report import write_json, write_jsonl
from .video import probe_video


SCHEMA_VERSION = "egoqc-vitra-undistortion-v1"
VITRA_UPSTREAM = {
    "repository": "https://github.com/microsoft/VITRA",
    "commit": "b35517202b39d32a753fdd42014b2cc3c41fab58",
    "license": "MIT",
}


def _ids(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        values = value if isinstance(value, list) else value.get("video_ids", [])
        return [str(item) for item in values]
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _task_id(kind: str, video_id: str, source: Path, intrinsics: Path) -> str:
    payload = f"{kind}\0{video_id}\0{source.resolve()}\0{intrinsics.resolve()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def plan_vitra_undistortion(
    dataset_kind: str,
    video_root: Path,
    intrinsics_root: Path,
    save_root: Path,
    output: Path,
    *,
    selection_list: Optional[Path] = None,
    aria_name_map: Optional[Path] = None,
) -> Dict[str, Any]:
    if dataset_kind not in {"ego4d", "egoexo4d"}:
        raise ValueError("dataset_kind 必须是 ego4d 或 egoexo4d")
    video_root = video_root.expanduser().resolve()
    intrinsics_root = intrinsics_root.expanduser().resolve()
    save_root = save_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if save_root == video_root or video_root in save_root.parents:
        raise ValueError("save_root 必须与 raw video_root 分离，且不能位于 raw 内部")
    selected = _ids(selection_list)
    aria_names: Dict[str, str] = {}

    if dataset_kind == "ego4d":
        source_ids = sorted(path.stem for path in video_root.glob("*.mp4"))
        video_ids = selected if selected is not None else source_ids
        camera_model = "opencv_omnidir"
        upstream_script = "data/preprocessing/undistort_video.py"
    else:
        if aria_name_map is None or not aria_name_map.is_file():
            raise ValueError("egoexo4d 需要 --aria-name-map")
        aria_names = {str(key): str(value) for key, value in json.loads(
            aria_name_map.read_text(encoding="utf-8")
        ).items()}
        takes_root = video_root / "takes"
        source_ids = sorted(path.name for path in takes_root.iterdir() if path.is_dir()) if takes_root.exists() else []
        video_ids = selected if selected is not None else source_ids
        camera_model = "fisheye624_to_linear"
        upstream_script = "data/preprocessing/undistort_video_egoexo4d.py"

    tasks = []
    for video_id in video_ids:
        if dataset_kind == "ego4d":
            source = video_root / f"{video_id}.mp4"
            intrinsics = intrinsics_root / f"{video_id}.npy"
            mapping_status = "not_applicable"
        else:
            aria_name = aria_names.get(video_id)
            source = (
                video_root / "takes" / video_id / "frame_aligned_videos" / f"{aria_name}_214-1.mp4"
                if aria_name else video_root / "takes" / video_id / "frame_aligned_videos" / "MISSING_ARIA_MAP.mp4"
            )
            intrinsics = intrinsics_root / f"{video_id}.json"
            mapping_status = "present" if aria_name else "missing"
        destination = save_root / f"{video_id}.mp4"
        reasons = []
        if not source.is_file():
            reasons.append("source_video_missing")
        if not intrinsics.is_file():
            reasons.append("intrinsics_missing")
        if mapping_status == "missing":
            reasons.append("aria_name_mapping_missing")
        state = "blocked" if reasons else "output_exists_unverified" if destination.is_file() else "ready"
        tasks.append({
            "schema_version": SCHEMA_VERSION,
            "task_id": _task_id(dataset_kind, video_id, source, intrinsics),
            "dataset_kind": dataset_kind,
            "video_id": video_id,
            "source_video": str(source),
            "source_fingerprint": _fingerprint(source) if source.is_file() else None,
            "intrinsics_file": str(intrinsics),
            "aria_name": aria_names.get(video_id) if dataset_kind == "egoexo4d" else None,
            "intrinsics_fingerprint": _fingerprint(intrinsics) if intrinsics.is_file() else None,
            "destination_video": str(destination),
            "camera_model": camera_model,
            "distortion_status_input": "raw",
            "distortion_status_output": "rectified_pinhole",
            "geometry": (
                {"method": "cv2.omnidir.RECTIFY_PERSPECTIVE", "output_size": "same_as_source"}
                if dataset_kind == "ego4d" else
                {"method": "projectaria_tools.calibration.distort_by_calibration",
                 "linear_width": 1408, "linear_height": 1408, "linear_focal_length": 412.5,
                 "vitra_training_transform": "resize_1408_to_448_then_center_crop_256"}
            ),
            "encoding": {"codec": "libx264", "crf": 22, "pix_fmt": "yuv420p"},
            "upstream": {**VITRA_UPSTREAM, "script": upstream_script},
            "state": state,
            "reason_codes": reasons,
            "raw_immutable": True,
            "code_version": code_version(),
        })

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "undistortion-tasks.jsonl", tasks)
    ready = [task for task in tasks if task["state"] == "ready"]
    blocked = [task for task in tasks if task["state"] == "blocked"]
    existing = [task for task in tasks if task["state"] == "output_exists_unverified"]
    write_jsonl(output / "ready.jsonl", ready)
    write_jsonl(output / "blocked.jsonl", blocked)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": dataset_kind,
        "source_population_videos": len(source_ids),
        "selected_videos": len(video_ids),
        "selected_source_videos": len(set(video_ids) & set(source_ids)),
        "selection_coverage_ratio": len(set(video_ids) & set(source_ids)) / len(source_ids) if source_ids else None,
        "ready": len(ready),
        "existing_unverified": len(existing),
        "blocked": len(blocked),
        "raw_immutable": True,
        "upstream": VITRA_UPSTREAM,
        "manifest": str(output / "undistortion-tasks.jsonl"),
    }
    write_json(output / "summary.json", summary)
    return summary


def _vitra_commit(vitra_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(vitra_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_vitra_undistortion(
    manifest: Path,
    vitra_root: Path,
    output: Path,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
    max_tasks: Optional[int] = None,
    batch_size: int = 2_000_000_000,
    crf: int = 22,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute pinned official VITRA transforms with resumable per-video isolation.

    VITRA owns the image geometry and ffmpeg settings. EgoQC only selects stable
    manifest shards, runs each video in a same-filesystem temporary directory,
    atomically publishes the MP4 and records failures for retry.
    """
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index 必须位于 [0, shard_count)")
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0")
    manifest = manifest.expanduser().resolve()
    vitra_root = vitra_root.expanduser().resolve()
    output = output.expanduser().resolve()
    preprocessing = vitra_root / "data" / "preprocessing"
    required = (
        preprocessing / "undistort_video.py",
        preprocessing / "undistort_video_egoexo4d.py",
        preprocessing / "utils.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"VITRA checkout 不完整: {missing}")
    upstream_commit = _vitra_commit(vitra_root)
    if upstream_commit != VITRA_UPSTREAM["commit"]:
        raise ValueError(
            f"VITRA commit 不匹配: expected={VITRA_UPSTREAM['commit']} actual={upstream_commit}"
        )

    tasks = [
        task for task in _read_jsonl(manifest)
        if task.get("state") == "ready"
        and int(task["task_id"], 16) % shard_count == shard_index
    ]
    if max_tasks is not None:
        tasks = tasks[:max(0, max_tasks)]
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"execution-shard-{shard_index:05d}-of-{shard_count:05d}.jsonl"
    completed = set()
    if result_path.is_file() and not overwrite:
        completed = {
            row["task_id"] for row in _read_jsonl(result_path)
            if row.get("decision") == "pass"
        }

    records: List[Dict[str, Any]] = []
    with result_path.open("a", encoding="utf-8") as result_handle:
        for task in tasks:
            task_id = task["task_id"]
            destination = Path(task["destination_video"])
            if task_id in completed or (destination.is_file() and not overwrite):
                records.append({"task_id": task_id, "video_id": task["video_id"], "decision": "skip_existing"})
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            record: Dict[str, Any] = {
                "task_id": task_id,
                "video_id": task["video_id"],
                "dataset_kind": task["dataset_kind"],
                "decision": "fail",
                "reason_code": None,
                "elapsed_seconds": None,
                "destination_video": str(destination),
                "upstream_commit": upstream_commit,
            }
            try:
                with tempfile.TemporaryDirectory(
                    prefix=f".egoqc-vitra-{task_id}-", dir=destination.parent
                ) as staging_value:
                    staging = Path(staging_value)
                    task_file = staging / "task.json"
                    task_file.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
                    command = [
                        sys.executable, "-m", "egoqc.vitra_worker",
                        "--task", str(task_file), "--vitra-root", str(vitra_root),
                        "--save-root", str(staging), "--batch-size", str(batch_size),
                        "--crf", str(crf),
                    ]
                    process = subprocess.run(command, text=True, capture_output=True)
                    if process.returncode != 0:
                        message = (process.stderr or process.stdout).strip()[-4000:]
                        raise RuntimeError(message or f"worker exit {process.returncode}")
                    staged_output = staging / f"{task['video_id']}.mp4"
                    if not staged_output.is_file() or staged_output.stat().st_size == 0:
                        raise RuntimeError("official worker did not create a non-empty MP4")
                    os.replace(staged_output, destination)
                record["decision"] = "pass"
            except Exception as error:
                record["reason_code"] = "vitra_worker_failed"
                record["error"] = str(error)
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            records.append(record)
            result_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            result_handle.flush()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest),
        "upstream": {**VITRA_UPSTREAM, "actual_commit": upstream_commit},
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_tasks": len(tasks),
        "processed_pass": sum(row["decision"] == "pass" for row in records),
        "processed_fail": sum(row["decision"] == "fail" for row in records),
        "skipped_existing": sum(row["decision"] == "skip_existing" for row in records),
        "result_log": str(result_path),
        "raw_immutable": True,
    }
    write_json(output / f"summary-shard-{shard_index:05d}-of-{shard_count:05d}.json", summary)
    return summary


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verify_vitra_undistortion(manifest: Path, output: Path) -> Dict[str, Any]:
    records = []
    for task in _read_jsonl(manifest):
        reasons = list(task.get("reason_codes", []))
        source = Path(task["source_video"])
        destination = Path(task["destination_video"])
        source_probe = destination_probe = None
        if task.get("source_fingerprint") and source.is_file():
            if _fingerprint(source) != task["source_fingerprint"]:
                reasons.append("source_changed_after_plan")
        if not destination.is_file():
            reasons.append("rectified_output_missing")
        if not reasons:
            try:
                source_probe, source_issues = probe_video(source, "count")
                destination_probe, destination_issues = probe_video(destination, "count")
                reasons.extend(f"source_{issue.code}" for issue in source_issues if issue.severity == "error")
                reasons.extend(f"output_{issue.code}" for issue in destination_issues if issue.severity == "error")
                if source_probe.get("counted_frames") != destination_probe.get("counted_frames"):
                    reasons.append("frame_count_mismatch")
                source_fps = source_probe.get("average_rate")
                destination_fps = destination_probe.get("average_rate")
                if source_fps is None or destination_fps is None or abs(source_fps - destination_fps) > 0.01:
                    reasons.append("fps_mismatch")
                if task["dataset_kind"] == "egoexo4d" and (
                    destination_probe.get("width"), destination_probe.get("height")
                ) != (1408, 1408):
                    reasons.append("egoexo4d_output_not_1408_square")
            except Exception as error:
                reasons.append("video_probe_failed")
                destination_probe = {"error": str(error)}
        records.append({
            "task_id": task["task_id"],
            "video_id": task["video_id"],
            "dataset_kind": task["dataset_kind"],
            "source_video": str(source),
            "destination_video": str(destination),
            "decision": "pass" if not reasons else "fail",
            "reason_codes": sorted(set(reasons)),
            "source_probe": source_probe,
            "destination_probe": destination_probe,
            "geometry_verified": False,
            "geometry_note": "frame equality is integrity-only; reprojection/overlay review remains required",
        })
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "undistortion-verification.jsonl", records)
    failures = [record for record in records if record["decision"] == "fail"]
    write_jsonl(output / "failures.jsonl", failures)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "tasks": len(records),
        "integrity_pass": len(records) - len(failures),
        "integrity_fail": len(failures),
        "geometry_verified": 0,
        "warning": "integrity pass is not geometric undistortion acceptance",
        "report": str(output / "undistortion-verification.jsonl"),
    }
    write_json(output / "summary.json", summary)
    return summary
