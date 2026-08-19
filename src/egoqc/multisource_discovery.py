from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .egodex_batch import ensure_readonly_source_boundary
from .report import write_json, write_jsonl


def _rank(path: Path, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{path.as_posix()}".encode()).hexdigest()


def _inspect_root(root: Path, source_root: Path) -> Dict[str, Any]:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features") or {}
    return {
        "dataset_root": str(root),
        "logical_path": root.relative_to(source_root).as_posix(),
        "task_group": root.parent.name,
        "dataset_id": root.name,
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "capability_hints": {
            "video": "observation.images.ego" in features,
            "state": "observation.state" in features,
            "state_mask": "state_mask" in features,
            "intrinsics": "intrinsics" in features,
            "extrinsics": "extrinsics_w2c" in features,
            "raw_world_hands": all(
                field in features
                for field in ("left_transl_world", "right_transl_world")
            ),
        },
        "source_readonly": True,
    }


def _candidate_roots(
    task_root: Path,
    source_root: Path,
    maximum_per_task: Optional[int],
    seed: int,
) -> List[Path]:
    # os.scandir preserves the directory-entry type returned by OSS/CPFS and
    # avoids a separate remote stat for every UUID directory.
    with os.scandir(task_root) as entries:
        candidates = [Path(entry.path) for entry in entries if entry.is_dir()]
    candidates.sort(
        key=lambda path: _rank(path.relative_to(source_root), seed)
    )
    return (
        candidates
        if maximum_per_task is None
        else candidates[:maximum_per_task]
    )


def discover_lerobot_roots(
    source_root: Path,
    output: Path,
    *,
    maximum_per_task: Optional[int] = 2,
    maximum_tasks: Optional[int] = None,
    workers: int = 16,
    seed: int = 17,
) -> Dict[str, Any]:
    """Discover the fixed ``task/dataset/meta/info.json`` layout without deep rglob."""
    if maximum_per_task is not None and maximum_per_task <= 0:
        raise ValueError("maximum_per_task 必须大于 0")
    if maximum_tasks is not None and maximum_tasks <= 0:
        raise ValueError("maximum_tasks 必须大于 0")
    if workers <= 0:
        raise ValueError("workers 必须大于 0")
    source_root = source_root.expanduser().resolve()
    output = output.expanduser().resolve()
    ensure_readonly_source_boundary(source_root, output)

    all_task_roots = sorted(
        path for path in source_root.iterdir() if path.is_dir()
    )
    task_roots = all_task_roots
    if maximum_tasks is not None:
        task_roots = sorted(
            all_task_roots,
            key=lambda path: _rank(path.relative_to(source_root), seed),
        )[:maximum_tasks]
    roots: List[Path] = []
    errors: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _candidate_roots,
                task_root,
                source_root,
                maximum_per_task,
                seed,
            ): task_root
            for task_root in task_roots
        }
        for future in as_completed(futures):
            task_root = futures[future]
            try:
                roots.extend(future.result())
            except Exception as error:
                errors.append({
                    "dataset_root": str(task_root),
                    "task_group": task_root.name,
                    "stage": "list_task_datasets",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "source_readonly": True,
                })

    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_inspect_root, root, source_root): root for root in roots
        }
        for future in as_completed(futures):
            root = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                errors.append({
                    "dataset_root": str(root),
                    "task_group": root.parent.name,
                    "stage": "inspect_dataset_root",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "source_readonly": True,
                })
    records.sort(key=lambda row: row["logical_path"])
    errors.sort(key=lambda row: row["dataset_root"])
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "datasets.jsonl", records)
    write_jsonl(output / "errors.jsonl", errors)
    dataset_list = output / "dataset-list.txt"
    temporary = dataset_list.with_name(f".{dataset_list.name}.tmp")
    temporary.write_text(
        "".join(f"{row['dataset_root']}\n" for row in records), encoding="utf-8"
    )
    temporary.replace(dataset_list)
    summary = {
        "schema_version": "egoqc-multisource-discovery-v1",
        "source_root": str(source_root),
        "source_readonly": True,
        "dataset_roots": len(records),
        "task_groups": len({row["task_group"] for row in records}),
        "available_task_groups": len(all_task_roots),
        "errors": len(errors),
        "maximum_per_task": maximum_per_task,
        "maximum_tasks": maximum_tasks,
        "workers": workers,
        "episode_hints": sum(int(row.get("total_episodes") or 0) for row in records),
        "frame_hints": sum(int(row.get("total_frames") or 0) for row in records),
        "capability_counts": {
            key: sum(bool(row["capability_hints"].get(key)) for row in records)
            for key in (
                "video", "state", "state_mask", "intrinsics", "extrinsics",
                "raw_world_hands",
            )
        },
        "artifacts": {
            "datasets": str(output / "datasets.jsonl"),
            "dataset_list": str(dataset_list),
            "errors": str(output / "errors.jsonl"),
        },
    }
    write_json(output / "summary.json", summary)
    return summary
