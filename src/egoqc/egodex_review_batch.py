from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .egodex_batch import ensure_readonly_source_boundary
from .provenance import code_version
from .report import write_json, write_jsonl


RAW_UNOBSERVABLE_TASKS = {"mano_overlay_drift"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rank(row: Dict[str, Any], seed: int) -> str:
    return hashlib.sha256(
        f"{seed}:{row['episode_id']}".encode("utf-8")
    ).hexdigest()


def _balanced_limit(
    rows: Sequence[Dict[str, Any]], maximum: Optional[int], seed: int
) -> List[Dict[str, Any]]:
    if maximum is None or len(rows) <= maximum:
        return sorted(rows, key=lambda row: _rank(row, seed))
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["partition"]), str(row["task"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: _rank(row, seed))
    selected: List[Dict[str, Any]] = []
    keys = sorted(grouped)
    while len(selected) < maximum:
        progressed = False
        for key in keys:
            if grouped[key] and len(selected) < maximum:
                selected.append(grouped[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _request_id(episode_id: str, start_s: float, end_s: float) -> str:
    value = f"{episode_id}:{start_s:.6f}:{end_s:.6f}:raw"
    return "egodex-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def build_egodex_review_batch(
    profiles: Path,
    task_config: Path,
    output: Path,
    *,
    dataset: Optional[Path] = None,
    window_s: float = 6.0,
    maximum_clean: Optional[int] = None,
    maximum_hard_negative: Optional[int] = None,
    maximum_review: Optional[int] = 128,
    seed: int = 17,
) -> Dict[str, Any]:
    if window_s <= 0:
        raise ValueError("window_s 必须大于 0")
    for value in (maximum_clean, maximum_hard_negative, maximum_review):
        if value is not None and value < 0:
            raise ValueError("maximum_* 不能小于 0")
    profiles = profiles.expanduser().resolve()
    output = output.expanduser().resolve()
    rows = _load_jsonl(profiles)
    if dataset is None:
        source_paths = [Path(row["hdf5_path"]) for row in rows if row.get("hdf5_path")]
        if not source_paths:
            raise ValueError("profiles 缺少 hdf5_path，无法推断只读数据集")
        # Expected layout: <dataset>/<partition>/<task>/<episode>.hdf5
        dataset = source_paths[0].parents[2]
    dataset = dataset.expanduser().resolve()
    ensure_readonly_source_boundary(dataset, output)

    limits = {
        "candidate-clean": maximum_clean,
        "hard-negative": maximum_hard_negative,
        "review": maximum_review,
    }
    selected: List[Dict[str, Any]] = []
    for tier, maximum in limits.items():
        selected.extend(_balanced_limit(
            [row for row in rows if row.get("candidate_tier") == tier],
            maximum,
            seed,
        ))
    selected.sort(key=lambda row: (row["candidate_tier"], row["episode_id"]))

    config = json.loads(task_config.read_text(encoding="utf-8"))
    dimensions = dict(config.get("assessment_dimensions") or {})
    model_tasks = [
        task for task in (config.get("model_tasks") or {})
        if task not in RAW_UNOBSERVABLE_TASKS
    ]
    if not model_tasks:
        raise ValueError("task config 没有适用于 raw 视频的视觉任务")

    review_rows = []
    teacher_rows = []
    for row in selected:
        duration_s = float(row["duration_s"])
        clip_duration = min(window_s, duration_s)
        start_s = max(0.0, (duration_s - clip_duration) / 2.0)
        end_s = start_s + clip_duration
        request_id = _request_id(row["episode_id"], start_s, end_s)
        context = {
            "candidate_tier": row["candidate_tier"],
            "annotation_score": row.get("annotation_score"),
            "hard_gates": row.get("hard_gates") or {},
            "hand_metrics": {
                key: value for key, value in (row.get("hand_metrics") or {}).items()
                if key not in {"left", "right"}
            },
        }
        base = {
            "request_id": request_id,
            "episode_id": row["episode_id"],
            "partition": row["partition"],
            "task": row["task"],
            "source_uri": row["video_path"],
            "raw_source_uri": row["video_path"],
            "clip_start_s": start_s,
            "clip_end_s": end_s,
            "candidate_tier": row["candidate_tier"],
            "weak_label_only": True,
            "task_context": row.get("labels") or {},
            "capability_context": row.get("capabilities") or {},
            "programmatic_context": context,
            "visual_evidence": "raw_video",
            "source_readonly": True,
            "provenance": {"code_version": code_version(), "raw_immutable": True},
        }
        review_rows.append({
            **base,
            "schema_version": "egoqc-egodex-review-item-v1",
            "review_status": "pending",
            "review_required_for_gold": True,
        })
        teacher_rows.append({
            **base,
            "schema_version": "egoqc-visual-teacher-request-v1",
            "prompt_version": "egoqc-visual-teacher-v3-open-world",
            "candidate_tasks": model_tasks,
            "trigger_tasks": [],
            "event_codes": [],
            "selection_source": f"egodex_{row['candidate_tier']}",
            "assessment_dimensions": dimensions,
            "output_path": str(output / "teacher-labels" / request_id / "teacher-label.json"),
            "required_response": {
                "schema_version": "egoqc-visual-teacher-v1",
                "overall": {
                    "training_usable": "boolean",
                    "recommended_route": "accept|human_review|reject",
                    "confidence": "float[0,1]",
                    "allowed_uses": "list[string]",
                },
                "tasks": {
                    task: {"probability": "float[0,1]", "confidence": "float[0,1]"}
                    for task in model_tasks
                },
                "findings": [{
                    "category": "open vocabulary string",
                    "severity": "info|warning|error",
                    "start_s": "number relative to clip",
                    "end_s": "number relative to clip",
                    "evidence": "short factual description",
                    "suggested_action": "accept|review|repair|reject",
                }],
                "missing_annotations": "list[string]",
                "summary": "short factual description",
            },
        })

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "review-queue.jsonl", review_rows)
    write_jsonl(output / "teacher-api-queue.jsonl", teacher_rows)
    tier_counts = {
        tier: sum(row["candidate_tier"] == tier for row in selected)
        for tier in limits
    }
    summary = {
        "schema_version": "egoqc-egodex-review-batch-v1",
        "profiles": str(profiles),
        "dataset": str(dataset),
        "output": str(output),
        "source_readonly": True,
        "items": len(selected),
        "tier_counts": tier_counts,
        "task_coverage": len({(row["partition"], row["task"]) for row in selected}),
        "window_s": window_s,
        "visual_evidence": "raw_video",
        "excluded_without_overlay": sorted(RAW_UNOBSERVABLE_TASKS),
        "teacher_api_called": False,
        "artifacts": {
            "review_queue": str(output / "review-queue.jsonl"),
            "teacher_api_queue": str(output / "teacher-api-queue.jsonl"),
        },
    }
    write_json(output / "summary.json", summary)
    return summary
