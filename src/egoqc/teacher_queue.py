from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-merged-teacher-queue-v1"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")
            yield value


def _rank(request_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{request_id}".encode("utf-8")).hexdigest()


def normalize_teacher_queue_provenance(
    queue: Path,
    output: Path,
    *,
    source_class: Optional[str] = None,
    source_dataset: Optional[str] = None,
    supplier_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Upgrade legacy teacher queues with raw-video provenance and split groups."""

    rows: List[Dict[str, Any]] = []
    inferred = Counter()
    for source_row in _read_jsonl(queue):
        row = copy.deepcopy(source_row)
        request_id = str(row.get("request_id") or "")
        if not request_id:
            raise ValueError("教师队列包含缺少 request_id 的记录")
        raw_source = str(row.get("raw_source_uri") or row.get("source_uri") or "")
        if not raw_source:
            raise ValueError(f"request_id={request_id} 缺少 source_uri")

        dataset_value = source_dataset or row.get("source_dataset")
        if not dataset_value and request_id.startswith("egodex-"):
            dataset_value = "egodex"
            inferred["source_dataset"] += 1
        if not dataset_value:
            raise ValueError(
                f"request_id={request_id} 缺少 source_dataset；请通过参数明确指定"
            )
        dataset_value = str(dataset_value)

        class_value = source_class or row.get("source_class")
        if not class_value and dataset_value == "egodex":
            class_value = "public_dataset"
            inferred["source_class"] += 1
        if not class_value:
            raise ValueError(
                f"request_id={request_id} 缺少 source_class；请通过参数明确指定"
            )
        class_value = str(class_value)

        supplier_value = supplier_id or row.get("supplier_id")
        if not supplier_value and class_value == "public_dataset":
            supplier_value = f"public:{dataset_value}"
            inferred["supplier_id"] += 1

        split_group = row.get("split_group")
        if not split_group:
            digest = hashlib.sha256(raw_source.encode("utf-8")).hexdigest()[:20]
            split_group = f"{dataset_value}:raw-video:{digest}"
            inferred["split_group"] += 1

        raw_tasks = row.get("tasks")
        if not raw_tasks:
            task = str(row.get("task") or "").strip()
            raw_tasks = [task] if task else []
            if raw_tasks:
                inferred["tasks"] += 1

        row.update({
            "source_class": class_value,
            "source_dataset": dataset_value,
            "supplier_id": supplier_value,
            "raw_source_uri": raw_source,
            "split_group": str(split_group),
            "split_group_source": str(
                row.get("split_group_source") or "raw_source_uri"
            ),
            "tasks": raw_tasks,
            "task_id": row.get("task_id") or row.get("task"),
            "source_readonly": True,
            "raw_source_readonly": True,
        })
        rows.append(row)

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "teacher-api-queue.jsonl"
    write_jsonl(artifact, rows)
    summary = {
        "schema_version": "egoqc-normalized-teacher-queue-v1",
        "input_queue": str(queue.expanduser().resolve()),
        "requests": len(rows),
        "source_counts": dict(
            Counter(str(row["source_dataset"]) for row in rows)
        ),
        "source_class_counts": dict(
            Counter(str(row["source_class"]) for row in rows)
        ),
        "selection_counts": dict(
            Counter(str(row.get("selection_source") or "unknown_selection") for row in rows)
        ),
        "split_groups": len({str(row["split_group"]) for row in rows}),
        "inferred_field_counts": dict(inferred),
        "raw_source_readonly": True,
        "code_version": code_version(),
        "queue": str(artifact),
    }
    write_json(output / "summary.json", summary)
    return summary


def merge_teacher_queues(
    queues: Sequence[Path],
    output: Path,
    *,
    maximum_requests: Optional[int] = None,
    seed: int = 17,
) -> Dict[str, Any]:
    """Deduplicate and round-robin teacher requests across source and recall strata."""

    if not queues:
        raise ValueError("至少需要一个教师队列")
    if maximum_requests is not None and maximum_requests < 0:
        raise ValueError("maximum_requests 不能为负数")
    unique: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    for queue in queues:
        for row in _read_jsonl(queue):
            request_id = str(row.get("request_id") or "")
            if not request_id:
                raise ValueError(f"{queue} 包含缺少 request_id 的记录")
            previous = unique.get(request_id)
            if previous is not None:
                if previous != row:
                    raise ValueError(f"request_id={request_id} 在队列间冲突")
                duplicates += 1
                continue
            unique[request_id] = row

    strata: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = {}
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        key = (
            str(row.get("source_dataset") or "unknown_dataset"),
            str(row.get("selection_source") or "unknown_selection"),
        )
        grouped[key].append(row)
    for key, rows in grouped.items():
        rows.sort(key=lambda row: _rank(str(row["request_id"]), seed))
        strata[key] = deque(rows)

    selected: List[Dict[str, Any]] = []
    limit = len(unique) if maximum_requests is None else maximum_requests
    active = deque(sorted(strata))
    while active and len(selected) < limit:
        key = active.popleft()
        values = strata[key]
        selected.append(values.popleft())
        if values:
            active.append(key)

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "teacher-api-queue.jsonl"
    write_jsonl(artifact, selected)
    source_counts = Counter(str(row.get("source_dataset") or "unknown_dataset") for row in selected)
    source_class_counts = Counter(str(row.get("source_class") or "unknown") for row in selected)
    selection_counts = Counter(str(row.get("selection_source") or "unknown") for row in selected)
    trigger_counts = Counter(
        str(task)
        for row in selected
        for task in (row.get("trigger_tasks") or [])
    )
    supplier_sources = sorted({
        str(row.get("source_dataset") or "unknown_dataset")
        for row in selected
        if row.get("source_class") == "supplier_dataset"
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_queues": [str(path.expanduser().resolve()) for path in queues],
        "input_unique_requests": len(unique),
        "duplicate_requests": duplicates,
        "selected_requests": len(selected),
        "maximum_requests": maximum_requests,
        "seed": seed,
        "source_counts": dict(source_counts),
        "source_class_counts": dict(source_class_counts),
        "selection_counts": dict(selection_counts),
        "trigger_task_counts": dict(trigger_counts),
        "split_groups": len({str(row.get("split_group") or row["request_id"]) for row in selected}),
        "external_transfer": {
            "contains_supplier_data": bool(supplier_sources),
            "supplier_sources": supplier_sources,
            "requires_explicit_runtime_authorization": bool(supplier_sources),
        },
        "raw_source_readonly": True,
        "code_version": code_version(),
        "queue": str(artifact),
    }
    write_json(output / "summary.json", summary)
    return summary


def build_overlay_teacher_queue(
    queue: Path,
    review_events: Path,
    output: Path,
) -> Dict[str, Any]:
    """Bind derived MANO overlay clips to teacher requests without losing raw provenance."""

    requests = {str(row["request_id"]): row for row in _read_jsonl(queue)}
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    missing_requests = []
    missing_overlays = []
    for event in _read_jsonl(review_events):
        request_id = str(event.get("video_id") or "")
        request = requests.get(request_id)
        if request is None:
            missing_requests.append(request_id)
            continue
        overlay = Path(str(event.get("annotated_clip_path") or ""))
        if not overlay.is_file():
            missing_overlays.append(request_id)
            continue
        row = copy.deepcopy(request)
        original_start = float(row["clip_start_s"])
        original_end = float(row["clip_end_s"])
        duration = original_end - original_start
        row["raw_source_uri"] = str(row.get("raw_source_uri") or row["source_uri"])
        row["source_uri"] = str(overlay.resolve())
        row["source_clip_start_s"] = original_start
        row["source_clip_end_s"] = original_end
        row["clip_start_s"] = 0.0
        row["clip_end_s"] = duration
        row["duration_s"] = duration
        row["visual_evidence"] = "mano_mesh_skeleton_overlay"
        row["overlay_artifact"] = str(overlay.resolve())
        row["prompt_version"] = "egoqc-visual-teacher-v4-mano-overlay"
        row["output_path"] = str(
            output / "teacher-labels" / request_id / "teacher-label.json"
        )
        trigger_tasks = set(str(value) for value in (row.get("trigger_tasks") or []))
        trigger_tasks.add("mano_overlay_drift")
        row["trigger_tasks"] = sorted(trigger_tasks)
        rows.append(row)

    artifact = output / "teacher-api-queue.jsonl"
    write_jsonl(artifact, rows)
    summary = {
        "schema_version": "egoqc-overlay-teacher-queue-v1",
        "input_queue": str(queue.expanduser().resolve()),
        "review_events": str(review_events.expanduser().resolve()),
        "requests": len(rows),
        "missing_requests": missing_requests,
        "missing_overlays": missing_overlays,
        "source_counts": dict(Counter(str(row.get("source_dataset")) for row in rows)),
        "visual_evidence": "mano_mesh_skeleton_overlay",
        "raw_source_readonly": True,
        "derived_media_only": True,
        "external_transfer": {
            "contains_supplier_data": any(
                row.get("source_class") == "supplier_dataset" for row in rows
            ),
            "requires_explicit_runtime_authorization": any(
                row.get("source_class") == "supplier_dataset" for row in rows
            ),
        },
        "queue": str(artifact),
    }
    write_json(output / "summary.json", summary)
    return summary
