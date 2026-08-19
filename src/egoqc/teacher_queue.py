from __future__ import annotations

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
