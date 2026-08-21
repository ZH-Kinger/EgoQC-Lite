from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .report import write_json, write_jsonl


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "config" / "task_taxonomy.json"


def load_task_taxonomy(path: Optional[Path] = None) -> Dict[str, Any]:
    taxonomy_path = (path or DEFAULT_TAXONOMY_PATH).expanduser().resolve()
    return json.loads(taxonomy_path.read_text(encoding="utf-8"))


def compose_task_text(*values: Any) -> str:
    """Flatten coarse source metadata into one honest, classifiable text field."""

    parts: List[str] = []

    def visit(value: Any) -> None:
        if value in (None, "", [], {}):
            return
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key])
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
        else:
            text = str(value).strip()
            if text and text not in parts:
                parts.append(text)

    for value in values:
        visit(value)
    return "；".join(parts) if parts else "unknown"


def _normalize(value: str) -> str:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    value = value.replace("_", " ").replace("-", " ").lower()
    return " ".join(value.split())


def _matches(text: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        normalized = _normalize(pattern)
        if not normalized:
            continue
        if re.search(r"[a-z0-9]", normalized):
            if re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", text):
                return True
        elif normalized in text:
            return True
    return False


def classify_task(task: str, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize(task)
    domains = sorted(
        name for name, patterns in taxonomy.get("task_domains", {}).items()
        if _matches(normalized, patterns)
    )
    primitives = sorted(
        name for name, patterns in taxonomy["interaction_primitives"].items()
        if _matches(normalized, patterns)
    )
    affordances = sorted(
        name for name, patterns in taxonomy["object_affordances"].items()
        if _matches(normalized, patterns)
    )
    unknown = not primitives
    fine = bool(set(primitives) & set(taxonomy["fine_manipulation_primitives"]))
    likely_bimanual = bool(
        set(primitives) & set(taxonomy["likely_bimanual_primitives"])
    )
    conjunction = bool(re.search(r"\b(and|then|after|before)\b|以及|然后|再", normalized))
    if len(primitives) >= 3 or conjunction:
        complexity = "composite"
    elif primitives:
        complexity = "atomic_or_short_sequence"
    else:
        complexity = "unknown"
    return {
        "schema_version": taxonomy["schema_version"],
        "task_text": task,
        "normalized_task": normalized,
        "task_domains": domains or ["unknown"],
        "interaction_primitives": primitives or ["unknown"],
        "object_affordances": affordances or ["unknown"],
        "manipulation_scale": "fine" if fine else "gross_or_unknown",
        "hand_mode": "likely_bimanual" if likely_bimanual else "unknown",
        "temporal_complexity": complexity,
        "scene_type": "unknown",
        "classification_source": "deterministic_lexical_rules",
        "confidence": 0.9 if primitives else 0.0,
        "requires_semantic_review": unknown,
        "warning": (
            "task text alone cannot prove scene, hand mode, or observed execution"
        ),
    }


def classify_task_records(
    input_path: Path,
    taxonomy_path: Path,
    output: Path,
    *,
    task_field: str,
    source_id: str,
) -> Dict[str, Any]:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    unique: Dict[str, Dict[str, Any]] = {}
    enriched: List[Dict[str, Any]] = []
    missing = 0
    for row in rows:
        raw = row.get(task_field)
        if raw in (None, ""):
            missing += 1
            task = "unknown"
        elif isinstance(raw, list):
            values = [str(value).strip() for value in raw if str(value).strip()]
            if not values:
                missing += 1
                task = "unknown"
            else:
                task = "；".join(values)
        else:
            task = str(raw)
        label = unique.setdefault(task, classify_task(task, taxonomy))
        enriched.append({**row, "task_taxonomy": label, "taxonomy_source_id": source_id})

    labels = sorted(unique.values(), key=lambda row: row["normalized_task"])
    allowed_primitives = sorted(taxonomy["interaction_primitives"])
    semantic_review = [
        {
            "schema_version": "egoqc-task-semantic-review-v1",
            "request_id": "task-" + hashlib.sha256(
                f"{source_id}:{label['normalized_task']}".encode()
            ).hexdigest()[:16],
            "source_id": source_id,
            "task_text": label["task_text"],
            "normalized_task": label["normalized_task"],
            "allowed_interaction_primitives": allowed_primitives,
            "required_response": {
                "interaction_primitives": "list of allowed values or proposed_new_primitive",
                "object_affordances": "list[string]",
                "manipulation_scale": "fine|gross|unknown",
                "hand_mode": "likely_unimanual|likely_bimanual|unknown",
                "temporal_complexity": "atomic|short_sequence|composite|unknown",
                "confidence": "float[0,1]",
                "reason": "short evidence from task text only",
            },
            "video_required": False,
            "status": "pending",
        }
        for label in labels if label["requires_semantic_review"]
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "task-taxonomy.jsonl", labels)
    write_jsonl(output / "records-with-taxonomy.jsonl", enriched)
    write_jsonl(output / "semantic-review-tasks.jsonl", semantic_review)
    primitive_counts = Counter(
        primitive for label in labels for primitive in label["interaction_primitives"]
    )
    domain_counts = Counter(
        domain for label in labels for domain in label["task_domains"]
    )
    summary = {
        "schema_version": "egoqc-task-taxonomy-run-v1",
        "source_id": source_id,
        "input": str(input_path),
        "records": len(rows),
        "unique_tasks": len(labels),
        "missing_task_field": missing,
        "classified_tasks": sum(not row["requires_semantic_review"] for row in labels),
        "semantic_review_tasks": sum(row["requires_semantic_review"] for row in labels),
        "primitive_counts_unique_tasks": dict(primitive_counts.most_common()),
        "domain_counts_unique_tasks": dict(domain_counts.most_common()),
        "taxonomy": str(output / "task-taxonomy.jsonl"),
        "enriched_records": str(output / "records-with-taxonomy.jsonl"),
        "semantic_review_queue": str(output / "semantic-review-tasks.jsonl"),
        "semantic_review_video_required": False,
    }
    write_json(output / "summary.json", summary)
    return summary
