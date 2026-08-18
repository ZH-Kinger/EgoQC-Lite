from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .adapters import inspect_adapter
from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-adapter-clip-candidates-v1"


def _request_id(episode: str, start_s: float, end_s: float) -> str:
    identity = f"{episode}:{start_s:.6f}:{end_s:.6f}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"adapter-{digest}"


def plan_adapter_clips(
    dataset: Path,
    episode: str,
    output: Path,
    task_config: Path,
    *,
    window_s: float = 6.0,
    maximum_clips: Optional[int] = 3,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Create bounded visual-teacher requests from a canonical readonly adapter."""

    if window_s <= 0:
        raise ValueError("window_s 必须大于 0")
    if maximum_clips is not None and maximum_clips < 1:
        raise ValueError("maximum_clips 必须大于 0")
    report = inspect_adapter(
        dataset.expanduser().resolve(),
        episode,
        confidence_threshold=confidence_threshold,
    )
    canonical = report.get("canonical")
    if not isinstance(canonical, dict):
        raise ValueError("该 adapter 尚未提供 CanonicalEpisode，不能生成教师队列")
    video = canonical.get("video") or {}
    source_uri = str(video.get("path") or "")
    duration_s = float(canonical.get("duration_s") or 0.0)
    if not source_uri or duration_s <= 0:
        raise ValueError("CanonicalEpisode 缺少视频路径或有效时长")

    config = json.loads(task_config.read_text(encoding="utf-8"))
    model_tasks = list((config.get("model_tasks") or {}).keys())
    dimensions = dict(config.get("assessment_dimensions") or {})
    if not model_tasks:
        raise ValueError("task config 没有 model_tasks")

    count = max(1, int((duration_s + window_s - 1e-9) // window_s))
    if duration_s > window_s and count == 1:
        count = 2
    if maximum_clips is not None:
        count = min(count, int(maximum_clips))
    count = max(1, count)
    starts = [0.0] if count == 1 else [
        index * max(0.0, duration_s - window_s) / (count - 1)
        for index in range(count)
    ]

    clip_rows = []
    api_rows = []
    output = output.expanduser().resolve()
    for start_s in starts:
        end_s = min(duration_s, start_s + window_s)
        request_id = _request_id(str(canonical["episode_id"]), start_s, end_s)
        clip_rows.append({
            "schema_version": SCHEMA_VERSION,
            "clip_id": request_id,
            "episode_id": canonical["episode_id"],
            "source_format": canonical.get("source_format"),
            "source_uri": source_uri,
            "clip_start_s": start_s,
            "clip_end_s": end_s,
            "selection_source": "adapter_uniform_control",
            "labels": canonical.get("labels") or {},
            "capabilities": canonical.get("capabilities") or {},
            "provenance": {"raw_immutable": True, "code_version": code_version()},
        })
        api_rows.append({
            "request_id": request_id,
            "schema_version": "egoqc-visual-teacher-request-v1",
            "prompt_version": "egoqc-visual-teacher-v3-open-world",
            "source_uri": source_uri,
            "clip_start_s": start_s,
            "clip_end_s": end_s,
            "candidate_tasks": model_tasks,
            "trigger_tasks": [],
            "event_codes": [],
            "selection_source": "adapter_uniform_control",
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
    write_jsonl(output / "clip-candidates.jsonl", clip_rows)
    write_jsonl(output / "teacher-api-queue.jsonl", api_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset.expanduser().resolve()),
        "episode": canonical["episode_id"],
        "source_format": canonical.get("source_format"),
        "duration_s": duration_s,
        "clips": len(clip_rows),
        "teacher_api_requests": len(api_rows),
        "window_s": window_s,
        "raw_immutable": True,
        "teacher_api_queue": str(output / "teacher-api-queue.jsonl"),
    }
    write_json(output / "summary.json", summary)
    return summary
