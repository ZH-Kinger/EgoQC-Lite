from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Sequence, Tuple

from .gold_review import CAUSE_OPTIONS, GOLD_LABELS, ISSUE_DESCRIPTIONS, ISSUE_LABELS
from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-queue-gold-review-v1"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object")
            yield value


def _balanced(rows: Sequence[Dict[str, Any]], maximum: int, seed: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row.get("source_dataset") or "unknown_dataset"),
            str(row.get("selection_source") or "unknown_selection"),
        )].append(row)
    queues: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = {}
    for key, values in grouped.items():
        values.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row['request_id']}".encode("utf-8")
        ).hexdigest())
        queues[key] = deque(values)
    active = deque(sorted(queues))
    selected: List[Dict[str, Any]] = []
    while active and len(selected) < maximum:
        key = active.popleft()
        selected.append(queues[key].popleft())
        if queues[key]:
            active.append(key)
    return selected


def _signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _materialize(request: Dict[str, Any], media_root: Path) -> Dict[str, Any]:
    source = Path(str(request["source_uri"])).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    request_id = str(request["request_id"])
    output = media_root / f"{request_id}-raw.mp4"
    provenance_path = output.with_suffix(".provenance.json")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "source": str(source),
        "source_signature": _signature(source),
        "clip_start_s": float(request["clip_start_s"]),
        "clip_end_s": float(request["clip_end_s"]),
    }
    if output.is_file() and provenance_path.is_file():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if all(previous.get(key) == value for key, value in expected.items()):
            return {"request_id": request_id, "output": str(output), "cached": True}
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
    duration = expected["clip_end_s"] - expected["clip_start_s"]
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(expected["clip_start_s"]), "-i", str(source),
            "-t", str(duration), "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(result.stderr[-1000:] or f"ffmpeg 失败: {source}")
    temporary.replace(output)
    write_json(provenance_path, {
        **expected,
        "output": str(output),
        "raw_source_readonly": True,
        "derived_media": True,
        "cached": False,
        "code_version": code_version(),
    })
    return {"request_id": request_id, "output": str(output), "cached": False}


def build_queue_gold_review(
    queue: Path,
    output: Path,
    *,
    maximum_events: int = 180,
    seed: int = 23,
    materialize_media: bool = False,
    workers: int = 8,
) -> Dict[str, Any]:
    """Build a balanced human Gold batch from teacher candidates."""

    if maximum_events < 0 or workers < 1:
        raise ValueError("maximum_events/workers 参数非法")
    started = time.perf_counter()
    rows = list(_read_jsonl(queue))
    selected = _balanced(rows, maximum_events, seed)
    output = output.expanduser().resolve()
    media_root = output / "media"
    output.mkdir(parents=True, exist_ok=True)
    media_root.mkdir(exist_ok=True)
    media: Dict[str, str] = {}
    failures: List[Dict[str, str]] = []
    cached = 0
    if materialize_media:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_materialize, row, media_root): row for row in selected}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                    media[str(row["request_id"])] = str(result["output"])
                    cached += int(bool(result["cached"]))
                except Exception as error:
                    failures.append({
                        "request_id": str(row["request_id"]),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })

    events = []
    for row in selected:
        request_id = str(row["request_id"])
        if materialize_media and request_id not in media:
            continue
        clip_duration = float(row["clip_end_s"]) - float(row["clip_start_s"])
        clip_path = media.get(request_id, str(row["source_uri"]))
        event_codes = [str(value) for value in (row.get("event_codes") or [])]
        selection_source = str(row.get("selection_source") or "unknown_selection")
        events.append({
            "event_id": f"queue-gold--{request_id}",
            "video_id": request_id,
            "kind": "episode_qc_gold_review",
            "category": "human_gold",
            "severity": "review",
            "start_s": 0.0 if materialize_media else float(row["clip_start_s"]),
            "end_s": clip_duration if materialize_media else float(row["clip_end_s"]),
            "duration_s": clip_duration,
            "clip": clip_path,
            "source_uri": str(row["source_uri"]),
            "priority": 20 if selection_source == "deterministic_bad_frame" else 10,
            "review_mode": "episode_gold",
            "schema_version": SCHEMA_VERSION,
            "source_class": row.get("source_class"),
            "source_dataset": row.get("source_dataset"),
            "supplier_id": row.get("supplier_id"),
            "parent_episode_index": row.get("parent_episode_index"),
            "tasks": row.get("tasks") or [],
            "selection_source": selection_source,
            "trigger_tasks": row.get("trigger_tasks") or [],
            "issue_codes": event_codes,
            "issue_labels": {code: ISSUE_LABELS.get(code, code) for code in event_codes},
            "issue_descriptions": {
                code: ISSUE_DESCRIPTIONS.get(
                    code, "机器规则召回该片段，请区分真实快速动作与追踪异常。"
                )
                for code in event_codes
            },
            "sample_frames": [
                int(frame) - int(row.get("clip_start_frame") or 0)
                for frame in (row.get("event_frames") or [])
                if int(frame) >= int(row.get("clip_start_frame") or 0)
            ],
            "fps": float(row.get("fps") or 30.0),
            "baseline_tier": "bronze" if selection_source == "deterministic_bad_frame" else "silver",
            "gold_labels": GOLD_LABELS,
            "cause_options": CAUSE_OPTIONS,
            "raw_clip_path": clip_path,
            "original_clip_start_s": float(row["clip_start_s"]),
            "original_clip_end_s": float(row["clip_end_s"]),
            "split_group": row.get("split_group"),
            "raw_source_readonly": True,
            "derived_media": materialize_media,
        })
    artifact = output / "review-events.jsonl"
    write_jsonl(artifact, events)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "queue": str(queue.expanduser().resolve()),
        "input_requests": len(rows),
        "selected_requests": len(selected),
        "review_events": len(events),
        "materialize_media": materialize_media,
        "media_cached": cached,
        "media_created": len(media) - cached,
        "media_failures": failures,
        "source_counts": dict(Counter(str(row.get("source_dataset")) for row in selected)),
        "selection_counts": dict(Counter(str(row.get("selection_source")) for row in selected)),
        "elapsed_s": time.perf_counter() - started,
        "raw_source_readonly": True,
        "review_events_path": str(artifact),
    }
    write_json(output / "summary.json", summary)
    return summary
