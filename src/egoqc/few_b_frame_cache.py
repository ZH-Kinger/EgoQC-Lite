from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .few_b_benchmark import (
    _clip_window,
    _prefetched_decodes,
    _read_jsonl,
    select_benchmark_rows,
)
from .report import write_json, write_jsonl
from .storage_safety import assert_derived_output, raw_file_stamp


SCHEMA_VERSION = "egoqc-few-b-frame-cache-v1"


def _reusable_record(
    record: Dict[str, Any],
    row: Dict[str, Any],
    output: Path,
    *,
    frame_count: int,
    maximum_edge: int,
    jpeg_quality: int,
) -> bool:
    source = Path(str(row.get("source_uri") or "")).expanduser().resolve()
    if not source.is_file():
        return False
    start_s, end_s = _clip_window(row)
    stamp = raw_file_stamp(source)
    expected_stamp = {
        "size": stamp.size,
        "mtime_ns": stamp.mtime_ns,
        "inode": stamp.inode,
        "device": stamp.device,
    }
    frames = [(output / value).resolve() for value in record.get("frames") or []]
    return (
        record.get("source_uri_sha256")
        == hashlib.sha256(str(source).encode("utf-8")).hexdigest()
        and record.get("source_file_stamp") == expected_stamp
        and abs(float(record.get("clip_start_s", -1)) - start_s) <= 1e-6
        and abs(float(record.get("clip_end_s", -1)) - end_s) <= 1e-6
        and int(record.get("frame_count") or 0) == frame_count
        and int(record.get("maximum_edge") or 0) == maximum_edge
        and int(record.get("jpeg_quality") or 0) == jpeg_quality
        and len(frames) == frame_count
        and all(output in path.parents and path.is_file() for path in frames)
    )


def predecode_few_b_frame_cache(
    manifest: Path,
    output: Path,
    *,
    maximum_clips: int = 434,
    frame_count: int = 8,
    maximum_edge: int = 448,
    jpeg_quality: int = 82,
    workers: int = 8,
    seed: int = 20260820,
    selection_strategy: str = "stable_random",
    resume: bool = True,
) -> Dict[str, Any]:
    if not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality must be in 1..95")
    output = assert_derived_output(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = select_benchmark_rows(
        _read_jsonl(manifest.expanduser().resolve()),
        maximum_clips,
        seed,
        strategy=selection_strategy,
    )
    existing: Dict[str, Dict[str, Any]] = {}
    index_path = output / "index.jsonl"
    if resume and index_path.is_file():
        selected_by_id = {
            str(row.get("video_id") or row.get("request_id") or row.get("record_id") or ""): row
            for row in rows
        }
        for record in _read_jsonl(index_path):
            identity = str(record.get("video_id") or "")
            selected = selected_by_id.get(identity)
            if selected is not None and _reusable_record(
                record,
                selected,
                output,
                frame_count=frame_count,
                maximum_edge=maximum_edge,
                jpeg_quality=jpeg_quality,
            ):
                existing[identity] = record
    pending = [
        row
        for row in rows
        if str(row.get("video_id") or row.get("request_id") or row.get("record_id") or "")
        not in existing
    ]
    records: List[Dict[str, Any]] = list(existing.values())
    started = time.perf_counter()
    new_records = 0
    for row, source, start_s, end_s, frames, decode_seconds, _, source_stamp in _prefetched_decodes(
        pending,
        frame_count,
        workers,
        maximum_edge=maximum_edge,
    ):
        identity = str(row.get("video_id") or row.get("request_id") or row.get("record_id") or "")
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        clip_dir = output / "clips" / key
        clip_dir.mkdir(parents=True, exist_ok=True)
        relative_frames = []
        encoded_bytes = 0
        try:
            for index, image in enumerate(frames):
                cached = image.copy()
                cached.thumbnail((maximum_edge, maximum_edge), Image.Resampling.LANCZOS)
                destination = clip_dir / f"{index:03d}.jpg"
                temporary = destination.with_suffix(".jpg.tmp")
                cached.save(temporary, format="JPEG", quality=jpeg_quality, optimize=True)
                temporary.replace(destination)
                cached.close()
                encoded_bytes += destination.stat().st_size
                relative_frames.append(destination.relative_to(output).as_posix())
        finally:
            for image in frames:
                image.close()
        record = {
            "schema_version": SCHEMA_VERSION,
            "video_id": identity,
            "source_uri_sha256": hashlib.sha256(str(source).encode("utf-8")).hexdigest(),
            "source_file_stamp": source_stamp,
            "clip_start_s": start_s,
            "clip_end_s": end_s,
            "frame_count": frame_count,
            "maximum_edge": maximum_edge,
            "jpeg_quality": jpeg_quality,
            "frames": relative_frames,
            "encoded_bytes": encoded_bytes,
            "decode_seconds": decode_seconds,
        }
        records.append(record)
        new_records += 1
        write_jsonl(index_path, records)
        elapsed = time.perf_counter() - started
        write_json(
            output / "progress.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "cached_clips": len(records),
                "target_clips": len(rows),
                "resumed_clips": len(existing),
                "new_clips": new_records,
                "elapsed_seconds": elapsed,
                "average_seconds_per_new_clip": elapsed / new_records,
                "eta_seconds": elapsed / new_records * (len(rows) - len(records)),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    total_elapsed = time.perf_counter() - started
    total_bytes = sum(int(row.get("encoded_bytes") or 0) for row in records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "cached_clips": len(records),
        "target_clips": len(rows),
        "resumed_clips": len(existing),
        "new_clips": new_records,
        "elapsed_seconds": total_elapsed,
        "average_seconds_per_new_clip": (
            total_elapsed / new_records if new_records else None
        ),
        "frame_count": frame_count,
        "maximum_edge": maximum_edge,
        "jpeg_quality": jpeg_quality,
        "workers": workers,
        "encoded_bytes": total_bytes,
        "raw_source_readonly": True,
        "source_paths_stored": False,
        "index": str(index_path),
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "progress.json",
        {**summary, "updated_at": datetime.now(timezone.utc).isoformat()},
    )
    return summary
