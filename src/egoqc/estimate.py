from __future__ import annotations

import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from .registry import _connect


def _duration_range(seconds: float) -> Dict[str, float]:
    return {
        "low_s": max(0.0, seconds * 0.75),
        "expected_s": max(0.0, seconds),
        "high_s": max(0.0, seconds * 1.40),
    }


def estimate_manifest(
    registry_path: Path,
    manifest_path: Path,
    config: Dict[str, Any],
    workers: int = 1,
) -> Dict[str, Any]:
    if workers < 1:
        raise ValueError("workers 必须大于等于 1")
    tasks = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset_ids = [task["dataset_id"] for task in tasks]
    kind_stats: Dict[str, Dict[str, int]] = {}
    history_rates: List[float] = []
    cache_ratios: List[float] = []
    db = _connect(registry_path)
    try:
        if dataset_ids:
            placeholders = ",".join("?" for _ in dataset_ids)
            for row in db.execute(
                f"""
                SELECT kind, COUNT(*) AS file_count,
                       COALESCE(SUM(CASE WHEN exists_flag=1 THEN size ELSE 0 END), 0) AS total_bytes
                FROM files WHERE dataset_id IN ({placeholders}) GROUP BY kind
                """,
                dataset_ids,
            ).fetchall():
                kind_stats[row["kind"]] = {
                    "file_count": int(row["file_count"]),
                    "total_bytes": int(row["total_bytes"]),
                }
        for row in db.execute(
            "SELECT summary_json FROM runs WHERE status='succeeded' AND summary_json IS NOT NULL"
        ).fetchall():
            try:
                summary = json.loads(row["summary_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            elapsed = float(summary.get("elapsed_s") or 0.0)
            logical_bytes = float(summary.get("logical_input_bytes") or 0.0)
            if elapsed > 0 and logical_bytes > 0:
                history_rates.append(logical_bytes / elapsed)
            ratios = [
                float(summary.get("parquet_cache_hit_ratio") or 0.0),
                float(summary.get("video_cache_hit_ratio") or 0.0),
            ]
            cache_ratios.append(sum(ratios) / len(ratios))
    finally:
        db.close()

    settings = config.get("estimation", {})
    fallback_mib_s = float(settings.get("fallback_logical_throughput_mib_s", 250.0))
    rate = statistics.median(history_rates) if history_rates else fallback_mib_s * 1024**2
    rate_source = "registry_history_median" if history_rates else "config_fallback"
    total_bytes = sum(int(task.get("total_bytes", 0)) for task in tasks)
    file_count = sum(int(task.get("file_count", 0)) for task in tasks)
    effective_workers = min(max(1, workers), max(1, len(tasks)))
    overhead_s = file_count * float(settings.get("file_overhead_s", 0.02))
    cold_s = (total_bytes / max(rate, 1.0) + overhead_s) / effective_workers
    warm_hit = (
        statistics.median(cache_ratios)
        if cache_ratios
        else float(settings.get("expected_warm_cache_hit_ratio", 0.70))
    )
    warm_hit = min(0.99, max(0.0, warm_hit))
    warm_s = cold_s * max(0.08, 1.0 - warm_hit * 0.85)
    now = datetime.now(timezone.utc)
    cold_range = _duration_range(cold_s)
    warm_range = _duration_range(warm_s)
    return {
        "manifest": str(manifest_path.resolve()),
        "registry": str(registry_path.resolve()),
        "task_count": len(tasks),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "workers": workers,
        "effective_dataset_workers": effective_workers,
        "logical_throughput_mib_s": rate / 1024**2,
        "throughput_source": rate_source,
        "history_samples": len(history_rates),
        "expected_warm_cache_hit_ratio": warm_hit,
        "generated_at": now.isoformat(),
        "cold": {
            **cold_range,
            "expected_finish_at": (now + timedelta(seconds=cold_range["expected_s"])).isoformat(),
        },
        "warm": {
            **warm_range,
            "expected_finish_at": (now + timedelta(seconds=warm_range["expected_s"])).isoformat(),
        },
        "by_kind": kind_stats,
        "notes": [
            "ETA 是逻辑吞吐估算；视频默认只探测容器头，不会完整解码。",
            "CPFS/OSS 首次冷读、并发限流和缓存命中会影响实际时间。",
        ],
    }
