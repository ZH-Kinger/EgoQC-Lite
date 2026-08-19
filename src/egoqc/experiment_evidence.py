from __future__ import annotations

import hashlib
import html
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .provenance import code_version
from .report import write_json, write_jsonl


SCHEMA_VERSION = "egoqc-experiment-evidence-v1"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _stable_rank(row: Dict[str, Any], seed: int) -> str:
    identity = str(row.get("event_id") or row.get("video_id") or "")
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()


def _bad_ratio(row: Dict[str, Any]) -> float:
    return float((row.get("metrics") or {}).get("bad_frame_ratio") or 0.0)


def select_evidence_events(
    rows: Sequence[Dict[str, Any]],
    *,
    maximum_rule_positive: int = 12,
    maximum_clean_control: int = 6,
    maximum_low_event_control: int = 6,
    seed: int = 43,
) -> List[Dict[str, Any]]:
    """Select deterministic contrast strata without treating rules as Gold."""

    if min(maximum_rule_positive, maximum_clean_control, maximum_low_event_control) < 0:
        raise ValueError("evidence limits cannot be negative")
    available = [
        row
        for row in rows
        if row.get("clip") and row.get("annotated_clip_path")
    ]
    positive = [
        row for row in available
        if row.get("selection_source") == "deterministic_bad_frame"
    ]
    clean = [
        row for row in available
        if row.get("selection_source") == "deterministic_clean_gap_control"
    ]
    low_event = [
        row for row in available
        if row.get("selection_source") == "deterministic_low_event_control"
    ]
    positive.sort(key=lambda row: (-_bad_ratio(row), _stable_rank(row, seed)))
    clean.sort(key=lambda row: (_bad_ratio(row), _stable_rank(row, seed + 1)))
    low_event.sort(key=lambda row: (_bad_ratio(row), _stable_rank(row, seed + 2)))
    selected: List[Dict[str, Any]] = []
    for bucket, values, limit in (
        ("rule-positive-high", positive, maximum_rule_positive),
        ("clean-control", clean, maximum_clean_control),
        ("low-event-control", low_event, maximum_low_event_control),
    ):
        for row in values[:limit]:
            selected.append({**row, "evidence_bucket": bucket})
    return selected


def _run(command: List[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-1500:] or "command failed")


def _comparison_job(row: Dict[str, Any], index: int, case_root: Path) -> Dict[str, Any]:
    event_id = str(row.get("event_id") or row.get("video_id") or f"case-{index}")
    digest = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:12]
    slug = f"case-{index:03d}-{digest}"
    raw = Path(str(row["clip"])).expanduser().resolve()
    overlay = Path(str(row["annotated_clip_path"])).expanduser().resolve()
    if not raw.is_file() or not overlay.is_file():
        raise FileNotFoundError(f"missing raw/overlay media for {event_id}")
    video = case_root / f"{slug}.mp4"
    image = case_root / f"{slug}.jpg"
    filter_with_labels = (
        "[0:v]scale=640:360:force_original_aspect_ratio=decrease,"
        "pad=640:360:(ow-iw)/2:(oh-ih)/2:black,"
        "drawtext=text=RAW:x=16:y=16:fontsize=24:fontcolor=white:"
        "box=1:boxcolor=black@0.65[left];"
        "[1:v]scale=640:360:force_original_aspect_ratio=decrease,"
        "pad=640:360:(ow-iw)/2:(oh-ih)/2:black,"
        "drawtext=text=MANO_OVERLAY:x=16:y=16:fontsize=24:fontcolor=white:"
        "box=1:boxcolor=black@0.65[right];"
        "[left][right]hstack=inputs=2:shortest=1[out]"
    )
    filter_without_labels = (
        "[0:v]scale=640:360:force_original_aspect_ratio=decrease,"
        "pad=640:360:(ow-iw)/2:(oh-ih)/2:black[left];"
        "[1:v]scale=640:360:force_original_aspect_ratio=decrease,"
        "pad=640:360:(ow-iw)/2:(oh-ih)/2:black[right];"
        "[left][right]hstack=inputs=2:shortest=1[out]"
    )
    base = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(raw), "-i", str(overlay), "-filter_complex",
    ]
    tail = [
        "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ]
    try:
        _run(base + [filter_with_labels] + tail)
    except RuntimeError:
        _run(base + [filter_without_labels] + tail)
    midpoint = max(0.0, float(row.get("duration_s") or 0.0) / 2.0)
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{midpoint:.6f}", "-i", str(video), "-frames:v", "1",
        "-q:v", "2", str(image),
    ])
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "case_id": slug,
        "event_id": event_id,
        "video_id": row.get("video_id"),
        "evidence_bucket": row["evidence_bucket"],
        "human_label_status": "pending" if not row.get("decision") else "available",
        "human_decision": row.get("decision"),
        "selection_source": row.get("selection_source"),
        "issue_codes": row.get("issue_codes") or [],
        "tasks": row.get("tasks") or [],
        "bad_frame_ratio": _bad_ratio(row),
        "clip_start_frame": row.get("clip_start_frame"),
        "clip_end_frame": row.get("clip_end_frame"),
        "duration_s": row.get("duration_s"),
        "source_dataset": row.get("source_dataset"),
        "source_uri_sha256": hashlib.sha256(
            str(row.get("source_uri") or "").encode("utf-8")
        ).hexdigest(),
        "raw_clip_sha256": _sha256(raw),
        "overlay_clip_sha256": _sha256(overlay),
        "comparison_video": video.name,
        "comparison_video_sha256": _sha256(video),
        "comparison_image": image.name,
        "comparison_image_sha256": _sha256(image),
    }
    write_json(case_root / f"{slug}.json", metadata)
    return metadata


def _git_commit(code_root: Optional[Path]) -> Optional[str]:
    if not code_root:
        return None
    result = subprocess.run(
        ["git", "-C", str(code_root.expanduser().resolve()), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _tool_version(command: List[str]) -> Optional[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return (result.stdout or result.stderr).splitlines()[0]


def _write_gallery(output: Path, cases: Sequence[Dict[str, Any]]) -> None:
    sections = []
    labels = {
        "rule-positive-high": "规则高风险片段",
        "clean-control": "严格干净对照",
        "low-event-control": "低事件对照",
    }
    for bucket in labels:
        cards = []
        for case in [value for value in cases if value["evidence_bucket"] == bucket]:
            tasks = " / ".join(str(value) for value in case.get("tasks") or [])
            issues = ", ".join(str(value) for value in case.get("issue_codes") or []) or "control"
            cards.append(
                f'<article><img loading="lazy" src="cases/{html.escape(case["comparison_image"])}" '
                f'alt="{html.escape(case["case_id"])} comparison frame">'
                f'<video controls preload="none" poster="cases/{html.escape(case["comparison_image"])}" '
                f'src="cases/{html.escape(case["comparison_video"])}"></video>'
                f'<h3>{html.escape(tasks or case["case_id"])}</h3>'
                f'<p><code>{html.escape(issues)}</code></p>'
                f'<p>坏帧率 {float(case["bad_frame_ratio"]):.2%} · 人工标签待确认</p></article>'
            )
        sections.append(f"<section><h2>{labels[bucket]}</h2><div class=grid>{''.join(cards)}</div></section>")
    document = f"""<!doctype html><html lang=zh-CN><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>EgoQC 实验证据</title>
<style>body{{margin:0;background:#f4f4ef;color:#171b18;font:14px system-ui,sans-serif}}main{{max-width:1420px;margin:auto;padding:30px}}h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:32px;border-bottom:1px solid #cfd4ce;padding-bottom:10px}}.muted,p{{color:#626963}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}article{{background:#fff;border:1px solid #d9ddd8;padding:12px}}img,video{{display:block;width:100%;background:#111}}img{{margin-bottom:8px}}h3{{margin:12px 0 6px;font-size:15px}}code{{font-size:12px}}</style></head><body><main>
<h1>EgoQC 实验证据对比</h1><p class=muted>左侧 raw，右侧 MANO overlay。机器分层不是人工真值。</p>
{''.join(sections)}</main></body></html>"""
    temporary = output / f".index.{os.getpid()}.tmp.html"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output / "index.html")


def build_experiment_evidence(
    events_path: Path,
    output: Path,
    *,
    experiment_id: str,
    code_root: Optional[Path] = None,
    config_path: Optional[Path] = None,
    run_results_path: Optional[Path] = None,
    maximum_rule_positive: int = 12,
    maximum_clean_control: int = 6,
    maximum_low_event_control: int = 6,
    workers: int = 4,
    seed: int = 43,
) -> Dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    started = time.perf_counter()
    events_path = events_path.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    case_root = output / "cases"
    case_root.mkdir(exist_ok=True)
    selected = select_evidence_events(
        _read_jsonl(events_path),
        maximum_rule_positive=maximum_rule_positive,
        maximum_clean_control=maximum_clean_control,
        maximum_low_event_control=maximum_low_event_control,
        seed=seed,
    )
    cases: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_comparison_job, row, index, case_root): row
            for index, row in enumerate(selected, 1)
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                cases.append(future.result())
            except Exception as error:
                failures.append({
                    "event_id": str(row.get("event_id") or row.get("video_id")),
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
    cases.sort(key=lambda row: row["case_id"])
    write_jsonl(output / "cases.jsonl", cases)
    _write_gallery(output, cases)
    input_files = {"events": events_path}
    if config_path:
        input_files["config"] = config_path.expanduser().resolve()
    if run_results_path:
        input_files["run_results"] = run_results_path.expanduser().resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version(),
        "git_commit": _git_commit(code_root),
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in input_files.items() if path.is_file()
        },
        "parameters": {
            "maximum_rule_positive": maximum_rule_positive,
            "maximum_clean_control": maximum_clean_control,
            "maximum_low_event_control": maximum_low_event_control,
            "workers": workers,
            "seed": seed,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "ffmpeg": _tool_version(["ffmpeg", "-version"]),
        },
        "selected": len(selected),
        "created": len(cases),
        "failures": failures,
        "bucket_counts": dict(Counter(row["evidence_bucket"] for row in cases)),
        "human_label_status": "pending",
        "raw_source_readonly": True,
        "external_api_called": False,
        "elapsed_s": time.perf_counter() - started,
        "gallery": str(output / "index.html"),
        "cases": str(output / "cases.jsonl"),
    }
    write_json(output / "experiment.json", manifest)
    return manifest
