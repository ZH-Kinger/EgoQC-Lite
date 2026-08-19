from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .review_template import WORKBENCH_TEMPLATE


def _jsonl_by_episode(path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("episode_index") is not None:
            rows[int(row["episode_index"])] = row
    return rows


def _annotated_uri(root: Optional[Path], episode: int) -> Optional[str]:
    if root is None:
        return None
    candidates = [
        root / f"episode-{episode:06d}-annotated.mp4",
        root / f"episode-{episode:06d}" / "annotated.mp4",
        root / f"episode-{episode:06d}-repaired-annotated.mp4",
        root / f"episode-{episode:06d}" / "repaired-annotated.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve().as_uri()
    return None


def _evidence_path(value: Any, output: Path) -> Optional[str]:
    if not value:
        return None
    path = Path(str(value))
    try:
        return path.resolve().relative_to(output.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def write_review_workbench(
    dataset: Path,
    output: Path,
    contact_frames: Dict[int, List[Tuple[int, Path]]],
    manifest: List[Dict[str, Any]],
    routes: Dict[int, Dict[str, Any]],
    quality_root: Optional[Path] = None,
    annotated_root: Optional[Path] = None,
) -> Path:
    """Write a self-contained, offline-first episode review application."""

    quality_root = quality_root or Path()
    decisions = _jsonl_by_episode(quality_root / "decisions" / "episode_decisions.jsonl")
    results = _jsonl_by_episode(quality_root / "episodes.jsonl")
    by_episode: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_episode[int(row["episode_index"])].append(row)

    records: List[Dict[str, Any]] = []
    for episode_index in sorted(contact_frames):
        route = routes[episode_index]
        result = results.get(episode_index, {})
        system = decisions.get(episode_index, {})
        issues = result.get("issues", [])
        records.append(
            {
                "episode_index": episode_index,
                "contact_sheet": f"episode-{episode_index:06d}-contact-sheet.jpg",
                "sampled_frames": [frame for frame, _ in contact_frames[episode_index]],
                "evidence_frames": [
                    {
                        "frame_index": int(row["frame_index"]),
                        "image": _evidence_path(row.get("image"), output),
                        "overlay_image": _evidence_path(row.get("overlay_image"), output),
                        "mano_status": row.get("mano_status", "disabled"),
                        "mano_error": row.get("mano_error"),
                    }
                    for row in sorted(
                        by_episode[episode_index], key=lambda item: int(item["frame_index"])
                    )
                ],
                "source_video": route["source_video"],
                "video_uri": route["video_path"].resolve().as_uri(),
                "annotated_video_uri": _annotated_uri(annotated_root, episode_index),
                "start_time": route["start_time"],
                "stop_time": route["stop_time"],
                "system_decision": system.get("decision", "unscored"),
                "motion_pass": system.get("motion_pass"),
                "tier": result.get("tier", "unknown"),
                "issues": issues,
                "metrics": result.get("metrics", {}),
                "mano_rendered": sum(
                    row.get("mano_status") == "rendered" for row in by_episode[episode_index]
                ),
                "mano_failed": sum(
                    row.get("mano_status") == "failed" for row in by_episode[episode_index]
                ),
            }
        )

    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    dataset_json = json.dumps(str(dataset), ensure_ascii=False).replace("</", "<\\/")
    dataset_label = html.escape(str(dataset))
    template = WORKBENCH_TEMPLATE
    document = (
        template.replace("__PAYLOAD__", payload)
        .replace("__DATASET__", dataset_json)
        .replace("__DATASET_LABEL__", dataset_label)
    )
    path = output / "review.html"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    media_routes = {
        str(episode): {
            "source": str(routes[episode]["video_path"].resolve()),
            "annotated": (
                str(Path(uri.removeprefix("file://")))
                if (uri := _annotated_uri(annotated_root, episode))
                else None
            ),
        }
        for episode in sorted(contact_frames)
    }
    media_path = output / "media-routes.json"
    media_temporary = media_path.with_name(f".{media_path.name}.{os.getpid()}.tmp")
    media_temporary.write_text(
        json.dumps(media_routes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    media_temporary.replace(media_path)
    return path
