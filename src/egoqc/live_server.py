from __future__ import annotations

import json
import mimetypes
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .registry import registry_status
from .review_workbench import write_review_workbench


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        return []
    return rows


def live_payload(quality_root: Path, registry: Optional[Path] = None) -> Dict[str, Any]:
    quality_root = quality_root.expanduser().resolve()
    current = _read_json(quality_root / "live" / "current.json")
    run_id = current.get("run_id")
    episode_rows: Dict[int, Dict[str, Any]] = {}
    if run_id:
        run_root = quality_root / "live" / "runs" / str(run_id)
        final_path = run_root / "episodes-final.jsonl"
        sources = [final_path] if final_path.exists() else sorted((run_root / "shards").glob("*.jsonl"))
        for source in sources:
            for row in _read_jsonl(source):
                episode_rows[int(row["episode_index"])] = row
    elif (quality_root / "episodes.jsonl").exists():
        for row in _read_jsonl(quality_root / "episodes.jsonl"):
            episode_rows[int(row["episode_index"])] = row

    registry_run = None
    if registry is not None and registry.exists():
        status = registry_status(registry)
        for run in status.get("latest_runs", []):
            if Path(run["output_path"]).expanduser().resolve() == quality_root:
                registry_run = run
                break
    return {
        "quality_root": str(quality_root),
        "status": current,
        "registry_run": registry_run,
        "episodes": [episode_rows[key] for key in sorted(episode_rows)],
        "episode_count": len(episode_rows),
    }


class _ReviewHandler(SimpleHTTPRequestHandler):
    quality_root: Path
    registry: Optional[Path]
    evidence_root: Path

    def _media_file(self, kind: str, episode: str) -> Optional[Path]:
        if kind not in {"source", "annotated"}:
            return None
        try:
            key = str(int(episode))
        except ValueError:
            return None
        routes = _read_json(self.evidence_root / "media-routes.json")
        value = routes.get(key, {}).get(kind)
        if not value:
            return None
        path = Path(value).expanduser().resolve()
        return path if path.is_file() else None

    def _serve_media(self, kind: str, episode: str, head_only: bool = False) -> None:
        path = self._media_file(kind, episode)
        if path is None:
            self.send_error(404, "media not found")
            return
        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        range_header = self.headers.get("Range")
        partial = False
        if range_header and range_header.startswith("bytes="):
            try:
                start_text, end_text = range_header[6:].split("-", 1)
                if start_text:
                    start = int(start_text)
                if end_text:
                    end = min(end, int(end_text))
                if start < 0 or start > end or start >= size:
                    raise ValueError
                partial = True
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler API
        request_path = urlparse(self.path).path
        if request_path == "/api/live":
            payload = live_payload(self.quality_root, self.registry)
            body = json.dumps(payload, ensure_ascii=False, allow_nan=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        parts = request_path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "media"]:
            self._serve_media(parts[2], parts[3])
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib HTTP handler API
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "media"]:
            self._serve_media(parts[2], parts[3], head_only=True)
            return
        super().do_HEAD()


def serve_review(
    evidence_root: Path,
    quality_root: Path,
    registry: Optional[Path] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    evidence_root = evidence_root.expanduser().resolve()
    quality_root = quality_root.expanduser().resolve()
    if not (evidence_root / "review.html").exists():
        evidence_root.mkdir(parents=True, exist_ok=True)
        current = _read_json(quality_root / "live" / "current.json")
        dataset = Path(current.get("dataset") or quality_root)
        write_review_workbench(
            dataset,
            evidence_root,
            {},
            [],
            {},
            quality_root=quality_root,
        )
    class BoundHandler(_ReviewHandler):
        pass

    BoundHandler.quality_root = quality_root
    BoundHandler.registry = registry.expanduser().resolve() if registry else None
    BoundHandler.evidence_root = evidence_root
    server = ThreadingHTTPServer((host, port), partial(BoundHandler, directory=str(evidence_root)))
    print(f"EgoQC review: http://{host}:{port}/review.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
