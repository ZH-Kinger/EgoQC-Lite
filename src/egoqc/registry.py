from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa

from .pipeline import run as run_scan
from .provenance import code_version, config_hash
from .validator import load_episode_index


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    return value.as_py() if isinstance(value, pa.Scalar) else value


def _safe_component(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_." else "-" for character in value)
    return cleaned.strip("-") or "dataset"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(os.fspath(path), timeout=30.0)
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS datasets (
          dataset_id TEXT PRIMARY KEY,
          source TEXT NOT NULL,
          logical_path TEXT NOT NULL,
          mounted_path TEXT NOT NULL,
          signature TEXT NOT NULL,
          file_count INTEGER NOT NULL,
          total_bytes INTEGER NOT NULL,
          missing_file_count INTEGER NOT NULL,
          registered_at TEXT NOT NULL,
          UNIQUE(source, logical_path)
        );

        CREATE TABLE IF NOT EXISTS files (
          dataset_id TEXT NOT NULL,
          relative_path TEXT NOT NULL,
          kind TEXT NOT NULL,
          size INTEGER NOT NULL,
          mtime_ns INTEGER NOT NULL,
          exists_flag INTEGER NOT NULL,
          PRIMARY KEY(dataset_id, relative_path),
          FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY,
          task_id TEXT NOT NULL,
          dataset_id TEXT NOT NULL,
          dataset_signature TEXT NOT NULL,
          standard_version TEXT NOT NULL,
          config_hash TEXT NOT NULL DEFAULT '',
          code_version TEXT NOT NULL DEFAULT '',
          output_path TEXT NOT NULL,
          status TEXT NOT NULL,
          claimed_by TEXT,
          lease_expires_at TEXT,
          heartbeat_at TEXT,
          progress_completed INTEGER NOT NULL DEFAULT 0,
          progress_total INTEGER NOT NULL DEFAULT 0,
          progress_fraction REAL NOT NULL DEFAULT 0,
          eta_seconds REAL,
          processed_bytes INTEGER NOT NULL DEFAULT 0,
          progress_path TEXT,
          started_at TEXT,
          finished_at TEXT,
          error TEXT,
          summary_json TEXT,
          FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
        );

        """
    )
    _ensure_column(db, "runs", "config_hash", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(db, "runs", "code_version", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(db, "runs", "claimed_by", "TEXT")
    _ensure_column(db, "runs", "lease_expires_at", "TEXT")
    _ensure_column(db, "runs", "heartbeat_at", "TEXT")
    _ensure_column(db, "runs", "progress_completed", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(db, "runs", "progress_total", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(db, "runs", "progress_fraction", "REAL NOT NULL DEFAULT 0")
    _ensure_column(db, "runs", "eta_seconds", "REAL")
    _ensure_column(db, "runs", "processed_bytes", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(db, "runs", "progress_path", "TEXT")
    db.executescript(
        """
        CREATE INDEX IF NOT EXISTS runs_lookup_v2
          ON runs(
            dataset_id, dataset_signature, standard_version,
            config_hash, code_version, status
          );
        """
    )
    return db


def _ensure_column(
    db: sqlite3.Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _relative(dataset: Path, path: Path) -> str:
    try:
        return path.relative_to(dataset).as_posix()
    except ValueError:
        return os.fspath(path)


def _collect_files(
    dataset: Path,
    video_key: str,
    finalized_marker: Optional[str] = None,
) -> List[Dict[str, Any]]:
    candidates: Dict[str, Tuple[Path, str]] = {}

    def add(path: Path, kind: str) -> None:
        relative = _relative(dataset, path)
        previous = candidates.get(relative)
        if previous is None or previous[1] == "other":
            candidates[relative] = (path, kind)

    add(dataset / "meta" / "info.json", "metadata")
    add(dataset / "meta" / "tasks.parquet", "metadata")
    if finalized_marker:
        marker_path = dataset / finalized_marker
        if not marker_path.is_file():
            raise FileNotFoundError(f"数据集未封板，缺少 {marker_path}")
        add(marker_path, "finalized_marker")
    episode_paths = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    for path in episode_paths:
        add(path, "episode_index")
    if not episode_paths:
        raise FileNotFoundError(f"{dataset}/meta/episodes 下没有 parquet")

    rows = load_episode_index(dataset).to_pylist()
    chunk_key = f"videos/{video_key}/chunk_index"
    file_key = f"videos/{video_key}/file_index"
    for row in rows:
        data_chunk = _value(row, "data/chunk_index")
        data_file = _value(row, "data/file_index")
        if data_chunk is not None and data_file is not None:
            add(
                dataset / "data" / f"chunk-{int(data_chunk):03d}" / f"file-{int(data_file):03d}.parquet",
                "data",
            )
        video_chunk = _value(row, chunk_key)
        video_file = _value(row, file_key)
        if video_chunk is not None and video_file is not None:
            add(
                dataset / "videos" / video_key / f"chunk-{int(video_chunk):03d}" / f"file-{int(video_file):03d}.mp4",
                "video",
            )

    records: List[Dict[str, Any]] = []
    for relative, (path, kind) in sorted(candidates.items()):
        try:
            stat = path.stat()
            records.append(
                {
                    "relative_path": relative,
                    "kind": kind,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "exists": True,
                }
            )
        except FileNotFoundError:
            records.append(
                {
                    "relative_path": relative,
                    "kind": kind,
                    "size": -1,
                    "mtime_ns": -1,
                    "exists": False,
                }
            )
    return records


def _signature(records: Sequence[Dict[str, Any]]) -> str:
    payload = [
        [
            record["relative_path"],
            record["kind"],
            record["size"],
            record["mtime_ns"],
            record["exists"],
        ]
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _collect_registered_files(
    db: sqlite3.Connection,
    dataset_id: str,
    mounted_path: Path,
) -> List[Dict[str, Any]]:
    rows = db.execute(
        """
        SELECT relative_path, kind
        FROM files
        WHERE dataset_id = ?
        ORDER BY relative_path
        """,
        (dataset_id,),
    ).fetchall()
    records: List[Dict[str, Any]] = []
    for row in rows:
        path = mounted_path / row["relative_path"]
        try:
            stat = path.stat()
            records.append(
                {
                    "relative_path": row["relative_path"],
                    "kind": row["kind"],
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "exists": True,
                }
            )
        except FileNotFoundError:
            records.append(
                {
                    "relative_path": row["relative_path"],
                    "kind": row["kind"],
                    "size": -1,
                    "mtime_ns": -1,
                    "exists": False,
                }
            )
    return records


def _logical_path(dataset: Path, source_root: Optional[Path]) -> str:
    if source_root is None:
        return os.fspath(dataset)
    try:
        return dataset.relative_to(source_root).as_posix()
    except ValueError as error:
        raise ValueError(f"数据集 {dataset} 不在 source-root {source_root} 下") from error


def register_datasets(
    registry_path: Path,
    datasets: Sequence[Path],
    source: str,
    video_key: str,
    source_root: Optional[Path] = None,
    finalized_marker: Optional[str] = None,
) -> Dict[str, Any]:
    source_root = source_root.resolve() if source_root else None
    db = _connect(registry_path)
    registered: List[Dict[str, Any]] = []
    try:
        for raw_path in datasets:
            dataset = raw_path.expanduser().resolve()
            logical_path = _logical_path(dataset, source_root)
            identity = f"{source}\0{logical_path}"
            suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            dataset_id = f"{_safe_component(dataset.name)}-{suffix}"
            records = _collect_files(dataset, video_key, finalized_marker)
            signature = _signature(records)
            total_bytes = sum(record["size"] for record in records if record["exists"])
            missing = sum(not record["exists"] for record in records)
            registered_at = _utc_now()
            with db:
                db.execute(
                    """
                    INSERT INTO datasets(
                      dataset_id, source, logical_path, mounted_path, signature,
                      file_count, total_bytes, missing_file_count, registered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(dataset_id) DO UPDATE SET
                      source=excluded.source,
                      logical_path=excluded.logical_path,
                      mounted_path=excluded.mounted_path,
                      signature=excluded.signature,
                      file_count=excluded.file_count,
                      total_bytes=excluded.total_bytes,
                      missing_file_count=excluded.missing_file_count,
                      registered_at=excluded.registered_at
                    """,
                    (
                        dataset_id,
                        source,
                        logical_path,
                        os.fspath(dataset),
                        signature,
                        len(records),
                        total_bytes,
                        missing,
                        registered_at,
                    ),
                )
                db.execute("DELETE FROM files WHERE dataset_id = ?", (dataset_id,))
                db.executemany(
                    """
                    INSERT INTO files(
                      dataset_id, relative_path, kind, size, mtime_ns, exists_flag
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            dataset_id,
                            record["relative_path"],
                            record["kind"],
                            record["size"],
                            record["mtime_ns"],
                            int(record["exists"]),
                        )
                        for record in records
                    ],
                )
            registered.append(
                {
                    "dataset_id": dataset_id,
                    "source": source,
                    "logical_path": logical_path,
                    "mounted_path": os.fspath(dataset),
                    "signature": signature,
                    "file_count": len(records),
                    "total_bytes": total_bytes,
                    "missing_file_count": missing,
                }
            )
    finally:
        db.close()
    return {"registry": os.fspath(registry_path.resolve()), "registered": registered}


def create_manifest(
    registry_path: Path,
    manifest_path: Path,
    output_root: Path,
    config: Dict[str, Any],
    dataset_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    standard_version = config["standard_version"]
    configuration_hash = config_hash(config)
    package_version = code_version()
    selected = set(dataset_ids or [])
    db = _connect(registry_path)
    tasks: List[Dict[str, Any]] = []
    skipped_current = 0
    try:
        rows = db.execute("SELECT * FROM datasets ORDER BY source, logical_path").fetchall()
        for row in rows:
            if selected and row["dataset_id"] not in selected:
                continue
            current = db.execute(
                """
                SELECT 1 FROM runs
                WHERE dataset_id = ? AND dataset_signature = ?
                  AND standard_version = ? AND config_hash = ?
                  AND code_version = ? AND status = 'succeeded'
                LIMIT 1
                """,
                (
                    row["dataset_id"],
                    row["signature"],
                    standard_version,
                    configuration_hash,
                    package_version,
                ),
            ).fetchone()
            if current:
                skipped_current += 1
                continue
            task_payload = (
                f"{row['dataset_id']}\0{row['signature']}\0{standard_version}"
                f"\0{configuration_hash}\0{package_version}"
            )
            task_id = hashlib.sha256(task_payload.encode("utf-8")).hexdigest()[:20]
            output_path = (
                output_root.expanduser().resolve()
                / row["dataset_id"]
                / _safe_component(standard_version)
                / configuration_hash[:12]
                / row["signature"][:12]
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "dataset_id": row["dataset_id"],
                    "dataset_path": row["mounted_path"],
                    "dataset_signature": row["signature"],
                    "standard_version": standard_version,
                    "config_hash": configuration_hash,
                    "code_version": package_version,
                    "output_path": os.fspath(output_path),
                    "file_count": row["file_count"],
                    "total_bytes": row["total_bytes"],
                    "missing_file_count": row["missing_file_count"],
                }
            )
    finally:
        db.close()

    _write_jsonl_atomic(manifest_path, tasks)
    return {
        "manifest": os.fspath(manifest_path.resolve()),
        "task_count": len(tasks),
        "skipped_current": skipped_current,
        "total_bytes": sum(task["total_bytes"] for task in tasks),
    }


def registry_status(registry_path: Path) -> Dict[str, Any]:
    db = _connect(registry_path)
    try:
        dataset = db.execute(
            """
            SELECT
              COUNT(*) AS dataset_count,
              COALESCE(SUM(file_count), 0) AS file_count,
              COALESCE(SUM(total_bytes), 0) AS total_bytes,
              COALESCE(SUM(missing_file_count), 0) AS missing_file_count
            FROM datasets
            """
        ).fetchone()
        run_counts = {
            row["status"]: row["count"]
            for row in db.execute(
                "SELECT status, COUNT(*) AS count FROM runs GROUP BY status"
            ).fetchall()
        }
        now = _utc_now()
        expired_leases = db.execute(
            """
            SELECT COUNT(*) FROM runs
            WHERE status = 'running'
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at < ?
            """,
            (now,),
        ).fetchone()[0]
        latest = [
            dict(row)
            for row in db.execute(
                """
                SELECT
                  dataset_id, dataset_signature, standard_version,
                  config_hash, code_version, status, output_path,
                  started_at, finished_at, error, progress_completed,
                  progress_total, progress_fraction, eta_seconds,
                  processed_bytes, progress_path, heartbeat_at
                FROM runs
                ORDER BY COALESCE(finished_at, started_at) DESC
                LIMIT 20
                """
            ).fetchall()
        ]
        return {
            "registry": str(registry_path.expanduser().resolve()),
            "dataset_count": dataset["dataset_count"],
            "file_count": dataset["file_count"],
            "total_bytes": dataset["total_bytes"],
            "missing_file_count": dataset["missing_file_count"],
            "run_counts": run_counts,
            "expired_leases": expired_leases,
            "latest_runs": latest,
        }
    finally:
        db.close()


def _write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _record_run(
    db: sqlite3.Connection,
    task: Dict[str, Any],
    status: str,
    started_at: Optional[str],
    finished_at: Optional[str],
    error: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    claimed_by: Optional[str] = None,
    lease_expires_at: Optional[str] = None,
    heartbeat_at: Optional[str] = None,
) -> None:
    run_id = task["task_id"]
    with db:
        db.execute(
            """
            INSERT INTO runs(
              run_id, task_id, dataset_id, dataset_signature, standard_version,
              config_hash, code_version, output_path, status, claimed_by,
              lease_expires_at, heartbeat_at, started_at, finished_at, error,
              summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
              status=excluded.status,
              claimed_by=excluded.claimed_by,
              lease_expires_at=excluded.lease_expires_at,
              heartbeat_at=excluded.heartbeat_at,
              started_at=excluded.started_at,
              finished_at=excluded.finished_at,
              error=excluded.error,
              summary_json=excluded.summary_json
            """,
            (
                run_id,
                task["task_id"],
                task["dataset_id"],
                task["dataset_signature"],
                task["standard_version"],
                task["config_hash"],
                task["code_version"],
                task["output_path"],
                status,
                claimed_by,
                lease_expires_at,
                heartbeat_at,
                started_at,
                finished_at,
                error,
                json.dumps(summary, ensure_ascii=False, allow_nan=True) if summary else None,
            ),
        )


def _claim_task(
    db: sqlite3.Connection,
    task: Dict[str, Any],
    worker_id: str,
    lease_seconds: int,
) -> str:
    now = _utc_now()
    lease = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
    db.execute("BEGIN IMMEDIATE")
    try:
        existing = db.execute(
            "SELECT status, claimed_by, lease_expires_at FROM runs WHERE run_id = ?",
            (task["task_id"],),
        ).fetchone()
        if existing and existing["status"] == "succeeded":
            db.commit()
            return "succeeded"
        if (
            existing
            and existing["status"] == "running"
            and existing["claimed_by"] != worker_id
            and existing["lease_expires_at"]
            and existing["lease_expires_at"] > now
        ):
            db.commit()
            return "busy"
        db.execute(
            """
            INSERT INTO runs(
              run_id, task_id, dataset_id, dataset_signature, standard_version,
              config_hash, code_version, output_path, status, claimed_by,
              lease_expires_at, heartbeat_at, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
              status='running',
              claimed_by=excluded.claimed_by,
              lease_expires_at=excluded.lease_expires_at,
              heartbeat_at=excluded.heartbeat_at,
              started_at=excluded.started_at,
              progress_completed=0,
              progress_total=0,
              progress_fraction=0,
              eta_seconds=NULL,
              processed_bytes=0,
              progress_path=NULL,
              finished_at=NULL,
              error=NULL
            """,
            (
                task["task_id"],
                task["task_id"],
                task["dataset_id"],
                task["dataset_signature"],
                task["standard_version"],
                task["config_hash"],
                task["code_version"],
                task["output_path"],
                worker_id,
                lease,
                now,
                now,
            ),
        )
        db.commit()
        return "claimed"
    except Exception:
        db.rollback()
        raise


def _heartbeat(
    db: sqlite3.Connection,
    task_id: str,
    worker_id: str,
    lease_seconds: int,
    progress: Optional[Dict[str, Any]] = None,
) -> None:
    now = _utc_now()
    lease = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
    with db:
        if progress is None:
            db.execute(
                """
                UPDATE runs SET heartbeat_at = ?, lease_expires_at = ?
                WHERE run_id = ? AND status = 'running' AND claimed_by = ?
                """,
                (now, lease, task_id, worker_id),
            )
        else:
            db.execute(
                """
                UPDATE runs SET heartbeat_at = ?, lease_expires_at = ?,
                  progress_completed = ?, progress_total = ?, progress_fraction = ?,
                  eta_seconds = ?, processed_bytes = ?, progress_path = ?
                WHERE run_id = ? AND status = 'running' AND claimed_by = ?
                """,
                (
                    now,
                    lease,
                    int(progress.get("completed", 0)),
                    int(progress.get("total", 0)),
                    float(progress.get("fraction", 0.0)),
                    progress.get("eta_s"),
                    int(progress.get("logical_input_bytes", 0)),
                    progress.get("path"),
                    task_id,
                    worker_id,
                ),
            )


def _execute_task(
    registry_path: Path,
    task: Dict[str, Any],
    config: Dict[str, Any],
    hash_mode: str,
    cache_root: Path,
    worker_id: str,
    lease_seconds: int,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    db = _connect(registry_path)
    try:
        claim = _claim_task(db, task, worker_id, lease_seconds)
        if claim == "succeeded":
            return {**task, "status": "skipped", "reason": "already_succeeded"}
        if claim == "busy":
            return {
                **task,
                "status": "busy",
                "reason": "claimed_by_another_worker",
            }

        started_at = _utc_now()
        try:
            dataset = Path(task["dataset_path"])
            current_records = _collect_registered_files(
                db,
                task["dataset_id"],
                dataset,
            )
            current_signature = _signature(current_records)
            if current_signature != task["dataset_signature"]:
                error = "数据集在生成 manifest 后发生变化，请重新 register 和 plan"
                _record_run(
                    db,
                    task,
                    "stale",
                    started_at,
                    _utc_now(),
                    error=error,
                    claimed_by=worker_id,
                )
                return {**task, "status": "stale", "error": error}

            def progress(event: Dict[str, Any]) -> None:
                _heartbeat(db, task["task_id"], worker_id, lease_seconds, event)
                if progress_callback:
                    progress_callback({"task_id": task["task_id"], "dataset_id": task["dataset_id"], **event})

            summary = run_scan(
                dataset,
                Path(task["output_path"]),
                config,
                hash_mode,
                cache_root=cache_root,
                progress_callback=progress,
            )
            finished_at = _utc_now()
            _record_run(
                db,
                task,
                "succeeded",
                started_at,
                finished_at,
                summary=summary,
                claimed_by=worker_id,
                heartbeat_at=finished_at,
            )
            return {
                **task,
                "status": "succeeded",
                "started_at": started_at,
                "finished_at": finished_at,
                "summary": summary,
            }
        except Exception as error:
            finished_at = _utc_now()
            message = f"{type(error).__name__}: {error}"
            _record_run(
                db,
                task,
                "failed",
                started_at,
                finished_at,
                error=message,
                claimed_by=worker_id,
                heartbeat_at=finished_at,
            )
            return {
                **task,
                "status": "failed",
                "started_at": started_at,
                "finished_at": finished_at,
                "error": message,
            }
    finally:
        db.close()


def run_manifest(
    registry_path: Path,
    manifest_path: Path,
    config: Dict[str, Any],
    hash_mode: str = "none",
    results_path: Optional[Path] = None,
    continue_on_error: bool = False,
    cache_root: Optional[Path] = None,
    worker_id: Optional[str] = None,
    lease_seconds: int = 3600,
    workers: int = 1,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    standard_version = config["standard_version"]
    configuration_hash = config_hash(config)
    package_version = code_version()
    tasks = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results_path = results_path or manifest_path.with_name(f"{manifest_path.stem}.results.jsonl")
    if workers < 1:
        raise ValueError("workers 必须大于等于 1")
    results: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    cache_root = cache_root or registry_path.parent / "artifact-cache"
    worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
    migration_db = _connect(registry_path)
    migration_db.close()
    counts = {
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "busy": 0,
        "stale": 0,
    }
    for task in tasks:
        if task["standard_version"] != standard_version:
            raise ValueError(
                f"任务 {task['task_id']} 标准版本为 {task['standard_version']}，"
                f"当前配置为 {standard_version}"
            )
        if task.get("config_hash") != configuration_hash:
            raise ValueError(f"任务 {task['task_id']} 的 config hash 与当前配置不一致")
        if task.get("code_version") != package_version:
            raise ValueError(f"任务 {task['task_id']} 的代码版本与当前包不一致")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _execute_task,
                registry_path,
                task,
                config,
                hash_mode,
                cache_root,
                f"{worker_id}-{index}",
                lease_seconds,
                progress_callback,
            ): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except Exception as error:
                task = tasks[index]
                result = {
                    **task,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            results[index] = result
            counts[result["status"]] += 1
            _write_jsonl_atomic(
                results_path,
                [value for value in results if value is not None],
            )

    completed_results = [value for value in results if value is not None]
    _write_jsonl_atomic(results_path, completed_results)
    if counts["failed"] and not continue_on_error:
        raise RuntimeError(
            f"{counts['failed']} 个任务失败，详情见 {results_path.resolve()}"
        )

    return {
        "manifest": os.fspath(manifest_path.resolve()),
        "results": os.fspath(results_path.resolve()),
        "task_count": len(tasks),
        "workers": workers,
        **counts,
    }
