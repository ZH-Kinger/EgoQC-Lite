from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Optional


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
              path TEXT PRIMARY KEY,
              size INTEGER NOT NULL,
              mtime_ns INTEGER NOT NULL,
              fingerprint TEXT NOT NULL,
              pipeline_version TEXT NOT NULL
            )
            """
        )
        self.db.commit()

    @staticmethod
    def fingerprint(path: Path, mode: str = "headtail") -> str:
        stat = path.stat()
        if mode == "none":
            return f"{stat.st_size}:{stat.st_mtime_ns}"
        digest = hashlib.sha256()
        block = 1024 * 1024
        with path.open("rb") as handle:
            digest.update(handle.read(block))
            if stat.st_size > block:
                handle.seek(max(0, stat.st_size - block))
                digest.update(handle.read(block))
        digest.update(str(stat.st_size).encode())
        return digest.hexdigest()

    def is_current(self, path: Path, pipeline_version: str, fingerprint: str) -> bool:
        stat = path.stat()
        row: Optional[tuple] = self.db.execute(
            "SELECT size, mtime_ns, fingerprint, pipeline_version FROM files WHERE path = ?",
            (os.fspath(path),),
        ).fetchone()
        return bool(
            row
            and row[0] == stat.st_size
            and row[1] == stat.st_mtime_ns
            and row[2] == fingerprint
            and row[3] == pipeline_version
        )

    def record(self, path: Path, pipeline_version: str, fingerprint: str) -> None:
        stat = path.stat()
        self.db.execute(
            """
            INSERT INTO files(path, size, mtime_ns, fingerprint, pipeline_version)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
              size=excluded.size,
              mtime_ns=excluded.mtime_ns,
              fingerprint=excluded.fingerprint,
              pipeline_version=excluded.pipeline_version
            """,
            (os.fspath(path), stat.st_size, stat.st_mtime_ns, fingerprint, pipeline_version),
        )

    def close(self) -> None:
        self.db.commit()
        self.db.close()
