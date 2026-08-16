from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional


class ArtifactCache:
    def __init__(self, root: Path):
        self.root = root

    def path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / key[:2] / f"{key}.json"

    def read(self, namespace: str, key: str) -> Optional[Any]:
        path = self.path(namespace, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def write(self, namespace: str, key: str, value: Any) -> Path:
        path = self.path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=True),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    @staticmethod
    def materialize(source: Path, destination: Path) -> None:
        if destination.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
