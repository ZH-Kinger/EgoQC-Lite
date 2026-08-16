from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from . import __version__


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def code_version() -> str:
    """Return a cache identity that changes when shipped Python sources change."""
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py"), key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    identity = f"{__version__}+src.{digest.hexdigest()[:16]}"
    build_id = os.environ.get("EGOQC_BUILD_ID", "").strip()
    if build_id:
        identity += f"+build.{hashlib.sha256(build_id.encode()).hexdigest()[:12]}"
    return identity
