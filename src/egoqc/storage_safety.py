from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_PROTECTED_RAW_ROOTS = (Path("/mnt/data"),)


def protected_raw_roots() -> tuple[Path, ...]:
    configured = os.environ.get("EGOQC_PROTECTED_RAW_ROOTS")
    values: Iterable[Path]
    if configured:
        values = (Path(value) for value in configured.split(os.pathsep) if value)
    else:
        values = DEFAULT_PROTECTED_RAW_ROOTS
    return tuple(path.expanduser().resolve(strict=False) for path in values)


def assert_derived_output(
    output: Path, *, protected_roots: Optional[Iterable[Path]] = None
) -> Path:
    resolved = output.expanduser().resolve(strict=False)
    roots = (
        tuple(path.expanduser().resolve(strict=False) for path in protected_roots)
        if protected_roots is not None
        else protected_raw_roots()
    )
    for root in roots:
        if resolved == root or root in resolved.parents:
            raise ValueError(
                f"refusing to write derived output under protected raw root: {resolved} "
                f"(protected: {root})"
            )
    return resolved


@dataclass(frozen=True)
class RawFileStamp:
    size: int
    mtime_ns: int
    inode: int
    device: int


def raw_file_stamp(path: Path) -> RawFileStamp:
    stat = path.stat()
    return RawFileStamp(
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        inode=int(stat.st_ino),
        device=int(stat.st_dev),
    )


def assert_raw_file_unchanged(path: Path, before: RawFileStamp) -> None:
    after = raw_file_stamp(path)
    if after != before:
        raise RuntimeError(
            f"protected raw source changed while being read: {path}; "
            f"before={before}, after={after}"
        )
