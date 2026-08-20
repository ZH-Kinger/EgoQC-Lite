from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


# /mnt/data and /mnt/workspace can be aliases of the same CPFS mount.  Protect
# actual raw namespaces, not one spelling of the mount root.
DEFAULT_PROTECTED_RAW_ROOTS = (
    Path("/mnt/data/oss"),
    Path("/mnt/data/shutu"),
    Path("/mnt/data/egodex"),
)
DEFAULT_MOUNT_ALIAS_GROUPS = ((Path("/mnt/data"), Path("/mnt/workspace")),)


def protected_raw_roots() -> tuple[Path, ...]:
    configured = os.environ.get("EGOQC_PROTECTED_RAW_ROOTS")
    values: Iterable[Path]
    if configured:
        values = (Path(value) for value in configured.split(os.pathsep) if value)
    else:
        values = DEFAULT_PROTECTED_RAW_ROOTS
    return tuple(path.expanduser().resolve(strict=False) for path in values)


def assert_derived_output(
    output: Path,
    *,
    protected_roots: Optional[Iterable[Path]] = None,
    mount_alias_groups: Optional[Sequence[Sequence[Path]]] = None,
) -> Path:
    resolved = output.expanduser().resolve(strict=False)
    roots = (
        tuple(path.expanduser().resolve(strict=False) for path in protected_roots)
        if protected_roots is not None
        else protected_raw_roots()
    )
    alias_groups = mount_alias_groups or DEFAULT_MOUNT_ALIAS_GROUPS
    candidates = _lexical_mount_aliases(resolved, alias_groups)
    for root in roots:
        if any(
            candidate == root
            or root in candidate.parents
            or _aliases_protected_subtree(candidate, root)
            for candidate in candidates
        ):
            raise ValueError(
                f"refusing to write derived output under protected raw root: {resolved} "
                f"(protected: {root})"
            )
    return resolved


def _lexical_mount_aliases(
    candidate: Path, groups: Sequence[Sequence[Path]]
) -> set[Path]:
    """Expand equivalent mount spellings, including hidden child mounts.

    A child mount such as /mnt/data/oss can have a different device/inode from
    the underlying /mnt/workspace/oss path.  Root identity alone cannot detect
    that shadowing, so preserve the relative suffix across declared aliases.
    """

    expanded = {candidate}
    for raw_group in groups:
        group = [path.expanduser().resolve(strict=False) for path in raw_group]
        for source in group:
            if candidate == source or source in candidate.parents:
                suffix = candidate.relative_to(source)
                expanded.update(target / suffix for target in group)
    return expanded


def _aliases_protected_subtree(candidate: Path, protected: Path) -> bool:
    """Detect a path below a protected directory through a bind/mount alias.

    Example: when /mnt/data and /mnt/workspace name the same mount,
    /mnt/workspace/oss/new.json must be treated as below /mnt/data/oss even
    though lexical Path.parents cannot see that relationship.
    """

    try:
        protected_stat = protected.stat()
    except OSError:
        return False
    for ancestor in (candidate, *candidate.parents):
        try:
            stat = ancestor.stat()
        except OSError:
            continue
        if stat.st_dev == protected_stat.st_dev and stat.st_ino == protected_stat.st_ino:
            return True
    return False


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
