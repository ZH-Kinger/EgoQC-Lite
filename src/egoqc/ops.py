from __future__ import annotations

import importlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .provenance import code_version, config_hash


def doctor(
    config: Dict[str, Any],
    registry_path: Optional[Path] = None,
    source_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    record(
        "python",
        sys.version_info >= (3, 9),
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    for module_name in ("numpy", "pyarrow", "av", "PIL"):
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "available")
            record(f"dependency:{module_name}", True, str(version))
        except Exception as error:
            record(f"dependency:{module_name}", False, str(error))

    required_config = {"standard_version", "video_key", "thresholds", "sampling"}
    missing_config = sorted(required_config - set(config))
    record(
        "config",
        not missing_config,
        (
            f"hash={config_hash(config)[:12]}"
            if not missing_config
            else f"missing={','.join(missing_config)}"
        ),
    )

    for name, path, writable in (
        ("source_root", source_root, False),
        ("output_root", output_root, True),
    ):
        if path is None:
            continue
        resolved = path.expanduser().resolve()
        if writable:
            try:
                resolved.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=resolved, prefix=".egoqc-doctor-"):
                    pass
                usage = shutil.disk_usage(resolved)
                record(
                    name,
                    True,
                    f"{resolved}, free={usage.free / (1024**3):.2f} GiB",
                )
            except Exception as error:
                record(name, False, f"{resolved}: {error}")
        else:
            record(
                name,
                resolved.is_dir() and os.access(resolved, os.R_OK),
                str(resolved),
            )

    if registry_path is not None:
        try:
            registry_path = registry_path.expanduser().resolve()
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(registry_path), timeout=30.0)
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TEMP TABLE _egoqc_doctor_lock_test (id INTEGER PRIMARY KEY)"
            )
            db.rollback()
            db.close()
            record("registry", integrity == "ok", f"{registry_path}: {integrity}")
        except Exception as error:
            record("registry", False, str(error))

    return {
        "ok": all(check["ok"] for check in checks),
        "egoqc_version": code_version(),
        "checks": checks,
    }


def self_test(workdir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    from .extract import extract_samples
    from .registry import create_manifest, register_datasets, run_manifest
    from .synthetic import create_synthetic_dataset

    workdir = workdir.expanduser().resolve()
    dataset = create_synthetic_dataset(workdir / "raw" / "batch-001")
    registry = workdir / "control" / "registry.sqlite"
    registration = register_datasets(
        registry,
        [dataset],
        source="synthetic",
        video_key=config["video_key"],
        source_root=workdir / "raw",
    )
    manifest = workdir / "control" / "self-test.jsonl"
    plan = create_manifest(
        registry,
        manifest,
        workdir / "results",
        config,
    )
    execution = run_manifest(
        registry,
        manifest,
        config,
        hash_mode="headtail",
        continue_on_error=False,
        workers=1,
    )
    second_manifest = workdir / "control" / "self-test-second.jsonl"
    incremental_plan = create_manifest(
        registry,
        second_manifest,
        workdir / "results",
        config,
    )
    task = json.loads(manifest.read_text(encoding="utf-8").strip())
    result_root = Path(task["output_path"])
    evidence = extract_samples(
        dataset,
        result_root / "sample_plan.jsonl",
        result_root / "evidence",
        config["video_key"],
    )
    required_outputs = [
        "summary.json",
        "episodes.parquet",
        "issues.parquet",
        "videos.parquet",
        "shards.parquet",
        "report.html",
    ]
    missing = [
        name for name in required_outputs if not (result_root / name).exists()
    ]
    return {
        "ok": (
            not missing
            and plan["task_count"] == 1
            and execution["succeeded"] == 1
            and evidence["frames_extracted"] > 0
            and incremental_plan["task_count"] == 0
        ),
        "workdir": str(workdir),
        "dataset_id": registration["registered"][0]["dataset_id"],
        "result_root": str(result_root),
        "missing_outputs": missing,
        "execution": execution,
        "incremental_plan": incremental_plan,
        "evidence": evidence,
    }
