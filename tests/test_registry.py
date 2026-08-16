import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from egoqc.registry import (
    _claim_task,
    _connect,
    create_manifest,
    register_datasets,
    run_manifest,
)

from create_fixture import create_fixture


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (Path(__file__).parents[1] / "config" / "default.json").read_text()
        )

    def test_incremental_mounted_directory_workflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mount = root / "cpfs"
            dataset = create_fixture(mount / "source-a" / "batch-001")
            registry = root / "quality" / "registry.sqlite"

            registration = register_datasets(
                registry,
                [dataset],
                source="oss-prod",
                video_key=self.config["video_key"],
                source_root=mount,
            )
            self.assertEqual(len(registration["registered"]), 1)
            self.assertEqual(registration["registered"][0]["missing_file_count"], 0)

            manifest = root / "quality" / "manifests" / "run-001.jsonl"
            planned = create_manifest(
                registry,
                manifest,
                root / "quality" / "runs",
                self.config,
            )
            self.assertEqual(planned["task_count"], 1)

            executed = run_manifest(registry, manifest, self.config)
            self.assertEqual(executed["succeeded"], 1)
            self.assertEqual(executed["failed"], 0)

            second_manifest = root / "quality" / "manifests" / "run-002.jsonl"
            second_plan = create_manifest(
                registry,
                second_manifest,
                root / "quality" / "runs",
                self.config,
            )
            self.assertEqual(second_plan["task_count"], 0)
            self.assertEqual(second_plan["skipped_current"], 1)

            data_path = dataset / "data" / "chunk-000" / "file-000.parquet"
            stat = data_path.stat()
            os.utime(data_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            changed = register_datasets(
                registry,
                [dataset],
                source="oss-prod",
                video_key=self.config["video_key"],
                source_root=mount,
            )
            self.assertNotEqual(
                changed["registered"][0]["signature"],
                registration["registered"][0]["signature"],
            )
            third_plan = create_manifest(
                registry,
                root / "quality" / "manifests" / "run-003.jsonl",
                root / "quality" / "runs",
                self.config,
            )
            self.assertEqual(third_plan["task_count"], 1)

    def test_manifest_detects_dataset_change_after_planning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "cpfs" / "batch-001")
            registry = root / "registry.sqlite"
            register_datasets(
                registry,
                [dataset],
                source="cpfs-dev",
                video_key=self.config["video_key"],
                source_root=root / "cpfs",
            )
            manifest = root / "manifest.jsonl"
            create_manifest(
                registry,
                manifest,
                root / "runs",
                self.config,
            )
            info_path = dataset / "meta" / "info.json"
            stat = info_path.stat()
            os.utime(info_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

            executed = run_manifest(registry, manifest, self.config)
            self.assertEqual(executed["stale"], 1)
            self.assertEqual(executed["succeeded"], 0)

    def test_config_change_with_same_standard_creates_new_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "cpfs" / "batch-001")
            registry = root / "registry.sqlite"
            register_datasets(
                registry,
                [dataset],
                source="cpfs-dev",
                video_key=self.config["video_key"],
                source_root=root / "cpfs",
            )
            first_manifest = root / "first.jsonl"
            create_manifest(registry, first_manifest, root / "runs", self.config)
            run_manifest(registry, first_manifest, self.config)

            changed_config = json.loads(json.dumps(self.config))
            changed_config["thresholds"]["position_warning_m"] = 0.019
            second_manifest = root / "second.jsonl"
            planned = create_manifest(
                registry,
                second_manifest,
                root / "runs",
                changed_config,
            )
            self.assertEqual(planned["task_count"], 1)
            first_task = json.loads(first_manifest.read_text().strip())
            second_task = json.loads(second_manifest.read_text().strip())
            self.assertNotEqual(first_task["config_hash"], second_task["config_hash"])

    def test_shared_cache_survives_metadata_only_dataset_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "cpfs" / "batch-001")
            registry = root / "registry.sqlite"
            register_datasets(
                registry,
                [dataset],
                source="cpfs-dev",
                video_key=self.config["video_key"],
                source_root=root / "cpfs",
            )
            first_manifest = root / "first.jsonl"
            create_manifest(registry, first_manifest, root / "runs", self.config)
            run_manifest(registry, first_manifest, self.config)

            tasks_path = dataset / "meta" / "tasks.parquet"
            stat = tasks_path.stat()
            os.utime(
                tasks_path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
            register_datasets(
                registry,
                [dataset],
                source="cpfs-dev",
                video_key=self.config["video_key"],
                source_root=root / "cpfs",
            )
            second_manifest = root / "second.jsonl"
            create_manifest(registry, second_manifest, root / "runs", self.config)
            executed = run_manifest(registry, second_manifest, self.config)
            self.assertEqual(executed["succeeded"], 1)
            second_task = json.loads(second_manifest.read_text().strip())
            summary = json.loads(
                (Path(second_task["output_path"]) / "summary.json").read_text()
            )
            self.assertEqual(summary["cache"]["parquet_hits"], 1)
            self.assertEqual(summary["cache"]["video_hits"], 1)

    def test_active_lease_prevents_duplicate_worker_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "cpfs" / "batch-001")
            registry = root / "registry.sqlite"
            register_datasets(
                registry,
                [dataset],
                source="cpfs-dev",
                video_key=self.config["video_key"],
                source_root=root / "cpfs",
            )
            manifest = root / "manifest.jsonl"
            create_manifest(registry, manifest, root / "runs", self.config)
            task = json.loads(manifest.read_text().strip())
            db = _connect(registry)
            try:
                self.assertEqual(
                    _claim_task(db, task, "worker-a", lease_seconds=3600),
                    "claimed",
                )
            finally:
                db.close()

            executed = run_manifest(
                registry,
                manifest,
                self.config,
                worker_id="worker-b",
            )
            self.assertEqual(executed["busy"], 1)
            self.assertEqual(executed["succeeded"], 0)

    def test_v01_registry_schema_is_migrated(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry.sqlite"
            db = sqlite3.connect(str(registry))
            db.execute(
                """
                CREATE TABLE runs (
                  run_id TEXT PRIMARY KEY,
                  task_id TEXT NOT NULL,
                  dataset_id TEXT NOT NULL,
                  dataset_signature TEXT NOT NULL,
                  standard_version TEXT NOT NULL,
                  output_path TEXT NOT NULL,
                  status TEXT NOT NULL,
                  started_at TEXT,
                  finished_at TEXT,
                  error TEXT,
                  summary_json TEXT
                )
                """
            )
            db.commit()
            db.close()

            migrated = _connect(registry)
            try:
                columns = {
                    row[1]
                    for row in migrated.execute("PRAGMA table_info(runs)").fetchall()
                }
            finally:
                migrated.close()
            self.assertTrue(
                {
                    "config_hash",
                    "code_version",
                    "claimed_by",
                    "lease_expires_at",
                    "heartbeat_at",
                }.issubset(columns)
            )

    def test_multiple_batches_run_in_parallel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mount = root / "cpfs"
            datasets = [
                create_fixture(mount / f"batch-{index:03d}", episodes=2)
                for index in range(3)
            ]
            registry = root / "registry.sqlite"
            register_datasets(
                registry,
                datasets,
                source="cpfs-dev",
                video_key=self.config["video_key"],
                source_root=mount,
            )
            manifest = root / "parallel.jsonl"
            planned = create_manifest(
                registry,
                manifest,
                root / "runs",
                self.config,
            )
            self.assertEqual(planned["task_count"], 3)
            executed = run_manifest(
                registry,
                manifest,
                self.config,
                workers=3,
            )
            self.assertEqual(executed["succeeded"], 3)
            self.assertEqual(executed["workers"], 3)
            results = [
                json.loads(line)
                for line in (root / "parallel.results.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertTrue(
                    (Path(result["output_path"]) / "episodes.parquet").exists()
                )

    def test_optional_finalized_marker_blocks_partial_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "cpfs" / "batch-001")
            registry = root / "registry.sqlite"
            with self.assertRaises(FileNotFoundError):
                register_datasets(
                    registry,
                    [dataset],
                    source="cpfs-dev",
                    video_key=self.config["video_key"],
                    source_root=root / "cpfs",
                    finalized_marker="_SUCCESS",
                )
            (dataset / "_SUCCESS").write_text("", encoding="utf-8")
            result = register_datasets(
                registry,
                [dataset],
                source="cpfs-dev",
                video_key=self.config["video_key"],
                source_root=root / "cpfs",
                finalized_marker="_SUCCESS",
            )
            self.assertEqual(result["registered"][0]["missing_file_count"], 0)


if __name__ == "__main__":
    unittest.main()
