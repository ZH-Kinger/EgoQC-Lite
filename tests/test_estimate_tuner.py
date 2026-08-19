import json
import tempfile
import unittest
from pathlib import Path

from create_fixture import create_fixture
from egoqc.estimate import estimate_manifest
from egoqc.pipeline import run
from egoqc.registry import create_manifest, register_datasets
from egoqc.tuner import write_tuner
from egoqc.decisions import create_retry_plan


class EstimateAndTunerTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (Path(__file__).parents[1] / "config" / "default.json").read_text()
        )

    def test_estimate_manifest_reports_cold_and_warm_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "mount" / "batch")
            registry = root / "registry.sqlite"
            register_datasets(
                registry,
                [dataset],
                "cpfs",
                self.config["video_key"],
                root / "mount",
            )
            manifest = root / "run.jsonl"
            create_manifest(registry, manifest, root / "results", self.config)
            estimate = estimate_manifest(registry, manifest, self.config, workers=4)
            self.assertEqual(estimate["task_count"], 1)
            self.assertEqual(estimate["effective_dataset_workers"], 1)
            self.assertGreater(estimate["cold"]["high_s"], estimate["cold"]["low_s"])
            self.assertIn("video", estimate["by_kind"])

    def test_tuner_is_offline_and_downloads_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "dataset")
            quality = root / "quality"
            run(dataset, quality, self.config)
            output = root / "tuner.html"
            summary = write_tuner(quality, self.config, output)
            document = output.read_text(encoding="utf-8")
            self.assertEqual(summary["episode_count"], 1)
            self.assertIn("交互式阈值调优", document)
            self.assertIn("position_jitter_error_m", document)
            self.assertIn('addControl("timing","bad_frame_ratio_max"', document)
            self.assertIn("bad_frame_ratio=", document)
            self.assertIn("下载版本化配置", document)

    def test_retry_plan_deduplicates_failed_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            quality = root / "quality" / "decisions"
            quality.mkdir(parents=True)
            row = {
                "dataset": "/path/to/example",
                "file": "data/chunk-000/file-001.parquet",
                "kind": "parquet",
                "episode_indices": [1, 2],
                "issue_codes": ["data_read_failed"],
            }
            (quality / "retry_files.jsonl").write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            output = root / "retry.jsonl"
            summary = create_retry_plan([root / "quality"], output)
            self.assertEqual(summary["task_count"], 1)
            self.assertIn("rerun_dataset_with_artifact_cache", output.read_text())


if __name__ == "__main__":
    unittest.main()
