import json
import tempfile
import unittest
from pathlib import Path

from egoqc.ops import doctor, self_test
from egoqc.dashboard import write_registry_dashboard
from egoqc.registry import registry_status


class OpsTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (Path(__file__).parents[1] / "config" / "default.json").read_text()
        )

    def test_doctor_checks_paths_config_and_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            result = doctor(
                self.config,
                registry_path=root / "control" / "registry.sqlite",
                source_root=source,
                output_root=root / "output",
            )
            self.assertTrue(result["ok"])
            self.assertTrue(all(check["ok"] for check in result["checks"]))

    def test_self_test_runs_complete_pipeline_and_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self_test(root, self.config)
            self.assertTrue(result["ok"])
            result_root = Path(result["result_root"])
            self.assertTrue((result_root / "episodes.parquet").exists())
            self.assertTrue(
                (result_root / "evidence" / "evidence_manifest.jsonl").exists()
            )

            status = registry_status(root / "control" / "registry.sqlite")
            self.assertEqual(status["dataset_count"], 1)
            self.assertEqual(status["run_counts"], {"succeeded": 1})
            dashboard_path = root / "dashboard.html"
            dashboard = write_registry_dashboard(
                root / "control" / "registry.sqlite",
                dashboard_path,
            )
            self.assertEqual(dashboard["dataset_count"], 1)
            document = dashboard_path.read_text(encoding="utf-8")
            self.assertIn("全局数据质量总览", document)
            self.assertIn('id="dataset-body"', document)


if __name__ == "__main__":
    unittest.main()
