import json
import tempfile
import unittest
from pathlib import Path

from egoqc.research_evaluation import evaluate_qc_research_protocol


class ResearchEvaluationTests(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    @staticmethod
    def _config(path: Path) -> None:
        path.write_text(json.dumps({
            "model_tasks": {
                "hand_absent": {
                    "minimum_auto_reject_precision": 0.99,
                }
            }
        }), encoding="utf-8")

    def test_threshold_is_frozen_on_validation_and_test_is_reported_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "tasks.json"
            self._config(config)
            validation_gold = []
            validation_predictions = []
            test_gold = []
            test_predictions = []
            for index in range(2500):
                positive = index < 500
                validation_gold.append({
                    "video_id": f"validation-{index}",
                    "person_id": f"validation-person-{index // 10}",
                    "supplier_id": "supplier-validation",
                    "camera_id": "camera-a",
                    "source_dataset": "validation-set",
                    "reviewer_id": "reviewer-a",
                    "reviewed_at": "2026-08-19T00:00:00Z",
                    "label_version": "egoqc-visual-gold-v1",
                    "labels": {"hand_absent": positive},
                })
                validation_predictions.append({
                    "video_id": f"validation-{index}",
                    "probabilities": {"hand_absent": 0.9 if positive else 0.1},
                })
                test_gold.append({
                    "video_id": f"test-{index}",
                    "person_id": f"test-person-{index // 10}",
                    "supplier_id": "supplier-test",
                    "camera_id": "camera-b",
                    "source_dataset": "test-set",
                    "reviewer_id": "reviewer-b",
                    "reviewed_at": "2026-08-19T00:00:00Z",
                    "label_version": "egoqc-visual-gold-v1",
                    "labels": {"hand_absent": positive},
                })
                test_predictions.append({
                    "video_id": f"test-{index}",
                    "probabilities": {"hand_absent": 0.92 if positive else 0.08},
                })
            files = {
                "validation_gold": validation_gold,
                "validation_predictions": validation_predictions,
                "test_gold": test_gold,
                "test_predictions": test_predictions,
            }
            for name, rows in files.items():
                self._write_jsonl(root / f"{name}.jsonl", rows)

            report = evaluate_qc_research_protocol(
                root / "validation_predictions.jsonl",
                root / "validation_gold.jsonl",
                root / "test_predictions.jsonl",
                root / "test_gold.jsonl",
                config,
                root / "output",
                bootstrap_replicates=50,
                minimum_group_samples=10,
            )

            task = report["tasks"]["hand_absent"]
            self.assertTrue(report["protocol_valid"])
            self.assertEqual(task["validation"]["selected_threshold"], 0.9)
            self.assertEqual(task["test"]["operating_point"]["precision"], 1.0)
            self.assertGreaterEqual(
                task["test"]["operating_point"]["precision_95_ci"][0], 0.99
            )
            self.assertTrue(task["auto_reject_authorized"])
            self.assertTrue((root / "output" / "qc-research-per-group.jsonl").is_file())

    def test_identity_overlap_blocks_publication_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "tasks.json"
            self._config(config)
            for split in ("validation", "test"):
                self._write_jsonl(root / f"{split}-gold.jsonl", [{
                    "video_id": f"{split}-video",
                    "person_id": "shared-person",
                    "supplier_id": "supplier-a",
                    "reviewer_id": "reviewer-a",
                    "reviewed_at": "2026-08-19T00:00:00Z",
                    "label_version": "egoqc-visual-gold-v1",
                    "labels": {"hand_absent": True},
                }])
                self._write_jsonl(root / f"{split}-predictions.jsonl", [{
                    "video_id": f"{split}-video",
                    "probabilities": {"hand_absent": 0.99},
                }])

            report = evaluate_qc_research_protocol(
                root / "validation-predictions.jsonl",
                root / "validation-gold.jsonl",
                root / "test-predictions.jsonl",
                root / "test-gold.jsonl",
                config,
                root / "output",
                bootstrap_replicates=0,
            )
            self.assertFalse(report["protocol_valid"])
            self.assertIn(
                "validation_test_identity_group_overlap", report["protocol_blockers"]
            )


if __name__ == "__main__":
    unittest.main()
