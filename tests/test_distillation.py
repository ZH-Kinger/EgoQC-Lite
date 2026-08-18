import json
import tempfile
import unittest
from pathlib import Path

from egoqc.distillation import (
    audit_qc_training_data,
    build_distillation_manifest,
    evaluate_qc_predictions,
)


class DistillationTests(unittest.TestCase):
    def test_gold_overrides_teacher_and_programmatic_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.jsonl"
            records.write_text(json.dumps({
                "video_id": "v1", "person_id": "p0", "source_uri": "/video.mp4", "duration_s": 10, "fps": 30,
                "vla_pretraining": {"candidate": True, "training_ready": False, "split": "train",
                    "clip_sampler": {"window_s": 4, "decode_fps": 8}},
            }) + "\n")
            hand = root / "hands" / "v1"
            hand.mkdir(parents=True)
            (hand / "hand-screen.json").write_text(json.dumps({"metrics": {
                "longest_no_hand_gap_s": 2.0,
                "suspected_extra_hands_ratio": 0.02,
                "suspected_extra_hand_segments": [{"start_s": 1, "end_s": 2}],
            }}))
            teacher = root / "teacher" / "v1"
            teacher.mkdir(parents=True)
            (teacher / "teacher-label.json").write_text(json.dumps({
                "schema_version": "egoqc-visual-teacher-v1", "teacher_model": "teacher",
                "prompt_version": "v1", "tasks": {
                    "hand_absent": {"probability": 0.9, "confidence": 0.8},
                    "severe_occlusion": {"probability": 0.6, "confidence": 0.5},
                },
            }))
            gold = root / "gold.jsonl"
            gold.write_text(json.dumps({
                "video_id": "v1", "reviewer_id": "human", "labels": {"hand_absent": False}
            }) + "\n")
            output = root / "output"
            summary = build_distillation_manifest(
                records,
                Path("config/visual_model_tasks.json"),
                output,
                hand_screen_root=root / "hands",
                teacher_root=root / "teacher",
                gold_labels=gold,
            )
            self.assertEqual(summary["records"], 1)
            row = json.loads((output / "qc-distillation.jsonl").read_text())
            labels = row["distillation"]
            self.assertEqual(labels["targets"]["hand_absent"], 0.0)
            self.assertEqual(labels["label_sources"]["hand_absent"], "human_gold")
            self.assertEqual(labels["label_weights"]["hand_absent"], 1.0)
            self.assertEqual(labels["label_sources"]["severe_occlusion"], "local_vlm_teacher")
            self.assertEqual(labels["label_weights"]["severe_occlusion"], 0.25)
            self.assertFalse(labels["acceptance_authority"])
            self.assertEqual(labels["split_group_source"], "person_id")
            self.assertEqual(labels["leakage_risk"], "low")

    def test_training_audit_detects_group_and_uri_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            rows = []
            for index, split in enumerate(("train", "test")):
                rows.append({
                    "video_id": f"v{index}",
                    "source_uri": "/same-source.mp4",
                    "supplier_id": "supplier-a",
                    "vla_pretraining": {"training_ready": True},
                    "distillation": {
                        "schema_version": "egoqc-qc-distillation-v1",
                        "split": split,
                        "split_group": "person:shared",
                        "split_group_source": "person_id",
                        "leakage_risk": "low",
                        "tasks": ["hand_absent"],
                        "targets": {"hand_absent": float(index == 1)},
                        "label_masks": {"hand_absent": 1},
                        "label_sources": {"hand_absent": "human_gold"},
                        "label_details": {"hand_absent": {"reviewer_id": "reviewer"}},
                    },
                })
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = audit_qc_training_data(
                manifest,
                Path("config/visual_model_tasks.json"),
                root / "audit",
            )
            self.assertFalse(report["ready_for_production_training"])
            self.assertIn("cross_split_leakage", report["global_block_reasons"])
            blockers = (root / "audit" / "qc-training-blockers.jsonl").read_text()
            self.assertIn("split_group_leakage", blockers)
            self.assertIn("source_uri_leakage", blockers)

    def test_gold_gate_enables_only_well_covered_high_precision_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            gold = root / "gold.jsonl"
            prediction_rows = []
            gold_rows = []
            for index in range(1000):
                positive = index < 500
                prediction_rows.append({
                    "video_id": f"v{index}",
                    "probabilities": {"hand_absent": 0.99 if positive else 0.01},
                })
                gold_rows.append({"video_id": f"v{index}", "labels": {"hand_absent": positive}})
            predictions.write_text("".join(json.dumps(row) + "\n" for row in prediction_rows))
            gold.write_text("".join(json.dumps(row) + "\n" for row in gold_rows))
            report = evaluate_qc_predictions(
                predictions, gold, Path("config/visual_model_tasks.json"), root / "evaluation"
            )
            self.assertTrue(report["tasks"]["hand_absent"]["auto_reject_enabled"])
            self.assertEqual(report["tasks"]["hand_absent"]["precision"], 1.0)
            self.assertGreaterEqual(
                report["tasks"]["hand_absent"]["precision_95_lower_bound"], 0.99
            )
            self.assertFalse(report["tasks"]["severe_occlusion"]["auto_reject_enabled"])

    def test_small_perfect_gold_set_does_not_claim_99_percent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            gold = root / "gold.jsonl"
            prediction_rows = []
            gold_rows = []
            for index in range(120):
                positive = index < 60
                prediction_rows.append({
                    "video_id": f"v{index}",
                    "probabilities": {"hand_absent": 0.99 if positive else 0.01},
                })
                gold_rows.append({"video_id": f"v{index}", "labels": {"hand_absent": positive}})
            predictions.write_text("".join(json.dumps(row) + "\n" for row in prediction_rows))
            gold.write_text("".join(json.dumps(row) + "\n" for row in gold_rows))
            report = evaluate_qc_predictions(
                predictions,
                gold,
                Path("config/visual_model_tasks.json"),
                root / "evaluation",
            )
            task = report["tasks"]["hand_absent"]
            self.assertEqual(task["empirical_precision"], 1.0)
            self.assertLess(task["precision_95_lower_bound"], 0.99)
            self.assertFalse(task["auto_reject_enabled"])
            self.assertIn("precision_confidence_bound_below_target", task["block_reasons"])


if __name__ == "__main__":
    unittest.main()
