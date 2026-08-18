import json
import tempfile
import unittest
from pathlib import Path

from egoqc.task_taxonomy import classify_task, classify_task_records


class TaskTaxonomyTests(unittest.TestCase):
    def test_multilabel_classification_does_not_invent_scene(self):
        taxonomy = json.loads(Path("config/task_taxonomy.json").read_text())
        label = classify_task("Insert and screw the bottle cap", taxonomy)
        self.assertIn("insert_remove", label["interaction_primitives"])
        self.assertIn("screw_unscrew", label["interaction_primitives"])
        self.assertIn("container", label["object_affordances"])
        self.assertEqual(label["manipulation_scale"], "fine")
        self.assertEqual(label["scene_type"], "unknown")
        self.assertEqual(label["temporal_complexity"], "composite")

    def test_unknown_task_routes_to_semantic_review(self):
        taxonomy = json.loads(Path("config/task_taxonomy.json").read_text())
        label = classify_task("Do the special thing", taxonomy)
        self.assertEqual(label["interaction_primitives"], ["unknown"])
        self.assertTrue(label["requires_semantic_review"])

    def test_extended_egocentric_task_families_are_covered(self):
        taxonomy = json.loads(Path("config/task_taxonomy.json").read_text())
        examples = {
            "boil_serve_egg": "food_preparation",
            "braid_unbraid": "personal_care",
            "charge_uncharge_airpods": "connect_disconnect",
            "play_piano": "play_game_instrument",
            "type_keyboard": "input_control",
            "zip_unzip_bag": "zip_unzip",
        }
        for task, expected in examples.items():
            with self.subTest(task=task):
                label = classify_task(task, taxonomy)
                self.assertIn(expected, label["interaction_primitives"])
                self.assertFalse(label["requires_semantic_review"])

    def test_batch_enriches_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "records.jsonl"
            output = root / "output"
            source.write_text(json.dumps({"task_group": "Fold_Paper"}) + "\n")
            summary = classify_task_records(
                source,
                Path("config/task_taxonomy.json"),
                output,
                task_field="task_group",
                source_id="test",
            )
            self.assertEqual(summary["classified_tasks"], 1)
            row = json.loads((output / "records-with-taxonomy.jsonl").read_text())
            self.assertIn("fold_unfold", row["task_taxonomy"]["interaction_primitives"])
            self.assertEqual(
                (output / "semantic-review-tasks.jsonl").read_text(), ""
            )

    def test_unknown_batch_writes_text_only_semantic_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "records.jsonl"
            output = root / "output"
            source.write_text(json.dumps({"task_group": "Special_Action_X"}) + "\n")
            summary = classify_task_records(
                source,
                Path("config/task_taxonomy.json"),
                output,
                task_field="task_group",
                source_id="test",
            )
            queue = json.loads((output / "semantic-review-tasks.jsonl").read_text())
            self.assertEqual(summary["semantic_review_tasks"], 1)
            self.assertFalse(summary["semantic_review_video_required"])
            self.assertFalse(queue["video_required"])
            self.assertEqual(queue["status"], "pending")


if __name__ == "__main__":
    unittest.main()
