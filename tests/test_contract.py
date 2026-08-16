import json
import unittest
from pathlib import Path


class EgoScaleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "config" / "egoscale_v3.contract.json"
        cls.contract = json.loads(path.read_text(encoding="utf-8"))

    def test_contract_preserves_tri_state_measurement_semantics(self):
        self.assertIn("null", self.contract["metric_statuses"])
        self.assertIn("not_applicable", self.contract["metric_statuses"])
        self.assertEqual(
            self.contract["policy"]["missing_ground_truth_metric"], "null"
        )
        self.assertFalse(self.contract["policy"]["reference_metric_can_reject"])

    def test_ground_truth_metrics_declare_dependency(self):
        metrics = self.contract["hard_metrics"]
        for name in (
            "hand_mpjpe",
            "thumb_tip_mpjpe",
            "index_tip_mpjpe",
            "middle_tip_mpjpe",
            "ring_tip_mpjpe",
            "pinky_tip_mpjpe",
            "odsr_micro_f1",
        ):
            self.assertTrue(metrics[name]["requires_ground_truth"], name)

    def test_world_modes_are_explicitly_alternative(self):
        world = self.contract["coordinate_systems"]["world_mode"]
        self.assertTrue(world["required_declaration"])
        self.assertEqual(
            set(world["allowed"]), {"first_valid_camera", "gravity_aligned"}
        )

    def test_sampling_is_reproducible(self):
        sampling = self.contract["sampling"]
        self.assertGreaterEqual(sampling["minimum_random_episodes"], 1000)
        self.assertTrue(sampling["seed_required"])
        self.assertTrue(sampling["sample_manifest_required"])


if __name__ == "__main__":
    unittest.main()
