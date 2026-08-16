import unittest
from pathlib import Path

from egoqc.hand_screen import _balanced_groups, summarize_hand_samples


def sample(time_s, count):
    return {
        "time_s": time_s,
        "hand_count": count,
        "edge_touch": False,
        "detections": [[0, 0, 1, 1, 0.9, 0]] if count else [],
    }


def extra_sample(time_s, confidence=0.9):
    return {
        "time_s": time_s,
        "hand_count": 3,
        "edge_touch": False,
        "detections": [
            [0, 0, 10, 10, confidence, 0],
            [20, 0, 30, 10, confidence, 1],
            [40, 0, 50, 10, confidence, 0],
        ],
    }


class HandScreenTests(unittest.TestCase):
    def test_duration_balancer_does_not_put_two_longest_in_same_group(self):
        class Adapter:
            def _row(self, dataset, video_id):
                return {"duration_s": {"long": 100, "medium": 80, "short": 10}[video_id]}

        groups = _balanced_groups(
            Adapter(), Path("."), ["short", "medium", "long"], workers=2
        )
        self.assertFalse(any("long" in group and "medium" in group for group in groups))

    def test_short_detector_flicker_is_bridged(self):
        rows = [sample(index * 0.2, 1) for index in range(35)]
        rows[10] = sample(2.0, 0)
        metrics = summarize_hand_samples(rows, sample_fps=5)
        self.assertEqual(metrics["longest_no_hand_gap_s"], 0)
        self.assertEqual(metrics["provisional_decision"], "candidate_for_mano")

    def test_long_absence_and_effective_duration(self):
        rows = [sample(index * 0.2, 1 if index < 30 or index >= 40 else 0) for index in range(70)]
        metrics = summarize_hand_samples(rows, sample_fps=5)
        self.assertAlmostEqual(metrics["longest_no_hand_gap_s"], 2.0)
        self.assertEqual(len(metrics["long_no_hand_segments"]), 1)
        self.assertAlmostEqual(metrics["effective_video_duration_s"], 2.0)
        self.assertEqual(metrics["provisional_decision"], "review_before_mano")

    def test_video_without_five_second_run_is_screened_out(self):
        rows = [sample(index * 0.2, 1 if index < 20 else 0) for index in range(30)]
        metrics = summarize_hand_samples(rows, sample_fps=5)
        self.assertEqual(metrics["effective_video_duration_s"], 0)
        self.assertEqual(metrics["provisional_decision"], "screen_out_before_mano")

    def test_ego_wrist_at_bottom_edge_is_informational(self):
        rows = [sample(index * 0.2, 2) for index in range(35)]
        for row in rows:
            row["edge_touch"] = True
        metrics = summarize_hand_samples(rows, sample_fps=5)
        self.assertEqual(metrics["edge_touch_ratio"], 1.0)
        self.assertEqual(metrics["provisional_decision"], "candidate_for_mano")

    def test_transient_or_low_confidence_third_box_does_not_trigger_review(self):
        rows = [sample(index * 0.2, 2) for index in range(35)]
        rows[5] = extra_sample(1.0)
        rows[10] = extra_sample(2.0, confidence=0.5)
        metrics = summarize_hand_samples(rows, sample_fps=5)
        self.assertGreater(metrics["raw_extra_hands_ratio"], 0)
        self.assertEqual(metrics["suspected_extra_hand_segments"], [])
        self.assertEqual(metrics["provisional_decision"], "candidate_for_mano")

    def test_persistent_high_confidence_third_hand_triggers_review(self):
        rows = [sample(index * 0.2, 2) for index in range(35)]
        rows[5:9] = [extra_sample(index * 0.2) for index in range(5, 9)]
        metrics = summarize_hand_samples(rows, sample_fps=5)
        self.assertEqual(len(metrics["suspected_extra_hand_segments"]), 1)
        self.assertEqual(metrics["provisional_decision"], "review_before_mano")


if __name__ == "__main__":
    unittest.main()
