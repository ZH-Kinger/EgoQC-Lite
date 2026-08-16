import unittest

from egoqc.decisions import acceptance_for
from egoqc.types import Issue


class DecisionPolicyTests(unittest.TestCase):
    def test_motion_failure_becomes_vendor_rework_in_procurement_mode(self):
        result = acceptance_for(
            [Issue("position_jitter", "warning", "too much jitter")],
            {"acceptance": {"motion_failure_decision": "rework"}},
        )
        self.assertFalse(result["motion_pass"])
        self.assertFalse(result["final_pass"])
        self.assertEqual(result["decision"], "rework")

    def test_warning_without_motion_failure_stays_review(self):
        result = acceptance_for(
            [Issue("video_blur_candidate", "warning", "inspect")],
            {"acceptance": {"motion_failure_decision": "rework"}},
        )
        self.assertEqual(result["decision"], "review")


if __name__ == "__main__":
    unittest.main()
