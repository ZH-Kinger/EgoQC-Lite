from __future__ import annotations

import unittest

import numpy as np

import json
import tempfile
from pathlib import Path

from egoqc.temporal import analyze_temporal_quality
from egoqc.temporal_plot import write_temporal_plot

from create_fixture import create_fixture


class TemporalQualityTests(unittest.TestCase):
    def _inputs(self, frames=120):
        time = np.arange(frames, dtype=np.float64) / 30.0
        left = np.column_stack([0.05 * time, np.zeros(frames), np.ones(frames)])
        right = left + np.array([0.2, 0.0, 0.0])
        orient = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
        pose = np.broadcast_to(np.eye(3), (frames, 15, 3, 3)).copy()
        mask = np.ones((frames, 2), dtype=bool)
        extr = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
        return left, right, orient, pose, mask, extr

    def _run(self, values, timestamps=None):
        left, right, orient, pose, mask, extr = values
        return analyze_temporal_quality(
            left,
            right,
            orient,
            orient.copy(),
            pose,
            pose.copy(),
            mask,
            extr,
            30.0,
            {"thresholds": {}, "temporal": {"robust_z": 6.0}},
            0,
            "fixture.parquet",
            timestamps=timestamps,
        )

    def test_smooth_constant_velocity_is_clean(self):
        metrics, issues, frames, bad_frames = self._run(self._inputs())
        self.assertAlmostEqual(metrics["left_position_jitter_p99_mm"], 0.0, places=6)
        self.assertNotIn("temporal_spike", {issue.code for issue in issues})
        self.assertEqual(frames, [])
        self.assertEqual(bad_frames, [])

    def test_single_frame_jump_is_detected_and_sampled(self):
        values = self._inputs()
        values[0][57, 1] += 0.10
        metrics, issues, frames, bad_frames = self._run(values)
        self.assertGreaterEqual(metrics["left_temporal_spike_count"], 1)
        self.assertIn("temporal_spike", {issue.code for issue in issues})
        self.assertIn(57, frames)
        self.assertTrue(any(event["frame_index"] == 57 for event in bad_frames))

    def test_short_mask_gap_is_flicker(self):
        values = self._inputs()
        values[4][40:42, 0] = False
        metrics, issues, frames, bad_frames = self._run(values)
        self.assertEqual(metrics["left_mask_flicker_count"], 1)
        self.assertIn("mask_flicker", {issue.code for issue in issues})
        self.assertIn(40, frames)
        self.assertEqual(
            [event["frame_index"] for event in bad_frames if event["code"] == "mask_flicker"],
            [40, 41],
        )

    def test_velocity_uses_variable_timestamp_delta(self):
        values = self._inputs(frames=8)
        timestamps = np.array([0.0, 0.03, 0.07, 0.10, 0.16, 0.19, 0.23, 0.30])
        values[0][:, 0] = timestamps * 0.5
        values[1][:, 0] = timestamps * 0.5 + 0.2
        metrics, issues, _, bad_frames = self._run(values, timestamps=timestamps)
        self.assertAlmostEqual(metrics["left_velocity_median_m_s"], 0.5)
        self.assertEqual(metrics["left_velocity_outlier_count"], 0)
        self.assertNotIn("instantaneous_velocity_outlier", {issue.code for issue in issues})
        self.assertFalse(any(event["code"] == "instantaneous_velocity_outlier" for event in bad_frames))

    def test_temporal_svg_is_generated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset")
            output = Path(temporary) / "temporal.svg"
            config = json.loads(
                (Path(__file__).parents[1] / "config" / "default.json").read_text()
            )
            summary = write_temporal_plot(root, 0, output, config)
            self.assertEqual(summary["frames"], 12)
            self.assertIn("Temporal QC", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
