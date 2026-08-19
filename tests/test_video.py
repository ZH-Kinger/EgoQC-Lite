import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from create_fixture import create_fixture

from egoqc.extract import _decode_requested_frames, _frame_index_from_pts
from egoqc.video import probe_video


class VideoProbeTests(unittest.TestCase):
    def test_half_frame_pts_does_not_collapse_adjacent_indices(self):
        time_base = Fraction(1, 15360)
        # 512 ticks/frame at 30 FPS, shifted by half a frame (256 ticks).
        self.assertEqual(_frame_index_from_pts(256, 0, time_base, 30.0), 0)
        self.assertEqual(_frame_index_from_pts(768, 0, time_base, 30.0), 1)
        self.assertEqual(_frame_index_from_pts(1280, 0, time_base, 30.0), 2)

    def test_header_mode_does_not_claim_exact_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset", frames=7)
            video = root / "videos/observation.images.ego/chunk-000/file-000.mp4"
            metadata, issues = probe_video(video)
            self.assertEqual(metadata["check_mode"], "header")
            self.assertNotIn("counted_frames", metadata)
            self.assertFalse(issues)

    def test_count_mode_decodes_exact_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset", frames=7)
            video = root / "videos/observation.images.ego/chunk-000/file-000.mp4"
            metadata, _ = probe_video(video, "count")
            self.assertEqual(metadata["counted_frames"], 7)
            self.assertIn("frame_interval_jitter_mean_ms", metadata)
            self.assertIn("frame_interval_jitter_max_ms", metadata)
            self.assertIn("frame_interval_jitter_events", metadata)
            self.assertEqual(metadata["non_monotonic_timestamps"], 0)

    def test_sample_quality_has_bounded_sample_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset", frames=12)
            video = root / "videos/observation.images.ego/chunk-000/file-000.mp4"
            metadata, _ = probe_video(video, "sample-quality", {"sample_frames": 4})
            self.assertGreater(metadata["quality_sample_count"], 0)
            self.assertLessEqual(metadata["quality_sample_count"], 4)

    def test_sparse_decoder_seeks_and_preserves_absolute_frame_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(
                Path(temporary) / "dataset", frames=12, episodes=3
            )
            video = root / "videos/observation.images.ego/chunk-000/file-000.mp4"
            images, statistics = _decode_requested_frames(
                video,
                [25, 35],
                30.0,
                seek=True,
                seek_margin_s=0.1,
                seek_min_frame=1,
            )
            self.assertEqual(set(images), {25, 35})
            self.assertTrue(statistics["seek_attempted"])
            self.assertEqual(statistics["missing_frames"], [])
            for image in images.values():
                image.close()


if __name__ == "__main__":
    unittest.main()
