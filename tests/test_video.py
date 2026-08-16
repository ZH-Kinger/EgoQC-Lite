import tempfile
import unittest
from pathlib import Path

from create_fixture import create_fixture
from egoqc.video import probe_video


class VideoProbeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
