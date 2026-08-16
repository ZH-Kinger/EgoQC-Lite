import json
import tempfile
import unittest
from pathlib import Path

from create_fixture import create_fixture
from egoqc.undistortion import plan_vitra_undistortion, verify_vitra_undistortion


class UndistortionTests(unittest.TestCase):
    def test_plan_separates_selection_coverage_and_missing_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "raw"
            videos.mkdir()
            (videos / "a.mp4").write_bytes(b"a")
            (videos / "b.mp4").write_bytes(b"b")
            intrinsics = root / "intrinsics"
            intrinsics.mkdir()
            (intrinsics / "a.npy").write_bytes(b"calibration")
            selected = root / "selected.txt"
            selected.write_text("a\nb\n")
            summary = plan_vitra_undistortion(
                "ego4d", videos, intrinsics, root / "derived", root / "plan",
                selection_list=selected,
            )
            self.assertEqual(summary["source_population_videos"], 2)
            self.assertEqual(summary["selection_coverage_ratio"], 1.0)
            self.assertEqual(summary["ready"], 1)
            self.assertEqual(summary["blocked"], 1)
            blocked = json.loads((root / "plan" / "blocked.jsonl").read_text())
            self.assertIn("intrinsics_missing", blocked["reason_codes"])

    def test_verifier_requires_exact_frame_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dataset = create_fixture(root / "source-fixture", frames=7)
            output_dataset = create_fixture(root / "output-fixture", frames=6)
            source = source_dataset / "videos/observation.images.ego/chunk-000/file-000.mp4"
            destination = output_dataset / "videos/observation.images.ego/chunk-000/file-000.mp4"
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "task_id": "task", "video_id": "video", "dataset_kind": "ego4d",
                "source_video": str(source), "destination_video": str(destination),
                "source_fingerprint": {"size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns},
                "reason_codes": [],
            }) + "\n")
            summary = verify_vitra_undistortion(manifest, root / "verify")
            self.assertEqual(summary["integrity_fail"], 1)
            failure = json.loads((root / "verify" / "failures.jsonl").read_text())
            self.assertIn("frame_count_mismatch", failure["reason_codes"])


if __name__ == "__main__":
    unittest.main()
