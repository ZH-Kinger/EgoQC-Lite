import json
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from egoqc.adapters import detect_adapter, inspect_adapter
from egoqc.canonical import CapabilityManifest, plan_use_cases, route_capabilities
from egoqc.generic_ego import build_generic_ego_views


def _video(path: Path, frames: int = 6) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class GenericEgoTests(unittest.TestCase):
    def test_video_only_source_enables_visual_but_not_mano_metrics(self):
        route = route_capabilities(CapabilityManifest(video=True, video_timestamps=True))
        self.assertTrue(route["enabled_stages"]["sparse_visual_qc"])
        self.assertTrue(route["training_objectives"]["video_representation"])
        self.assertFalse(route["enabled_stages"]["mano_visual_reprojection_qc"])
        self.assertIn("MPJPE", route["unavailable_metrics"])
        self.assertFalse(route["missing_optional_modalities_are_failures"])

    def test_use_case_planner_separates_pretraining_from_robot_imitation(self):
        video_only = plan_use_cases(CapabilityManifest(video=True, video_timestamps=True))
        self.assertEqual(video_only["video_self_supervised_pretraining"]["status"], "ready")
        self.assertEqual(video_only["robot_imitation_learning"]["status"], "blocked")
        robot = plan_use_cases(CapabilityManifest(
            video=True,
            video_timestamps=True,
            independent_timestamps=True,
            robot_action=True,
            robot_state=True,
            task_labels=True,
            camera_intrinsics=True,
        ))
        self.assertEqual(robot["robot_imitation_learning"]["status"], "ready")

    def test_raw_video_adapter_uses_optional_sidecar_capabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "clip.mp4"
            _video(path)
            path.with_suffix(".json").write_text(json.dumps({
                "task": "pick up the cup",
                "person_id": "person-1",
                "intrinsics": [100, 0, 32, 0, 100, 24, 0, 0, 1],
            }), encoding="utf-8")
            self.assertEqual(detect_adapter(path), "raw_video")
            report = inspect_adapter(path, None)
            self.assertTrue(report["compatible"])
            self.assertTrue(report["capabilities"]["task_labels"])
            self.assertTrue(report["capabilities"]["camera_intrinsics"])
            self.assertTrue(report["capability_route"]["enabled_stages"]["task_semantic_qc"])
            self.assertFalse(report["capabilities"]["mano_parameters"])
            self.assertEqual(
                report["use_case_eligibility"]["video_language_pretraining"]["status"],
                "ready",
            )

    def test_builds_capability_aware_manifest_without_writing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "raw"
            output = base / "derived"
            nested = source / "kitchen"
            nested.mkdir(parents=True)
            path = nested / "episode-1.mp4"
            _video(path)
            path.with_suffix(".json").write_text(json.dumps({
                "task_label": "place cup on table",
                "supplier_id": "supplier-a",
                "collection_session_id": "session-1",
            }), encoding="utf-8")

            report = build_generic_ego_views(
                source,
                output,
                source_dataset="generic-pilot",
                source_class="supplier_dataset",
                license_id="internal-approval-1",
                workers=2,
            )
            self.assertEqual(report["records"], 1)
            self.assertEqual(report["errors"], 0)
            self.assertEqual(report["training_ready"], 1)
            row = json.loads((output / "generic-ego.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["video_id"], "kitchen/episode-1")
            self.assertTrue(row["vla_pretraining"]["loss_masks"]["video_representation"])
            self.assertTrue(row["vla_pretraining"]["loss_masks"]["video_text_alignment"])
            self.assertFalse(row["vla_pretraining"]["loss_masks"]["mano_motion"])
            self.assertTrue(row["provenance"]["raw_immutable"])
            self.assertEqual(
                row["use_case_eligibility"]["robot_imitation_learning"]["status"],
                "blocked",
            )
            self.assertTrue((output / "generic-ego.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
