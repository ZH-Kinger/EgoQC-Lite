import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from egoqc.training_views import build_rekadaily_training_views


class RekaDailyTrainingViewTests(unittest.TestCase):
    def test_two_views_and_strict_mano_stage_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reka"
            output = Path(temporary) / "views"
            (root / "metadata").mkdir(parents=True)
            rows = [
                {"video_id": "good", "project": "home", "duration_s": 12.0, "fps": 30.0,
                 "width": 1280, "height": 720, "codec": "h264", "src_ext": "mp4",
                 "activities": ["pick up cup"], "file_size_bytes": 100},
                {"video_id": "lowfps", "project": "home", "duration_s": 12.0, "fps": 24.0,
                 "width": 1280, "height": 720, "codec": "h264", "src_ext": "mp4",
                 "activities": None, "file_size_bytes": 100},
                {"video_id": "tarred", "project": "home", "duration_s": 8.0, "fps": 30.0,
                 "width": 1920, "height": 1080, "codec": "h264", "src_ext": "mov",
                 "activities": None, "file_size_bytes": 100},
                {"video_id": "missing", "project": "home", "duration_s": 8.0, "fps": 30.0,
                 "width": 1920, "height": 1080, "codec": "h264", "src_ext": "mp4",
                 "activities": None, "file_size_bytes": 100},
            ]
            pq.write_table(pa.Table.from_pylist(rows), root / "metadata" / "index.parquet")
            loose = root / "sample" / "home" / "good.mp4"
            loose.parent.mkdir(parents=True)
            loose.write_bytes(b"not-opened")
            lowfps = loose.with_name("lowfps.mp4")
            lowfps.write_bytes(b"not-opened")
            shard = root / "data" / "home" / "shard-00000.tar"
            shard.parent.mkdir(parents=True)
            with tarfile.open(shard, "w") as archive:
                info = tarfile.TarInfo("tarred.mov")
                info.size = 4
                archive.addfile(info, io.BytesIO(b"data"))

            hand_root = Path(temporary) / "hands"
            (hand_root / "good").mkdir(parents=True)
            (hand_root / "good" / "hand-screen.json").write_text(json.dumps({
                "metrics": {"provisional_decision": "candidate_for_mano"}
            }))
            mano_root = Path(temporary) / "mano"
            (mano_root / "good").mkdir(parents=True)
            (mano_root / "good" / "mano-fit.json").write_text(json.dumps({
                "status": "succeeded",
                "capabilities": {"wrist_pose": True, "mano_pose": True, "betas": True, "state_mask": True},
            }))
            alignment_root = Path(temporary) / "alignment"
            (alignment_root / "good").mkdir(parents=True)
            (alignment_root / "good" / "alignment-qc.json").write_text(json.dumps({
                "decision": "accept", "human_reviewed": True,
            }))

            summary = build_rekadaily_training_views(
                root, output, materialized_only=True, hand_screen_root=hand_root,
                mano_root=mano_root, alignment_root=alignment_root, license_id="approved-test",
            )
            self.assertEqual(summary["records_written"], 3)
            self.assertEqual(summary["video_pretrain"]["technical_candidates"], 2)
            self.assertEqual(summary["video_pretrain"]["training_ready"], 2)
            self.assertEqual(summary["mano_silver"]["training_ready"], 1)
            records = [json.loads(line) for line in (output / "all-records.jsonl").read_text().splitlines()]
            by_id = {row["video_id"]: row for row in records}
            self.assertEqual(by_id["good"]["mano_silver"]["stage"], "eligible_mano_silver")
            self.assertEqual(by_id["good"]["capability_class"], "rgb_mano")
            self.assertIn("pick_place", by_id["good"]["task_taxonomy"]["interaction_primitives"])
            self.assertEqual(
                by_id["good"]["annotation_provenance"]["mano"],
                "derived_silver_prediction_human_approved",
            )
            self.assertFalse(by_id["good"]["annotation_provenance"]["mano_is_ground_truth"])
            self.assertIn("fps_below_29_9", by_id["lowfps"]["video_pretrain"]["reason_codes"])
            self.assertTrue(by_id["tarred"]["video_pretrain"]["needs_transcode"])
            self.assertEqual(by_id["good"]["vla_pretraining"]["loss_masks"]["robot_action"], 0)
            self.assertEqual(by_id["good"]["vla_pretraining"]["loss_masks"]["video_text_alignment"], 1)
            self.assertEqual(by_id["tarred"]["vla_pretraining"]["loss_masks"]["video_text_alignment"], 0)
            self.assertIn("mano_motion_modeling", by_id["good"]["vla_pretraining"]["allowed_objectives"])
            self.assertIn(by_id["good"]["vla_pretraining"]["split"], {"train", "validation", "test"})
            self.assertEqual(summary["classification"]["capability_class_counts"]["rgb_mano"], 1)
            self.assertEqual(by_id["good"]["task_taxonomy"]["task_domains"], ["unknown"])
            rerun = build_rekadaily_training_views(root, output, materialized_only=True)
            self.assertEqual(rerun["inventory"]["tar_cache_hits"], 1)
            self.assertEqual(rerun["inventory"]["tar_shards_scanned"], 0)

    def test_license_is_required_for_ready_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "reka"
            (root / "metadata").mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([{
                "video_id": "v1", "project": "p", "duration_s": 8.0, "fps": 30.0,
                "width": 1280, "height": 720, "src_ext": "mp4", "codec": "h264",
            }]), root / "metadata" / "index.parquet")
            video = root / "sample" / "p" / "v1.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"x")
            output = Path(temporary) / "views"
            summary = build_rekadaily_training_views(root, output, materialized_only=True)
            self.assertEqual(summary["video_pretrain"]["technical_candidates"], 1)
            self.assertEqual(summary["video_pretrain"]["training_ready"], 0)
            self.assertEqual(summary["vla_pretraining"]["training_ready"], 0)
            self.assertEqual((output / "video-pretrain-ready.jsonl").read_text(), "")
            candidate = json.loads((output / "vla-pretrain-candidates.jsonl").read_text())
            self.assertEqual(candidate["vla_pretraining"]["loss_masks"]["mano_motion"], 0)
            self.assertEqual(candidate["vla_pretraining"]["loss_masks"]["robot_action"], 0)

    def test_invalid_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            build_rekadaily_training_views(Path("missing"), Path("out"), limit=0)


if __name__ == "__main__":
    unittest.main()
