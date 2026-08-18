import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from create_fixture import create_fixture
from egoqc.vla_dataset import VLAPretrainDataset, collate_vla_samples, smoke_vla_loader


class VLADatasetTests(unittest.TestCase):
    def test_candidate_requires_explicit_debug_switch_and_masks_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset", frames=12)
            video = root / "videos/observation.images.ego/chunk-000/file-000.mp4"
            manifest = Path(temporary) / "manifest.jsonl"
            row = {
                "video_id": "sample-1",
                "source_uri": str(video),
                "duration_s": 0.4,
                "fps": 30.0,
                "activities": ["pick cup"],
                "vla_pretraining": {
                    "candidate": True,
                    "training_ready": False,
                    "split": "train",
                    "allowed_objectives": ["video_representation", "video_text_alignment"],
                    "loss_masks": {
                        "video_representation": 1, "temporal_prediction": 1,
                        "video_text_alignment": 1, "hand_presence_auxiliary": 0,
                        "mano_motion": 0, "robot_action": 0, "camera_pose": 0, "tactile": 0,
                    },
                    "clip_sampler": {"window_s": 0.25, "decode_fps": 8.0},
                },
                "provenance": {"raw_immutable": True},
            }
            manifest.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(VLAPretrainDataset(manifest)), 0)
            dataset = VLAPretrainDataset(manifest, allow_technical_candidates=True)
            sample = dataset[0]
            self.assertEqual(sample["frames"].shape, (2, 224, 224, 3))
            self.assertEqual(sample["text"], "pick cup")
            self.assertEqual(sample["loss_masks"]["robot_action"], 0)
            batch = collate_vla_samples([sample, sample])
            self.assertEqual(batch["frames"].shape, (2, 2, 224, 224, 3))
            output = Path(temporary) / "smoke"
            summary = smoke_vla_loader(
                manifest, output, batch_size=1, allow_technical_candidates=True
            )
            self.assertEqual(summary["frames_shape"], [1, 2, 224, 224, 3])
            self.assertTrue((output / "vla-loader-contact-sheet.jpg").is_file())

            row["vla_pretraining"]["clip_sampler"] = {
                "mode": "fixed_reviewed_window",
                "fixed_start_s": 0.1,
                "window_s": 0.2,
                "decode_fps": 8.0,
            }
            manifest.write_text(json.dumps(row) + "\n")
            fixed_sample = VLAPretrainDataset(
                manifest, allow_technical_candidates=True
            )[0]
            self.assertAlmostEqual(fixed_sample["clip_start_s"], 0.1)

            shard = Path(temporary) / "shard.tar"
            with tarfile.open(shard, "w") as archive:
                archive.add(video, arcname="nested/sample-1.mp4")
            row["source_uri"] = f"tar://{shard}!/nested/sample-1.mp4"
            manifest.write_text(json.dumps(row) + "\n")
            tar_sample = VLAPretrainDataset(
                manifest, allow_technical_candidates=True
            )[0]
            self.assertEqual(tar_sample["frames"].shape, (2, 224, 224, 3))


if __name__ == "__main__":
    unittest.main()
