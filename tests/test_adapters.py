import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egoqc.adapters import (
    EgoDexHDF5Adapter,
    ManoHamerAdapter,
    RekaDailyRawAdapter,
    detect_adapter,
)
from egoqc.canonical import CanonicalEpisode, CapabilityManifest, HandTrack, VideoReference


class AdapterTests(unittest.TestCase):
    def test_canonical_episode_rejects_frame_count_mismatch(self):
        episode = CanonicalEpisode(
            episode_id="sample/0",
            source_format="test",
            timestamps=np.arange(3, dtype=np.float64) / 30.0,
            video=VideoReference(Path("0.mp4"), 30.0, 2, 32, 24, "h264"),
            capabilities=CapabilityManifest(),
        )
        with self.assertRaisesRegex(ValueError, "timestamp count"):
            episode.validate()

    def test_detects_egodex_collection_without_recursive_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "part1").mkdir()
            self.assertEqual(detect_adapter(root), "egodex_collection")

    def test_rekadaily_index_is_metadata_only_capability_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata").mkdir()
            table = pa.Table.from_pylist([
                {
                    "video_id": "a", "project": "household", "category": None,
                    "subcategory": None, "flow": "Kitchen", "activities": ["wash"],
                    "lighting": "Artificial", "duration_s": 10.0, "fps": 30.0,
                    "width": 1280, "height": 720, "num_frames": 300, "codec": "h264",
                    "file_size_bytes": 1000, "src_ext": "mp4", "collector": "x",
                },
                {
                    "video_id": "b", "project": "phone", "category": "daily",
                    "subcategory": "clean", "flow": None, "activities": None,
                    "lighting": None, "duration_s": 12.0, "fps": 24.0,
                    "width": 848, "height": 480, "num_frames": 288, "codec": "h264",
                    "file_size_bytes": 2000, "src_ext": "mov", "collector": "y",
                },
            ])
            pq.write_table(table, root / "metadata" / "index.parquet")
            self.assertEqual(detect_adapter(root), "rekadaily_raw")
            summary = RekaDailyRawAdapter().summarize_index(root)
            self.assertEqual(summary["videos"], 2)
            self.assertEqual(summary["screening_counts"]["fps_below_29_9"], 1)
            self.assertEqual(summary["screening_counts"]["short_edge_below_720"], 1)
            self.assertEqual(
                summary["screening_counts"]["metadata_candidates_for_expensive_stage"], 1
            )
            self.assertTrue(summary["capabilities"]["video"])
            self.assertFalse(summary["capabilities"]["mano_parameters"])
            self.assertIn("MPJPE", summary["unavailable_acceptance_metrics"])

    def test_rekadaily_locates_video_inside_tar_without_extracting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_root = root / "data" / "household"
            shard_root.mkdir(parents=True)
            payload = b"video-placeholder"
            with tarfile.open(shard_root / "shard-00000.tar", "w") as archive:
                member = tarfile.TarInfo("nested/a.mp4")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            located = RekaDailyRawAdapter._tar_video(
                root,
                {"video_id": "a", "project": "household", "src_ext": "mp4"},
            )
            self.assertIsNotNone(located)
            self.assertEqual(located[1], "nested/a.mp4")

    def test_mano_hamer_row_becomes_standard_view(self):
        state = np.zeros(61)
        state[48:51] = [1.0, 2.0, 3.0]
        state[51:61] = np.arange(10)
        row = {
            "timestamp": 0.0,
            "frame_index": 0,
            "episode_index": 2,
            "index": 9,
            "task_index": 3,
            "camera.intrinsic": [100.0, 0.0, 80.0, 0.0, 100.0, 45.0, 0.0, 0.0, 1.0],
            "camera.extrinsic.T_world_camera": np.eye(4).reshape(-1).tolist(),
        }
        for side, present in (("left", 1), ("right", 0)):
            prefix = f"observation.mano.{side}"
            row[f"{prefix}.present"] = present
            row[f"{prefix}.state"] = state.tolist()
            row[f"{prefix}.global_orient"] = [0.0, 0.0, 0.0]
            row[f"{prefix}.body_pose"] = [0.0] * 45
            row[f"{prefix}.cam_trans"] = [1.0, 2.0, 3.0]
            row[f"{prefix}.betas"] = list(range(10))
        info = {"features": {"front_camera": {"shape": [90, 160, 3]}}}
        record = ManoHamerAdapter().normalize_rows([row], info)[0].to_dict()
        self.assertEqual(len(record["observation.state"]), 122)
        self.assertEqual(record["state_mask"], [True, False])
        self.assertEqual(record["main_type"], 0)
        np.testing.assert_allclose(record["left_transl_world"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(record["observation.state"][0:3], [1.0, 2.0, 3.0])
        self.assertEqual(record["observation.state"][51:61], list(range(10)))

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py optional dependency")
    def test_egodex_pair_becomes_canonical_episode(self):
        import av
        import h5py

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = root / "part1" / "pick_place"
            task.mkdir(parents=True)
            hdf5_path = task / "0.hdf5"
            frames = 4
            identity = np.broadcast_to(np.eye(4, dtype=np.float32), (frames, 4, 4))
            with h5py.File(hdf5_path, "w") as handle:
                handle.attrs["task"] = "Pick Place"
                handle.attrs["llm_description"] = "Pick up the block and place it down."
                handle.create_dataset("camera/intrinsic", data=np.eye(3, dtype=np.float32))
                handle.create_dataset("transforms/camera", data=identity)
                for side in ("left", "right"):
                    names = [f"{side}Hand"]
                    names.extend(
                        f"{side}Thumb{segment}"
                        for segment in (
                            "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"
                        )
                    )
                    names.extend(
                        f"{side}{finger}Finger{segment}"
                        for finger in ("Index", "Middle", "Ring", "Little")
                        for segment in (
                            "Metacarpal", "Knuckle", "IntermediateBase",
                            "IntermediateTip", "Tip",
                        )
                    )
                    for name in names:
                        handle.create_dataset(f"transforms/{name}", data=identity)
                        handle.create_dataset(
                            f"confidences/{name}", data=np.ones(frames)
                        )
            video_path = task / "0.mp4"
            with av.open(str(video_path), "w") as container:
                stream = container.add_stream("mpeg4", rate=30)
                stream.width = 32
                stream.height = 24
                stream.pix_fmt = "yuv420p"
                for _ in range(frames):
                    frame = av.VideoFrame.from_ndarray(
                        np.zeros((24, 32, 3), dtype=np.uint8), format="rgb24"
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

            canonical = EgoDexHDF5Adapter().load_episode(
                root, "part1/pick_place/0", confidence_threshold=0.5
            )
            self.assertEqual(canonical.episode_id, "part1/pick_place/0")
            self.assertEqual(canonical.frame_count, frames)
            self.assertEqual(canonical.video.frame_count, frames)
            self.assertEqual(canonical.labels["task"], "Pick Place")
            self.assertEqual(canonical.hands["left"].joint_names[0], "leftHand")
            self.assertEqual(len(canonical.hands["left"].joint_names), 25)
            self.assertIn("leftThumbTip", canonical.hands["left"].joint_names)
            self.assertTrue(canonical.capabilities.hand_joint_transforms)
            self.assertFalse(canonical.capabilities.mano_parameters)


if __name__ == "__main__":
    unittest.main()
