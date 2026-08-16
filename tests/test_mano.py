import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import av
from PIL import Image

from create_fixture import create_fixture
from egoqc.extract import extract_samples
from egoqc.annotated_video import render_annotated_episode
from egoqc.mano import (
    _enable_chumpy_compatibility,
    HaworManoBackend,
    ManoOverlayRenderer,
    camera_intrinsics,
    restore_left_pose,
    world_to_pixels,
    mano_skeleton_edges,
)
from egoqc.pipeline import run


class _FakeManoBackend:
    def j0_canon(self, betas, is_right):
        return np.broadcast_to([0.1, 0.0, 0.0], (len(betas), 3)).astype(np.float32)

    def forward(self, transl, orient_rotmat, hand_pose_rotmat, betas, is_right):
        # Mirror MANO's transl contract: joint 0 = transl + J0_canon.
        wrist = transl + self.j0_canon(betas, is_right)
        offsets = np.asarray(
            [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.1, 0.0]],
            dtype=np.float32,
        )
        vertices = wrist[:, None, :] + offsets[None, :, :]
        joints = wrist[:, None, :]
        return vertices, joints, np.asarray([[0, 1, 2]], dtype=np.int64)

    def provenance(self):
        return {"backend": "fake"}


class ManoTests(unittest.TestCase):
    def test_chumpy_compatibility_restores_removed_aliases(self):
        _enable_chumpy_compatibility()
        import inspect

        self.assertTrue(hasattr(inspect, "getargspec"))
        for name in ("bool", "int", "float", "complex", "object", "unicode", "str"):
            self.assertIn(name, np.__dict__)

    def test_mano21_skeleton_has_five_four_bone_chains(self):
        edges = mano_skeleton_edges(21)
        self.assertEqual(len(edges), 20)
        self.assertEqual(edges[0:4], ((0, 1), (1, 2), (2, 3), (3, 4)))
        self.assertEqual(edges[-4:], ((0, 17), (17, 18), (18, 19), (19, 20)))
        self.assertEqual(mano_skeleton_edges(7), ())

    def test_left_pose_restoration_is_involution(self):
        angle = 0.7
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.assertTrue(
            np.allclose(restore_left_pose(restore_left_pose(rotation)), rotation)
        )

    def test_fov_intrinsics_and_projection(self):
        k = camera_intrinsics(100, 100, fov=np.asarray([np.pi / 2, np.pi / 2]))
        pixels, depth, valid = world_to_pixels(
            np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, -1.0]]),
            np.eye(4),
            k,
        )
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertTrue(np.allclose(pixels[0], [50.0, 50.0]))
        self.assertEqual(depth[0], 1.0)

    def test_j0_compensation_places_wrist_at_world_translation(self):
        renderer = ManoOverlayRenderer(_FakeManoBackend())
        state = np.zeros(122, dtype=np.float32)
        record = {
            "state_mask": [True, False],
            "observation.state": state.tolist(),
            "fov": [np.pi / 2, np.pi / 2],
            "extrinsics_w2c": np.eye(4).reshape(-1).tolist(),
            "left_transl_world": [0.0, 0.0, 1.0],
            "left_orient_world": np.eye(3).reshape(-1).tolist(),
            "left_hand_pose": np.broadcast_to(np.eye(3), (15, 3, 3)).reshape(-1).tolist(),
        }
        rendered, metrics = renderer.render(Image.new("RGB", (100, 100)), record)
        self.assertEqual(metrics["hands_rendered"], 1)
        self.assertTrue(np.allclose(metrics["sides"]["left"]["wrist_pixel"], [50.0, 50.0]))
        self.assertGreater(np.asarray(rendered).sum(), 0)

    def test_hawor_backend_reports_missing_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                HaworManoBackend(Path(temporary))

    def test_extract_samples_writes_mano_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            dataset = create_fixture(temporary_path / "dataset", frames=4)
            quality = temporary_path / "quality"
            config = json.loads(
                (Path(__file__).parents[1] / "config" / "default.json").read_text()
            )
            run(dataset, quality, config)
            evidence = temporary_path / "evidence"
            summary = extract_samples(
                dataset,
                quality / "sample_plan.jsonl",
                evidence,
                mano_renderer=ManoOverlayRenderer(_FakeManoBackend()),
            )
            self.assertTrue(summary["mano_enabled"])
            self.assertGreater(summary["mano_frames_rendered"], 0)
            self.assertEqual(summary["mano_failures"], 0)
            self.assertTrue((evidence / "mano_provenance.json").exists())
            self.assertTrue(list(evidence.glob("episode-*/*-mano.jpg")))

    def test_render_annotated_episode_mp4(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset", frames=4)
            output = Path(temporary) / "episode-annotated.mp4"
            summary = render_annotated_episode(
                root,
                0,
                output,
                ManoOverlayRenderer(_FakeManoBackend()),
                batch_size=2,
                start_frame=1,
                max_frames=2,
            )
            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["render_start_frame"], 1)
            self.assertEqual(summary["render_end_frame"], 3)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".provenance.json").exists())
            with av.open(str(output)) as container:
                decoded = sum(1 for _ in container.decode(video=0))
            self.assertEqual(decoded, 2)


if __name__ == "__main__":
    unittest.main()
