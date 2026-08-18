import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from egoqc.canonical import CanonicalEpisode, CapabilityManifest, HandTrack, VideoReference
from egoqc.egodex_batch import build_egodex_training_candidates, select_egodex_episodes


def _canonical(episode_id: str, score: float, *, width: int = 1280) -> CanonicalEpisode:
    frames = 180
    confidence = np.full((frames, 1), score, dtype=np.float64)
    transform = np.broadcast_to(np.eye(4), (frames, 1, 4, 4)).copy()
    hands = {
        side: HandTrack(
            side=side,
            joint_names=[f"{side}Hand"],
            transforms=transform.copy(),
            confidences=confidence.copy(),
            valid=np.ones(frames, dtype=bool),
            local_origin="wrist",
            source_model="test",
            confidence_threshold=0.5,
        )
        for side in ("left", "right")
    }
    return CanonicalEpisode(
        episode_id=episode_id,
        source_format="egodex_hdf5",
        timestamps=np.arange(frames, dtype=np.float64) / 30.0,
        video=VideoReference(Path(f"{episode_id}.mp4"), 30.0, frames, width, 720, "h264"),
        capabilities=CapabilityManifest(video=True, prediction_confidence=True),
        hands=hands,
        provenance={"readonly": True},
    )


class EgoDexBatchTests(unittest.TestCase):
    def test_selection_is_task_balanced_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in ("pick", "place"):
                task_root = root / "part1" / task
                task_root.mkdir(parents=True)
                for index in range(4):
                    (task_root / f"{index}.hdf5").touch()
            first = select_egodex_episodes(root, episodes_per_task=2, seed=9)
            second = select_egodex_episodes(root, episodes_per_task=2, seed=9)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)
            self.assertEqual({row["task"] for row in first}, {"pick", "place"})

    def test_batch_keeps_failures_and_marks_candidates_as_weak(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            output = Path(temporary) / "output"
            task = root / "part1" / "pick"
            task.mkdir(parents=True)
            for name in ("good", "low", "bad-resolution", "broken"):
                (task / f"{name}.hdf5").touch()

            def load(_adapter, _dataset, episode, confidence_threshold=0.5):
                if episode.endswith("broken"):
                    raise ValueError("broken pair")
                if episode.endswith("bad-resolution"):
                    return _canonical(episode, 0.9, width=640)
                return _canonical(episode, 0.95 if episode.endswith("good") else 0.55)

            with patch("egoqc.egodex_batch.EgoDexHDF5Adapter.load_episode", new=load):
                summary = build_egodex_training_candidates(
                    root,
                    output,
                    episodes_per_task=4,
                    workers=2,
                    clean_quantile=0.67,
                    hard_negative_quantile=0.2,
                )
            self.assertEqual(summary["selection"]["selected"], 4)
            self.assertEqual(summary["profiled"], 3)
            self.assertEqual(summary["errors"], 1)
            self.assertEqual(sum(summary["tier_counts"].values()), 4)
            self.assertTrue(summary["label_policy"]["validation_and_test_require_human_gold"])
            content = (output / "candidate-clean.jsonl").read_text(encoding="utf-8")
            self.assertIn('"label_status": "weak_candidate_not_gold"', content)
            self.assertIn("good", content)
            self.assertIn("broken pair", (output / "errors.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
