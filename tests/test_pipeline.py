import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egoqc.extract import extract_samples
from egoqc.pipeline import run
from egoqc.episode_qc import inspect_episode
from egoqc.validator import analyze_hand_visibility

from create_fixture import create_fixture


class PipelineTests(unittest.TestCase):
    def test_hand_visibility_uses_internal_gap_and_five_second_warmup(self):
        mask = np.zeros((300, 2), dtype=bool)
        mask[15:195, 0] = True
        mask[240:285, 1] = True
        metrics = analyze_hand_visibility(mask, fps=30.0, minimum_continuous_s=5.0)
        self.assertEqual(metrics["first_hand_valid_frame"], 15)
        self.assertEqual(metrics["last_hand_valid_frame"], 284)
        self.assertAlmostEqual(metrics["leading_hand_missing_s"], 0.5)
        self.assertAlmostEqual(metrics["trailing_hand_missing_s"], 0.5)
        self.assertAlmostEqual(metrics["longest_internal_hand_missing_gap_s"], 1.5)
        self.assertEqual(metrics["qualifying_hand_visible_segment_count"], 1)
        self.assertAlmostEqual(metrics["qualified_visible_duration_s"], 6.0)
        self.assertAlmostEqual(metrics["effective_video_duration_s"], 1.0)

    def test_inspect_single_episode_without_full_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset", episodes=3)
            config = json.loads(
                (Path(__file__).parents[1] / "config" / "default.json").read_text()
            )
            summary = inspect_episode(root, 1, config)
            self.assertEqual(summary["episode_index"], 1)
            self.assertEqual(summary["result"]["tier"], "gold")
            self.assertEqual(summary["result"]["length"], 12)

    def test_end_to_end_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset")
            output = Path(temporary) / "quality"
            config = json.loads(
                (Path(__file__).parents[1] / "config" / "default.json").read_text()
            )
            summary = run(root, output, config)
            self.assertEqual(summary["tier_counts"], {"gold": 1})
            self.assertEqual(summary["cache"]["parquet_misses"], 1)
            self.assertEqual(summary["cache"]["video_misses"], 1)
            self.assertIn("effective_video_hours", summary["visibility"])
            self.assertEqual(summary["visibility"]["episodes_hand_out_of_view_too_long"], 0)
            self.assertTrue((output / "report.html").exists())
            self.assertTrue((output / "live" / "current.json").exists())
            live_status = json.loads((output / "live" / "current.json").read_text())
            self.assertEqual(live_status["status"], "succeeded")
            self.assertEqual(live_status["fraction"], 1.0)
            self.assertTrue((output / "decisions" / "episode_decisions.parquet").exists())
            self.assertTrue((output / "bad_frames.jsonl").exists())
            self.assertTrue((output / "bad_frames.parquet").exists())
            self.assertEqual(summary["bad_frames"]["bad_frame_count"], 0)
            self.assertTrue((output / "decisions" / "review_manifest.jsonl").exists())
            decisions = (output / "decisions" / "episode_decisions.jsonl").read_text()
            self.assertIn('"decision": "accept"', decisions)
            report = (output / "report.html").read_text(encoding="utf-8")
            self.assertIn('id="tier-filter"', report)
            self.assertIn('id="shard-performance"', report)
            self.assertIn("有效视频", report)
            cache_files = list((output / "shard_cache").glob("*.json"))
            self.assertEqual(len(cache_files), 1)
            cache_mtime = cache_files[0].stat().st_mtime_ns
            second_summary = run(root, output, config)
            self.assertEqual(second_summary["tier_counts"], {"gold": 1})
            self.assertEqual(second_summary["cache"]["parquet_hits"], 1)
            self.assertEqual(second_summary["cache"]["video_hits"], 1)
            self.assertEqual(cache_files[0].stat().st_mtime_ns, cache_mtime)
            evidence = Path(temporary) / "evidence"
            extracted = extract_samples(root, output / "sample_plan.jsonl", evidence)
            self.assertGreater(extracted["frames_extracted"], 0)
            self.assertTrue((evidence / "episode-000000-contact-sheet.jpg").exists())
            self.assertTrue((evidence / "index.html").exists())
            self.assertTrue((evidence / "review.html").exists())
            self.assertTrue((evidence / "media-routes.json").exists())
            self.assertTrue((evidence / "episodes-vlc.xspf").exists())
            self.assertIn(
                "手部数据证据画廊",
                (evidence / "index.html").read_text(encoding="utf-8"),
            )
            review = (evidence / "review.html").read_text(encoding="utf-8")
            self.assertIn("EgoQC 人工复检", review)
            self.assertIn("human-reviews.jsonl", review)
            self.assertIn("供应商返工", review)
            self.assertIn("override_requested", review)
            self.assertIn("原始视频", review)
            self.assertIn('"evidence_frames"', review)
            self.assertIn("逐项复核", review)
            self.assertIn("issue_verdicts", review)
            self.assertIn("审核记录", review)
            self.assertIn("键盘快捷键", review)
            playlist = (evidence / "episodes-vlc.xspf").read_text(encoding="utf-8")
            self.assertIn("start-time=", playlist)
            self.assertIn("stop-time=", playlist)

    def test_position_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset")
            data_path = root / "data" / "chunk-000" / "file-000.parquet"
            table = pq.read_table(data_path)
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
            states[:, 0] += 0.2
            table = table.set_column(
                table.schema.get_field_index("observation.state"),
                "observation.state",
                pa.array(states.tolist()),
            )
            pq.write_table(table, data_path)
            output = Path(temporary) / "quality"
            config = json.loads(
                (Path(__file__).parents[1] / "config" / "default.json").read_text()
            )
            summary = run(root, output, config)
            self.assertEqual(summary["tier_counts"], {"bronze": 1})
            issues = (output / "issues.jsonl").read_text()
            self.assertIn("world_camera_position_mismatch", issues)
            rejected = (output / "decisions" / "reject_manifest.jsonl").read_text()
            self.assertIn('"decision": "reject"', rejected)

    def test_timestamp_jitter_emits_bad_frame_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset")
            data_path = root / "data" / "chunk-000" / "file-000.parquet"
            table = pq.read_table(data_path)
            timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
            timestamps[5] += 0.020
            table = table.set_column(
                table.schema.get_field_index("timestamp"),
                "timestamp",
                pa.array(timestamps),
            )
            pq.write_table(table, data_path)
            output = Path(temporary) / "quality"
            config = json.loads(
                (Path(__file__).parents[1] / "config" / "default.json").read_text()
            )
            summary = run(root, output, config)
            events = [
                json.loads(line)
                for line in (output / "bad_frames.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertGreater(summary["bad_frames"]["bad_frame_count"], 0)
            self.assertTrue(any(event["code"] == "numeric_frame_interval_jitter" for event in events))
            issues = (output / "issues.jsonl").read_text()
            self.assertIn("bad_frame_ratio_exceeded", issues)

    def test_multi_episode_shard_is_grouped_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = create_fixture(Path(temporary) / "dataset", episodes=8)
            output = Path(temporary) / "quality"
            config = json.loads(
                (Path(__file__).parents[1] / "config" / "default.json").read_text()
            )
            summary = run(root, output, config)
            self.assertEqual(summary["episode_count"], 8)
            self.assertEqual(summary["tier_counts"], {"gold": 8})
            self.assertEqual(summary["cache"]["parquet_misses"], 1)


if __name__ == "__main__":
    unittest.main()
