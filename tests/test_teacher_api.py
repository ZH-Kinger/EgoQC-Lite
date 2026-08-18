from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from create_fixture import create_fixture
from egoqc.teacher_api import (
    build_chat_payload,
    extract_clip_frames,
    run_teacher_api,
    validate_teacher_label,
)


def _queue_row(video: Path, output_path: Path):
    return {
        "request_id": "clip-001",
        "prompt_version": "test-v1",
        "source_uri": str(video),
        "clip_start_s": 0.0,
        "clip_end_s": 4.0,
        "candidate_tasks": ["hand_absent"],
        "trigger_tasks": [],
        "event_codes": [],
        "assessment_dimensions": {"open_world_findings": "发现其他异常"},
        "output_path": str(output_path),
        "required_response": {
            "schema_version": "egoqc-visual-teacher-v1",
            "overall": {},
            "tasks": {
                "hand_absent": {
                    "probability": "float[0,1]",
                    "confidence": "float[0,1]",
                }
            },
            "findings": [],
        },
    }


class _FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.value).encode()


def _fake_urlopen(captured):
    def invoke(request, timeout):
        payload = json.loads(request.data)
        captured["authorization"] = request.headers.get("Authorization")
        captured["timeout"] = timeout
        content = payload["messages"][1]["content"]
        captured["image_count"] = sum(
            item.get("type") == "image_url" for item in content
        )
        label = {
            "schema_version": "egoqc-visual-teacher-v1",
            "overall": {
                "training_usable": True,
                "recommended_route": "accept",
                "confidence": 0.91,
                "allowed_uses": ["video_representation"],
            },
            "tasks": {
                "hand_absent": {"probability": 0.05, "confidence": 0.9}
            },
            "findings": [],
            "missing_annotations": [],
            "summary": "双手可见",
        }
        response = {
            "id": "response-1",
            "choices": [{"message": {"content": json.dumps(label)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
        return _FakeResponse(response)

    return invoke


class TeacherApiTests(unittest.TestCase):
    def test_bailian_preset_uses_video_frame_array_and_dashscope_key(self):
        request = _queue_row(Path("/video.mp4"), Path("/teacher-label.json"))
        request["task_context"] = {"task": "add/remove lids"}
        request["capability_context"] = {"mano_parameters": False}
        request["visual_evidence"] = "annotation_overlay"
        frames = [
            {"relative_time_s": 0.0, "data_url": "data:image/jpeg;base64,AA=="},
            {"relative_time_s": 0.5, "data_url": "data:image/jpeg;base64,AQ=="},
        ]
        payload = build_chat_payload(
            request,
            "qwen3-vl-plus",
            frames,
            media_mode="bailian_video_frames",
            sample_fps=2,
        )
        media = payload["messages"][1]["content"][-1]
        prompt = payload["messages"][1]["content"][0]["text"]
        self.assertEqual(media["type"], "video")
        self.assertEqual(media["video"], [frame["data_url"] for frame in frames])
        self.assertEqual(media["fps"], 2)
        self.assertIn("add/remove lids", prompt)
        self.assertIn("annotation_overlay", prompt)

    def test_low_cost_profile_reduces_visual_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "dataset", frames=150)
            video = dataset / "videos/observation.images.ego/chunk-000/file-000.mp4"
            queue = root / "queue.jsonl"
            queue.write_text(
                json.dumps(_queue_row(video, root / "labels/teacher-label.json")) + "\n"
            )
            summary = run_teacher_api(
                queue,
                root / "dry-run",
                provider="bailian",
                region="beijing",
                base_url=None,
                model=None,
                dry_run=True,
            )
            self.assertEqual(summary["cost_profile"], "low")
            self.assertEqual(summary["max_frames"], 12)
            self.assertEqual(summary["max_edge"], 448)
            self.assertEqual(summary["jpeg_quality"], 72)

    def test_rejects_unknown_task_and_out_of_range_finding(self):
        request = _queue_row(Path("/video.mp4"), Path("/teacher-label.json"))
        label = {
            "schema_version": "egoqc-visual-teacher-v1",
            "overall": {
                "training_usable": False,
                "recommended_route": "human_review",
                "confidence": 0.5,
                "allowed_uses": [],
            },
            "tasks": {
                "hand_absent": {"probability": 0.5, "confidence": 0.5},
                "invented_task": {"probability": 1.0, "confidence": 1.0},
            },
            "findings": [{
                "category": "unknown",
                "severity": "error",
                "start_s": 0,
                "end_s": 8,
            }],
            "missing_annotations": [],
        }
        with self.assertRaisesRegex(ValueError, "未知 tasks"):
            validate_teacher_label(label, request)
        label["tasks"].pop("invented_task")
        with self.assertRaisesRegex(ValueError, "时间范围越界"):
            validate_teacher_label(label, request)

    def test_normalizes_finding_time_and_preserves_original_range(self):
        from egoqc.teacher_api import normalize_teacher_label

        request = _queue_row(Path("/video.mp4"), Path("/teacher-label.json"))
        label = {
            "findings": [{
                "category": "blur",
                "severity": "warning",
                "start_s": -0.2,
                "end_s": 8.0,
            }]
        }

        normalize_teacher_label(label, request)

        finding = label["findings"][0]
        self.assertEqual(finding["start_s"], 0.0)
        self.assertEqual(
            finding["end_s"],
            request["clip_end_s"] - request["clip_start_s"],
        )
        self.assertEqual(
            finding["time_normalization"]["reason"],
            "clamped_to_reviewed_clip",
        )

    def test_extracts_bounded_ordered_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "dataset", frames=150)
            video = dataset / "videos/observation.images.ego/chunk-000/file-000.mp4"
            frames, metadata = extract_clip_frames(
                str(video), 0.5, 4.5, sample_fps=2, max_frames=6, max_edge=96
            )
            self.assertEqual(len(frames), 6)
            self.assertEqual(metadata["frame_count"], 6)
            self.assertTrue(all(max(size) <= 96 for size in metadata["encoded_sizes"]))
            self.assertEqual(
                [frame["time_s"] for frame in frames],
                sorted(frame["time_s"] for frame in frames),
            )
            self.assertTrue(all(frame["data_url"].startswith("data:image/jpeg;base64,") for frame in frames))

    def test_dry_run_builds_payload_without_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "dataset", frames=150)
            video = dataset / "videos/observation.images.ego/chunk-000/file-000.mp4"
            queue = root / "queue.jsonl"
            queue.write_text(
                json.dumps(_queue_row(video, root / "labels/teacher-label.json")) + "\n"
            )
            summary = run_teacher_api(
                queue,
                root / "dry-run",
                base_url=None,
                model=None,
                dry_run=True,
                max_frames=4,
            )
            self.assertEqual(summary["status_counts"]["dry_run"], 1)
            self.assertFalse(summary["credentials_stored"])
            self.assertFalse((root / "labels/teacher-label.json").exists())

    def test_openai_compatible_request_is_cached_after_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = create_fixture(root / "dataset", frames=150)
            video = dataset / "videos/observation.images.ego/chunk-000/file-000.mp4"
            label_path = root / "labels/teacher-label.json"
            queue = root / "queue.jsonl"
            queue.write_text(json.dumps(_queue_row(video, label_path)) + "\n")
            captured = {}
            with patch("egoqc.teacher_api.urllib.request.urlopen", _fake_urlopen(captured)):
                with patch.dict(os.environ, {"TEST_TEACHER_KEY": "secret-value"}):
                    summary = run_teacher_api(
                        queue,
                        root / "run-1",
                        base_url="https://teacher.example/v1",
                        model="test-model",
                        api_key_env="TEST_TEACHER_KEY",
                        max_frames=4,
                        max_retries=0,
                    )
                    cached = run_teacher_api(
                        queue,
                        root / "run-2",
                        base_url="https://teacher.example/v1",
                        model="test-model",
                        api_key_env="TEST_TEACHER_KEY",
                        max_frames=4,
                        max_retries=0,
                    )
            self.assertEqual(summary["status_counts"]["succeeded"], 1)
            self.assertEqual(cached["status_counts"]["cached"], 1)
            self.assertEqual(captured["authorization"], "Bearer secret-value")
            self.assertEqual(captured["image_count"], 4)
            label = json.loads(label_path.read_text())
            self.assertEqual(label["teacher_model"], "test-model")
            self.assertNotIn("secret-value", label_path.read_text())
            self.assertEqual(summary["usage"]["input_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
