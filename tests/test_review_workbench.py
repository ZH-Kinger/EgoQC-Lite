from pathlib import Path

from egoqc.review_workbench import _annotated_uri


def test_review_workbench_discovers_original_annotation_video(tmp_path: Path) -> None:
    video = tmp_path / "episode-000012-annotated.mp4"
    video.write_bytes(b"derived")
    assert _annotated_uri(tmp_path, 12) == video.resolve().as_uri()
