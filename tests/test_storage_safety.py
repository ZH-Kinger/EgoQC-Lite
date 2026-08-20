from pathlib import Path

import pytest

from egoqc.storage_safety import (
    assert_derived_output,
    assert_raw_file_unchanged,
    raw_file_stamp,
)


def test_output_under_protected_raw_root_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    with pytest.raises(ValueError, match="protected raw root"):
        assert_derived_output(raw / "derived", protected_roots=[raw])

    output = assert_derived_output(
        tmp_path / "workspace" / "derived", protected_roots=[raw]
    )
    assert output == (tmp_path / "workspace" / "derived").resolve()


def test_raw_stamp_detects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "raw.mp4"
    source.write_bytes(b"raw")
    before = raw_file_stamp(source)

    assert_raw_file_unchanged(source, before)
    source.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="changed while being read"):
        assert_raw_file_unchanged(source, before)
