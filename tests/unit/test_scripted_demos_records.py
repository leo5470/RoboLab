"""Shared file helpers (robolab.core.scripted_demos.records)."""

import json

import pytest

from robolab.core.scripted_demos import records


def test_read_jsonl_raises_on_a_bad_row_unless_asked_to_skip(tmp_path):
    path = tmp_path / "a.jsonl"
    assert records.read_jsonl(path) == []
    path.write_text('{"a": 1}\n\n{"b": 2}\n{"c":')  # a row still being written
    with pytest.raises(json.JSONDecodeError):  # a resuming writer must not append after it
        records.read_jsonl(path)
    assert records.read_jsonl(path, skip_bad=True) == [{"a": 1}, {"b": 2}]
    path.write_text('{"a": 1}\n{bad}\n{"c": 3}\n')
    assert records.read_jsonl(path, skip_bad=True) == [{"a": 1}, {"c": 3}]


def test_read_json_default_and_parse_targets(tmp_path):
    assert records.read_json(tmp_path / "missing.json", {"x": 0}) == {"x": 0}
    (tmp_path / "half.json").write_text('{"a": ')
    assert records.read_json(tmp_path / "half.json", None) is None
    assert records.parse_targets("1:5 09:3") == {"01": 5, "09": 3}


def test_file_sha256(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert records.file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
