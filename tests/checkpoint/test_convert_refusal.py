"""Protect existing FTW output before any conversion or device setup."""

import pytest

from freetoken.checkpoint import convert
from freetoken.checkpoint.ftw import INDEX_NAME


def _snapshot(directory):
    return {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.iterdir()
        if path.is_file()
    }


@pytest.mark.parametrize("dirname", ["output", "output[1]"])
@pytest.mark.parametrize("finished", [True, False])
def test_occupied_output_is_refused_before_device_setup(tmp_path, monkeypatch, dirname, finished):
    from freetoken import distributed

    def unexpected_setup():
        pytest.fail("conversion reached TP/device setup with occupied output")

    monkeypatch.setattr(distributed, "try_get_tp_info", unexpected_setup)
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / dirname
    output.mkdir()
    (output / "freetoken-00001.ftw").write_bytes(b"second shard")
    (output / "freetoken-00000.ftw").write_bytes(b"first shard")
    if finished:
        (output / INDEX_NAME).write_text('{"shards": []}', encoding="utf-8")
    before = _snapshot(output)

    with pytest.raises(SystemExit, match="already holds an FTW checkpoint") as exc:
        convert.convert_checkpoint(str(source), str(output))

    assert str(output) in str(exc.value)
    assert (INDEX_NAME if finished else "freetoken-00000.ftw") in str(exc.value)
    assert _snapshot(output) == before


def test_absent_empty_or_metadata_only_output_is_allowed(tmp_path):
    absent = tmp_path / "absent"
    convert._refuse_occupied_output(str(absent))
    assert not absent.exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    convert._refuse_occupied_output(str(empty))
    assert list(empty.iterdir()) == []

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text("{}", encoding="utf-8")
    (metadata / "notes.ftw.bak").write_bytes(b"unrelated")
    before = _snapshot(metadata)
    convert._refuse_occupied_output(str(metadata))
    assert _snapshot(metadata) == before
