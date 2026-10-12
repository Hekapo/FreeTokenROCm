from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from freetoken.server.args import _download_modelscope


@pytest.mark.parametrize("directory", [False, True])
def test_modelscope_local_path_needs_no_sdk(tmp_path, monkeypatch, directory):
    path = tmp_path / ("model-dir" if directory else "model.gguf")
    if directory:
        path.mkdir()
    else:
        path.write_bytes(b"GGUF")
    monkeypatch.setitem(sys.modules, "modelscope", None)
    assert _download_modelscope(str(path), False) == str(path)


def test_modelscope_missing_extra_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "modelscope", None)
    with pytest.raises(ValueError, match=r"freetoken\[modelscope\]"):
        _download_modelscope("owner/model", False)


@pytest.mark.parametrize("dummy", [False, True])
def test_modelscope_remote_download_keeps_weight_filter(monkeypatch, dummy):
    calls = []

    def download(model, **kwargs):
        calls.append((model, kwargs))
        return "downloaded-model"

    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=download))
    assert _download_modelscope("owner/model", dummy) == "downloaded-model"
    patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"] if dummy else []
    assert calls == [("owner/model", {"ignore_patterns": patterns})]


def test_modelscope_broken_transitive_dependency_is_not_missing_extra(monkeypatch):
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "modelscope":
            raise ModuleNotFoundError("No module named requests", name="requests")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(ModuleNotFoundError) as raised:
        _download_modelscope("owner/model", False)
    assert raised.value.name == "requests"
