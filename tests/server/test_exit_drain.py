"""Workers drain queued GPU work before dying of an exception (Windows/ROCm exit hang)."""

import sys
from types import SimpleNamespace

import pytest

from freetoken.server import launch


def _fake_torch(monkeypatch, *, available=True, initialized=True, sync_error=None):
    calls = []

    def synchronize():
        calls.append("sync")
        if sync_error is not None:
            raise sync_error

    cuda = SimpleNamespace(
        is_available=lambda: available,
        is_initialized=lambda: initialized,
        synchronize=synchronize,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    return calls


def test_drain_synchronizes_an_initialized_device(monkeypatch):
    calls = _fake_torch(monkeypatch)
    launch._drain_device_before_exit()
    assert calls == ["sync"]


@pytest.mark.parametrize(("available", "initialized"), [(False, False), (True, False)])
def test_drain_never_initializes_cuda(monkeypatch, available, initialized):
    calls = _fake_torch(monkeypatch, available=available, initialized=initialized)
    launch._drain_device_before_exit()
    assert calls == []


def test_drain_failure_is_swallowed(monkeypatch):
    calls = _fake_torch(monkeypatch, sync_error=RuntimeError("device lost"))
    launch._drain_device_before_exit()  # must not raise over the original exception
    assert calls == ["sync"]
