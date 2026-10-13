import importlib.util
from importlib import metadata
from pathlib import Path
import sys

import pytest


@pytest.fixture
def safety():
    path = Path(__file__).resolve().parents[2] / 'scripts/windows-rocm1010/compiler_safety.py'
    spec = importlib.util.spec_from_file_location('rocm1010_compiler_safety_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


SOURCE = '''@staticmethod
def make_ttgir(mod, metadata, options):
    pm = []
    before(pm)
    amd.passes.ttgpuir.add_fold_true_cmpi(pm)
    after(pm)
    return mod
'''


@pytest.fixture
def candidate_runtime():
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 12):
        pytest.skip('needs the isolated Windows / Python 3.12 candidate')
    import torch
    import triton

    if (torch.__version__ != '2.14.0+rocm10.1.0'
            or metadata.version('triton-windows') != '3.8.0.post29'):
        pytest.skip('needs the pinned ROCm 10.1 / Triton post29 candidate')
    assert not torch.cuda.is_initialized()
    return torch, triton


def test_keeps_surrounding_passes_and_input(safety):
    calls = []
    function = safety.without_range_fold(
        SOURCE, {'before': lambda pm: calls.append('before'),
                 'after': lambda pm: calls.append('after')},
    )
    value = object()
    assert function(value, {}, None) is value
    assert calls == ['before', 'after']


@pytest.mark.parametrize('source', [
    SOURCE.replace('    amd.passes.ttgpuir.add_fold_true_cmpi(pm)\n', ''),
    SOURCE.replace('    amd.passes.ttgpuir.add_fold_true_cmpi(pm)',
                   '    amd.passes.ttgpuir.add_fold_true_cmpi(pm)\n'
                   '    amd.passes.ttgpuir.add_fold_true_cmpi(pm)'),
    SOURCE.replace('add_fold_true_cmpi(pm)', 'add_fold_true_cmpi(mod)'),
    SOURCE.replace('    amd.passes.ttgpuir.add_fold_true_cmpi(pm)',
                   '    if options:\n        amd.passes.ttgpuir.add_fold_true_cmpi(pm)'),
    SOURCE.replace('@staticmethod', '@unknown_decorator'),
])
def test_rejects_unreviewed_pipeline_shapes(safety, source):
    with pytest.raises(RuntimeError, match='pipeline'):
        safety.without_range_fold(source, {})


def test_import_does_not_import_or_activate_gpu_libraries(monkeypatch):
    import importlib

    original = importlib.import_module

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'triton'}:
            pytest.fail('GPU library imported without an explicit install request')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, 'import_module', guarded)
    path = Path(__file__).resolve().parents[2] / 'scripts/windows-rocm1010/compiler_safety.py'
    spec = importlib.util.spec_from_file_location('compiler_safety_import_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_hook_uses_a_separate_compilation_cache_and_restores(safety, candidate_runtime):
    torch, triton = candidate_runtime
    original = triton.knobs.runtime.add_stages_inspection_hook
    assert original is None
    try:
        state = safety.install_range_safety()
        hook = triton.knobs.runtime.add_stages_inspection_hook
        key, digest = hook()
        assert key.startswith('freetoken-rocm1010-range-safety:')
        assert state['cache_key'] == key
        assert state['hook_sha256'] == digest
        assert len(digest) == 64
        assert safety.install_range_safety() == state
        assert not torch.cuda.is_initialized()
    finally:
        triton.knobs.runtime.add_stages_inspection_hook = original


def test_does_not_replace_an_existing_compiler_hook(safety, candidate_runtime):
    _, triton = candidate_runtime
    original = triton.knobs.runtime.add_stages_inspection_hook
    existing = lambda: ('other-tool', 'other-hash')
    triton.knobs.runtime.add_stages_inspection_hook = existing
    try:
        with pytest.raises(RuntimeError, match='existing'):
            safety.install_range_safety()
        assert triton.knobs.runtime.add_stages_inspection_hook is existing
    finally:
        triton.knobs.runtime.add_stages_inspection_hook = original


def test_rejects_a_different_torch_stack(safety, candidate_runtime, monkeypatch):
    torch, triton = candidate_runtime
    monkeypatch.setattr(torch, '__version__', '2.13.0+rocm10.0.0')
    with pytest.raises(RuntimeError, match='stack'):
        safety.install_range_safety()
    assert triton.knobs.runtime.add_stages_inspection_hook is None


def test_rejects_a_different_compiler_file(safety, candidate_runtime, monkeypatch):
    _, triton = candidate_runtime
    monkeypatch.setattr(safety, 'COMPILER_SHA256', '0' * 64)
    with pytest.raises(RuntimeError, match='compiler file'):
        safety.install_range_safety()
    assert triton.knobs.runtime.add_stages_inspection_hook is None


def test_rejects_installation_after_gpu_initialization(safety, candidate_runtime, monkeypatch):
    torch, triton = candidate_runtime
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(RuntimeError, match='before GPU'):
        safety.install_range_safety()
    assert triton.knobs.runtime.add_stages_inspection_hook is None
