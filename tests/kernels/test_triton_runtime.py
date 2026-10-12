import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def runtime_case(monkeypatch):
    from freetoken import _triton_runtime as runtime
    import triton.knobs as knobs

    runtime._initialize_amd_buffer_ops.cache_clear()
    monkeypatch.setenv('FREETOKEN_TRITON_CACHE_AMD_KNOB', '1')
    calls = {'imports': [], 'get': 0, 'set': [], 'effective': True}

    def read(key, default):
        calls['get'] += 1
        assert key == 'AMDGCN_USE_BUFFER_OPS'
        return calls['effective']

    monkeypatch.setattr(knobs, 'getenv_bool', read)
    monkeypatch.setattr(knobs, 'setenv', lambda key, value: calls['set'].append((key, value)))

    class Amd:
        use_buffer_ops = knobs.env_bool('AMDGCN_USE_BUFFER_OPS', True)

    amd = Amd()
    triton = SimpleNamespace(__version__='3.8.0', knobs=SimpleNamespace(amd=amd, env_bool=knobs.env_bool))
    torch = SimpleNamespace(version=SimpleNamespace(hip='7.15.0'))
    original_import = importlib.import_module

    def load(name, *args, **kwargs):
        if name in ('torch', 'triton'):
            calls['imports'].append(name)
            return torch if name == 'torch' else triton
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(runtime.importlib, 'import_module', load)
    yield SimpleNamespace(runtime=runtime, torch=torch, triton=triton, amd=amd, calls=calls, knobs=knobs)
    runtime._initialize_amd_buffer_ops.cache_clear()


@pytest.mark.parametrize('effective', [True, False])
def test_caches_effective_boolean_once_with_torch_first(runtime_case, effective):
    case = runtime_case
    case.calls['effective'] = effective
    result = case.runtime.initialize_amd_buffer_ops()
    assert result.status == 'cached'
    assert result.value is effective
    assert case.calls['imports'] == ['torch', 'triton']
    assert case.calls['get'] == 1
    assert len(case.calls['set']) == 1
    for _ in range(100):
        assert case.amd.use_buffer_ops is effective
        assert case.runtime.initialize_amd_buffer_ops() is result
    assert case.calls['get'] == 1
    assert len(case.calls['set']) == 1


@pytest.mark.parametrize('flag', [None, '', '0', 'false', 'off'])
def test_disabled_does_not_import_runtime(runtime_case, monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv('FREETOKEN_TRITON_CACHE_AMD_KNOB')
    else:
        monkeypatch.setenv('FREETOKEN_TRITON_CACHE_AMD_KNOB', flag)
    case = runtime_case
    assert case.runtime.initialize_amd_buffer_ops().status == 'disabled'
    assert case.calls['imports'] == []
    assert case.calls['get'] == 0
    assert case.calls['set'] == []


def test_cuda_runtime_does_not_import_triton(runtime_case):
    case = runtime_case
    case.torch.version.hip = None
    assert case.runtime.initialize_amd_buffer_ops().status == 'not-rocm'
    assert case.calls['imports'] == ['torch']
    assert case.calls['get'] == 0
    assert case.calls['set'] == []


@pytest.mark.parametrize('version', ['3.7.1', '3.8.1', '3.9.0', '3.8.0a0'])
def test_unqualified_version_skips_without_reading_knob(runtime_case, version):
    case = runtime_case
    case.triton.__version__ = version
    assert case.runtime.initialize_amd_buffer_ops().status == 'unsupported-version'
    assert case.calls['get'] == 0
    assert case.calls['set'] == []


@pytest.mark.parametrize('kind', ['missing', 'property', 'wrong-key', 'subclass'])
def test_unsupported_descriptor_skips_without_reading_knob(runtime_case, kind):
    case = runtime_case
    if kind == 'missing':
        del type(case.amd).use_buffer_ops
    elif kind == 'property':
        type(case.amd).use_buffer_ops = property(lambda self: pytest.fail('unknown descriptor was read'))
    elif kind == 'wrong-key':
        type(case.amd).__dict__['use_buffer_ops'].key = 'UNKNOWN_ENV_KEY'
    else:
        class ChangedBool(case.knobs.env_bool):
            pass
        descriptor = ChangedBool('AMDGCN_USE_BUFFER_OPS', True)
        descriptor.__set_name__(type(case.amd), 'use_buffer_ops')
        type(case.amd).use_buffer_ops = descriptor
    assert case.runtime.initialize_amd_buffer_ops().status == 'unsupported-knob'
    assert case.calls['get'] == 0
    assert case.calls['set'] == []


@pytest.mark.parametrize('value', [True, False])
def test_preserves_python_override_and_later_assignment(runtime_case, value):
    case = runtime_case
    case.amd.use_buffer_ops = value
    case.calls['set'].clear()
    case.calls['effective'] = not value
    result = case.runtime.initialize_amd_buffer_ops()
    assert result.status == 'already-cached'
    assert result.value is value
    assert case.calls['get'] == 0
    assert case.calls['set'] == []
    case.amd.use_buffer_ops = not value
    case.runtime.initialize_amd_buffer_ops()
    assert case.amd.use_buffer_ops is not value


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_non_boolean_value_is_not_pinned(runtime_case, value):
    case = runtime_case
    case.calls['effective'] = value
    assert case.runtime.initialize_amd_buffer_ops().status == 'invalid-value'
    assert case.amd.__dict__ == {}
    assert case.calls['set'] == []


@pytest.mark.parametrize('flag,value', [('0', '0'), ('1', '0'), ('1', '1')])
def test_kernel_import_and_fresh_process_rollback(flag, value):
    source = r'''
import json
import torch
import triton
descriptor = type(triton.knobs.amd).__dict__['use_buffer_ops']
original = descriptor.get
calls = 0
def read():
    global calls
    calls += 1
    return original()
descriptor.get = read
import freetoken.kernel
from freetoken._triton_runtime import initialize_amd_buffer_ops
result = initialize_amd_buffer_ops()
for _ in range(1000):
    assert triton.knobs.amd.use_buffer_ops is EXPECTED
assert not torch.cuda.is_initialized()
print(json.dumps({'state': result._asdict(), 'getter_calls': calls,
                  'shadowed': 'use_buffer_ops' in triton.knobs.amd.__dict__}))
'''
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root / 'python'), PYTHONNOUSERSITE='1',
               CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1',
               FREETOKEN_TRITON_CACHE_AMD_KNOB=flag, AMDGCN_USE_BUFFER_OPS=value)
    proc = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c',
                           'EXPECTED = ' + str(value == '1') + '\n' + source],
                          env=env, capture_output=True, text=True, timeout=45)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data['shadowed'] is (flag == '1')
    assert data['getter_calls'] == (1 if flag == '1' else 1000)


def test_optin_keeps_cli_and_daemon_imports_torch_free():
    source = r'''
import sys
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'triton'):
            raise AssertionError('runtime imported before kernels: ' + name)
sys.meta_path.insert(0, Blocker())
import freetoken.cli
import freetoken.daemon
assert 'torch' not in sys.modules and 'triton' not in sys.modules
'''
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root / 'python'), FREETOKEN_TRITON_CACHE_AMD_KNOB='1')
    proc = subprocess.run([sys.executable, '-B', '-c', source], env=env,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
