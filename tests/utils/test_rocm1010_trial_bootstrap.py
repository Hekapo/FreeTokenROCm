from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def trial_environment(tmp_path):
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 12):
        pytest.skip('requires the isolated Windows candidate')
    import torch
    if torch.__version__ != '2.14.0+rocm10.1.0' or metadata.version('triton-windows') != '3.8.0.post29':
        pytest.skip('requires the pinned new stack')
    root = Path(__file__).resolve().parents[2]
    scripts = root / 'scripts/windows-rocm1010'
    shutil.copyfile(scripts / 'trial_sitecustomize.py', tmp_path / 'sitecustomize.py')
    shutil.copyfile(scripts / 'compiler_safety.py', tmp_path / 'compiler_safety.py')
    shutil.copyfile(root / 'scripts/windows-rocm64k/stable_triton_knob.py', tmp_path / 'stable_triton_knob.py')
    trace = tmp_path / 'initializations.jsonl'
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + str(root / 'python'),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1',
               FT_ROCM1010_RANGE_SAFE='1', FT_ROCM1010_INIT_OUTPUT=str(trace),
               FREETOKEN_TRITON_CACHE_AMD_KNOB='0', AMDGCN_USE_BUFFER_OPS='1')
    return tmp_path, trace, env


def test_cli_remains_runtime_free_until_kernel_import(trial_environment):
    _, trace, env = trial_environment
    code = '''import sys
import freetoken.cli
assert 'torch' not in sys.modules and 'triton' not in sys.modules
import freetoken.kernel
import torch, triton
assert not torch.cuda.is_initialized()
assert triton.knobs.runtime.add_stages_inspection_hook is not None
assert triton.knobs.amd.__dict__['use_buffer_ops'] is True
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['before_gpu_initialization'] is True


def test_spawned_worker_installs_its_own_hook(trial_environment):
    directory, trace, env = trial_environment
    script = directory / 'spawn_probe.py'
    script.write_text('''import multiprocessing as mp
import sys
def worker():
    import freetoken.kernel
    import torch, triton
    assert not torch.cuda.is_initialized()
    assert triton.knobs.runtime.add_stages_inspection_hook is not None
if __name__ == '__main__':
    assert 'torch' not in sys.modules
    process = mp.get_context('spawn').Process(target=worker)
    process.start()
    process.join(30)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise RuntimeError('spawn probe timed out')
    assert process.exitcode == 0
    assert 'torch' not in sys.modules
''', encoding='utf-8')
    result = subprocess.run([sys.executable, '-B', str(script)], env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['pid'] != rows[0]['parent_pid']
    assert rows[0]['before_gpu_initialization'] is True


def test_bootstrap_failure_terminates_before_kernel_execution(trial_environment):
    _, trace, env = trial_environment
    code = '''import torch
torch.__version__ = 'unqualified'
import freetoken.kernel
raise RuntimeError('kernel import should not have continued')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 86
    assert 'worker bootstrap failed' in result.stderr
    assert not trace.exists()
