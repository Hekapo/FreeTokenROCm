"""Verify the reviewed source and Python/native environment without loading a GPU runtime."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

NATIVE_FILES = {
    '_rocm_sdk_devel/lib/llvm/bin/clang-cl.exe', '_rocm_sdk_devel/bin/amdhip64_7.dll',
    'torch/lib/torch_hip.dll', 'torch/_C.cp312-win_amd64.pyd', 'triton/_C/libtriton.pyd',
}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def source_paths(repo: Path) -> set[str]:
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z']).decode('utf-8')
    untracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '--others', '--exclude-standard', '-z']).decode('utf-8')
    return {name for name in (tracked + untracked).split('\0')
            if name.startswith('python/') and (repo / name).is_file()}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(repo: Path, state: dict, venv: Path) -> dict:
    require(state.get('schema') == 1 and state.get('mode') == 'repository64k-reviewed-source', 'Wrong reviewed source schema')
    require(identity(venv) == identity(sys.prefix), 'Wrong reviewed Python environment')
    require(sys.version_info[:3] == (3, 12, 0), 'Wrong reviewed Python version')
    files = state['python_files']
    require(set(files) == source_paths(repo), 'Reviewed Python file set changed')
    require(set(state['canonical_python_files']) == set(files), 'Wrong canonical source file set')
    build = state['build_files']
    require(set(build) == {'pyproject.toml', 'setup.py'}, 'Wrong reviewed build file set')
    require(set(state['canonical_build_files']) == set(build), 'Wrong canonical build file set')
    for name, expected in {**state['canonical_python_files'], **state['canonical_build_files']}.items():
        path = (repo / name).resolve()
        require(path.is_relative_to(repo.resolve()), 'Reviewed file escapes source root')
        require(hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected,
                'Reviewed source changed: ' + name)
    expected = state['packages']
    require(len(expected) == 74 and expected.get('torch') == '2.13.0+rocm10.0.0'
            and expected.get('triton-windows') == '3.8.0.post28'
            and expected.get('transformers') == '5.19.0', 'Wrong reviewed package profile')
    installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions()}
    require(installed == expected, 'Reviewed package versions changed')
    native = state['native_files']
    require(set(native) == NATIVE_FILES, 'Wrong reviewed native file set')
    site = Path(sys.prefix) / 'Lib/site-packages'
    for name, expected in native.items():
        require(digest(site / name) == expected, 'Reviewed native binary changed: ' + name)
    return {'reviewed_source_verified': True, 'python_files': len(files), 'build_files': len(build),
            'packages': len(installed), 'native_files': len(native), 'gpu_runtime_loaded': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--verify', type=Path, required=True)
    parser.add_argument('--venv-root', type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.verify.read_text(encoding='utf-8-sig'))
    print(json.dumps(verify(args.repo, state, args.venv_root)))
