"""Freeze a separate managed 64K smoke runtime from the verified controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError('Reviewed runtime template changed: ' + before[:80])
    return text.replace(before, after)


def prepare(repo: Path, candidate: Path, output: Path, operators: Path,
            state_root: Path | None = None) -> dict:
    base = repo / 'scripts/windows-rocm64k'
    release = json.loads((base / 'reviewed-release.rocm1010.json').read_text())
    old = json.loads((base / 'reviewed-release.json').read_text())
    inputs = {}
    for record in release['runtime_hashes'] + release['evidence_hashes'] + old['runtime_hashes'] + old['evidence_hashes']:
        if record['path'] in inputs and inputs[record['path']]['sha256'] != record['sha256']:
            raise ValueError('Conflicting reviewed runtime input')
        inputs[record['path']] = record
    records = list(inputs.values())
    for record in records:
        path = (base / record['path']).resolve()
        if not path.is_relative_to(base.resolve()) or digest(path) != record['sha256']:
            raise ValueError('The reviewed ROCm10.1 runtime changed')
    gpu = json.loads((operators / 'checks/gpu-qualification.json').read_text())
    xml = ET.parse(operators / 'checks/operators.xml').getroot().find('testsuite').attrib
    if (not gpu['passed'] or gpu['suite'] != 'q4-attention' or int(xml['tests']) != 3
            or any(int(xml[name]) for name in ('errors', 'failures', 'skipped'))):
        raise ValueError('Q4_1 / attention qualification has not passed')
    if digest(base / 'rocm1010/compiler_safety.py') != gpu['compiler_safety']['hook_sha256']:
        raise ValueError('The GPU-qualified compiler workaround changed')
    runtime = output / 'runtime'
    state = state_root if state_root is not None else output / 'state'
    copy_name = 'freetoken__fast_index_copy_1310720_32_2048_1024_4_1'
    copy_dll = (state / 'runs' / 'ctx64k-20261013-000000-00000000' / 'tvm-ffi-cache'
                / (copy_name + '_0000000000000000') / (copy_name + '.dll'))
    if len(str(copy_dll)) >= 260:
        raise ValueError(f'MSVC JIT output would be {len(str(copy_dll))} characters; '
                         'use --state-root with a shorter fresh path')
    if state == repo or state == runtime or state.is_relative_to(runtime):
        raise ValueError('Mutable state must be outside the repository root and runtime')
    if output.exists() or state.exists():
        raise ValueError('Use fresh managed runtime and state directories')
    runtime.mkdir(parents=True)
    state.mkdir(parents=True)
    for record in records:
        destination = runtime / record['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / record['path'], destination)
    controller = (runtime / 'service64k.ps1').read_text()
    quoted_root = "'" + str(repo).replace("'", "''") + "'"
    controller = replace_once(controller,
        "$source64k = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\\..'))",
        '$source64k = ' + quoted_root)
    (runtime / 'service64k.ps1').write_text(controller, encoding='utf-8')
    for record in release['runtime_hashes']:
        record['sha256'] = digest(runtime / record['path'])
    (runtime / 'reviewed-release.rocm1010.json').write_text(json.dumps(release, indent=2) + '\n', encoding='utf-8')
    for record in old['runtime_hashes']:
        record['sha256'] = digest(runtime / record['path'])
    (runtime / 'reviewed-release.json').write_text(json.dumps(old, indent=2) + '\n', encoding='utf-8')
    config = json.loads((repo / '.local/windows-rocm64k/config.json').read_text())
    config.update(schema=2, runtime='rocm1010', venv_root=str(candidate), state_root=str(state))
    config_path = state / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    result = {'runtime': str(runtime), 'config': str(config_path), 'state': str(state),
              'source': str(repo), 'venv': str(candidate), 'model_started': False,
              'ordinary_profile_selected': False}
    (output / 'trial.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--venv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state-root', type=Path,
                        help='Fresh short state/cache path for the Windows linker')
    parser.add_argument('--operators', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo.resolve(), args.venv.resolve(), args.output.resolve(), args.operators.resolve(),
            args.state_root.resolve() if args.state_root is not None else None)
