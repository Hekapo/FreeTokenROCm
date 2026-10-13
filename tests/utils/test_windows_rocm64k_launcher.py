"""Portable source identity and managed launch contracts; no model or GPU runtime."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts/windows-rocm64k"
SPEC = importlib.util.spec_from_file_location("reviewed_windows64k_source", MODULE / "reviewed_source.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
PWSH = os.environ.get("FT_TEST_PWSH") or shutil.which("pwsh")
WINDOWS = pytest.mark.skipif(os.name != "nt" or not PWSH, reason="Windows PowerShell 7 contract")


@pytest.fixture
def source(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    paths = {"python/freetoken/a.py": b"value = 1\n", "pyproject.toml": b"name = 'fixture'\n",
             "setup.py": b"value = 2\n"}
    for name, data in paths.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in paths.items()}
    accepted = json.loads((MODULE / "accepted-source.json").read_text(encoding="utf-8"))
    state = {"schema": 1, "mode": "repository64k-reviewed-source",
             "python_files": {"python/freetoken/a.py": hashes["python/freetoken/a.py"]},
             "build_files": {name: hashes[name] for name in ("pyproject.toml", "setup.py")},
             "packages": accepted["packages"],
             "native_files": {name: "0" * 64 for name in CHECKER.NATIVE_FILES}}
    state["canonical_python_files"] = state["python_files"].copy()
    state["canonical_build_files"] = state["build_files"].copy()
    distributions = [SimpleNamespace(metadata={"Name": name}, version=version)
                     for name, version in state["packages"].items()]
    venv = tmp_path / "venv"
    with mock.patch.object(CHECKER, "sys", SimpleNamespace(prefix=str(venv), version_info=(3, 12, 0))), \
         mock.patch.object(CHECKER.metadata, "distributions", return_value=distributions), \
         mock.patch.object(CHECKER, "digest", return_value="0" * 64):
        yield tmp_path, state, venv


def test_relocated_crlf_checkout_and_docs_preserve_source_identity(source):
    repo, state, venv = source
    assert CHECKER.verify(repo, state, venv)["reviewed_source_verified"]
    (repo / "python/freetoken/a.py").write_bytes(b"value = 1\r\n")
    (repo / "README.md").write_text("Documentation-only change", encoding="utf-8")
    assert CHECKER.verify(repo, state, venv)["gpu_runtime_loaded"] is False


def test_changed_source_is_rejected(source):
    repo, state, venv = source
    (repo / "python/freetoken/a.py").write_bytes(b"value = 3\n")
    with pytest.raises(ValueError, match="Reviewed source changed"):
        CHECKER.verify(repo, state, venv)


def test_new_untracked_python_file_is_rejected(source):
    repo, state, venv = source
    (repo / "python/freetoken/new.py").write_bytes(b"value = 3\n")
    with pytest.raises(ValueError, match="file set changed"):
        CHECKER.verify(repo, state, venv)


def test_changed_build_file_is_rejected(source):
    repo, state, venv = source
    (repo / "setup.py").write_bytes(b"value = 3\n")
    with pytest.raises(ValueError, match="Reviewed source changed"):
        CHECKER.verify(repo, state, venv)


def test_other_venv_is_rejected(source):
    repo, state, venv = source
    with pytest.raises(ValueError, match="Python environment"):
        CHECKER.verify(repo, state, venv / "other")


def test_changed_package_is_rejected(source):
    repo, state, venv = source
    state["packages"] = {**state["packages"], "numpy": "0.0"}
    with pytest.raises(ValueError, match="package versions changed"):
        CHECKER.verify(repo, state, venv)


def test_changed_native_binary_is_rejected(source):
    repo, state, venv = source
    state["native_files"][next(iter(CHECKER.NATIVE_FILES))] = "1" * 64
    with pytest.raises(ValueError, match="native binary changed"):
        CHECKER.verify(repo, state, venv)


def test_rocm1010_source_requires_matching_environment(source):
    repo, state, venv = source
    qualification = json.loads((ROOT / 'constraints/windows-rocm1010-qualification-20261012.json').read_text())
    state['packages'] = qualification['packages']
    distributions = [SimpleNamespace(metadata={'Name': name}, version=version)
                     for name, version in state['packages'].items()]
    with mock.patch.object(CHECKER.metadata, 'distributions', return_value=distributions):
        assert CHECKER.verify(repo, state, venv, runtime='rocm1010')['gpu_runtime_loaded'] is False
        with pytest.raises(ValueError, match='package profile'):
            CHECKER.verify(repo, state, venv, runtime='rocm1000')


def test_old_packages_cannot_be_selected_as_rocm1010(source):
    repo, state, venv = source
    with pytest.raises(ValueError, match='package profile'):
        CHECKER.verify(repo, state, venv, runtime='rocm1010')


def run_ps(code, payload=None, module=MODULE):
    env = dict(os.environ, FT_TEST_MODULE=str(module), FT_TEST_PAYLOAD=json.dumps(payload), FT_TEST_PYTHON=sys.executable)
    result = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-Command", code], env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


@WINDOWS
@pytest.mark.parametrize('schema,runtime', [(1, None), (2, 'rocm1000'), (2, 'rocm1010')])
def test_configuration_selects_pinned_runtime(tmp_path, schema, runtime):
    payload = {'schema': schema, 'venv_root': str(tmp_path / 'venv'),
               'model_path': str(tmp_path / 'model.gguf'), 'state_root': str(tmp_path / 'state'),
               'msvc_root': str(tmp_path / 'msvc'), 'windows_sdk_root': str(tmp_path / 'sdk')}
    if runtime:
        payload['runtime'] = runtime
    result = run_ps(r'''
    $source64k=$env:FT_TEST_MODULE; $labRoot=$env:FT_TEST_MODULE
    . (Join-Path $env:FT_TEST_MODULE 'service64k_paths.ps1')
    $config=Resolve-Service64kConfig ($env:FT_TEST_PAYLOAD | ConvertFrom-Json)
    @{runtime=$config.runtime;schema=$config.schema;mode=$config.backend_cpu_affinity} | ConvertTo-Json
    ''', payload)
    assert result['runtime'] == (runtime or 'rocm1000') and result['schema'] == 3
    assert result['mode'] == 'windows'


@WINDOWS
@pytest.mark.parametrize('runtime', ['rocm1100', '', 'ROCM1010'])
def test_unknown_runtime_configuration_is_rejected(tmp_path, runtime):
    payload = {'schema': 2, 'runtime': runtime, 'venv_root': str(tmp_path / 'venv'),
               'model_path': str(tmp_path / 'model.gguf'), 'state_root': str(tmp_path / 'state'),
               'msvc_root': str(tmp_path / 'msvc'), 'windows_sdk_root': str(tmp_path / 'sdk')}
    result = run_ps(r'''
    $source64k=$env:FT_TEST_MODULE; $labRoot=$env:FT_TEST_MODULE
    . (Join-Path $env:FT_TEST_MODULE 'service64k_paths.ps1')
    try {$null=Resolve-Service64kConfig ($env:FT_TEST_PAYLOAD | ConvertFrom-Json); @{allowed=$true} | ConvertTo-Json}
    catch {@{allowed=$false} | ConvertTo-Json}
    ''', payload)
    assert result['allowed'] is False


@WINDOWS
@pytest.mark.parametrize('length,allowed', [(40, True), (180, False)])
def test_jit_output_path_is_checked_before_model_start(length, allowed):
    result = run_ps(r'''
    . (Join-Path $env:FT_TEST_MODULE 'service64k_paths.ps1')
    $root='C:\' + ('x' * ($env:FT_TEST_PAYLOAD | ConvertFrom-Json))
    try {Assert-Service64kJitPaths $root; @{allowed=$true} | ConvertTo-Json}
    catch {@{allowed=$false} | ConvertTo-Json}
    ''', length)
    assert result['allowed'] is allowed


@WINDOWS
@pytest.mark.parametrize('case', ['pass', 'wrong-capacity', 'wrong-content', 'no-stop', 'late-hook',
                                 'wrong-native', 'wrong-hook', 'weak-headroom', 'wrong-scope'])
def test_rocm1010_proof_preserves_numerical_and_stop_contracts(case):
    proof = json.loads((MODULE / 'evidence/rocm1010-short.json').read_text())
    if case == 'wrong-capacity':
        proof['live_kv']['total_pages'] = 8192
    elif case == 'wrong-content':
        proof['smoke']['content_sha256'] = '0' * 64
    elif case == 'no-stop':
        proof['stop']['listeners'] = 1
    elif case == 'late-hook':
        proof['bootstrap_processes'][0]['before_gpu_initialization'] = False
    elif case == 'wrong-native':
        proof['native_files']['torch/lib/torch_hip.dll'] = '0' * 64
    elif case == 'wrong-hook':
        proof['compiler_hook_sha256'] = '0' * 64
    elif case == 'weak-headroom':
        proof['resources']['sampled_minimum_gib']['free_commit_gib'] = 3.9
    elif case == 'wrong-scope':
        proof['long64k_run'] = True
    result = run_ps(r'''
    . (Join-Path $env:FT_TEST_MODULE 'service64k_release.ps1')
    try {Assert-Service64kRocm1010Proof ($env:FT_TEST_PAYLOAD | ConvertFrom-Json); @{allowed=$true} | ConvertTo-Json}
    catch {@{allowed=$false} | ConvertTo-Json}
    ''', proof)
    assert result['allowed'] is (case == 'pass')


@WINDOWS
def test_all_powershell_files_parse():
    result = run_ps(r"""
    $failures = @()
    foreach ($file in (Get-ChildItem -LiteralPath $env:FT_TEST_MODULE -Filter '*.ps1')) {
        $tokens=$null; $errors=$null
        $null=[System.Management.Automation.Language.Parser]::ParseFile($file.FullName,[ref]$tokens,[ref]$errors)
        $failures += @($errors | ForEach-Object {$_.Message})
    }
    ConvertTo-Json -InputObject $failures
    """)
    assert result == []


@WINDOWS
def test_windows_arguments_round_trip_with_spaces_quotes_and_trailing_slashes(tmp_path):
    output = tmp_path / "output with spaces.json"
    values = ["", "path with spaces\\", 'quoted "word"', "кириллица", "--port", "1919"]
    result = run_ps(r"""
    . (Join-Path $env:FT_TEST_MODULE 'service64k_paths.ps1')
    $payload=$env:FT_TEST_PAYLOAD | ConvertFrom-Json
    $arguments=@('-c','import json,sys; print(json.dumps(sys.argv[1:]))') + @($payload.values)
    $p=Start-Process -FilePath $env:FT_TEST_PYTHON -ArgumentList (Join-Service64kArguments $arguments) `
        -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $payload.output
    if ($p.ExitCode -ne 0) {throw 'Argument fixture failed'}
    Get-Content -LiteralPath $payload.output -Raw
    """, {"values": values, "output": str(output)})
    assert result == values


@WINDOWS
@pytest.mark.parametrize("case", ["pass", "low-commit", "nonfinite", "recovery"])
def test_resource_floors(case):
    result = run_ps(r"""
    . (Join-Path $env:FT_TEST_MODULE 'service38_safety.ps1')
    $guard=Get-Content -LiteralPath (Join-Path $env:FT_TEST_MODULE 'profile.json') -Raw | ConvertFrom-Json
    $reading=[pscustomobject]@{free_physical_gib=38;free_commit_gib=45;gpu_local_used_gib=1;gpu_local_free_gib=14}
    $case=$env:FT_TEST_PAYLOAD | ConvertFrom-Json
    if ($case -eq 'low-commit') {$reading.free_commit_gib=44}
    if ($case -eq 'nonfinite') {$reading.free_commit_gib=[double]::NaN}
    if ($case -eq 'recovery') {$guard=Get-Service38RecoveryGuard $guard;$reading.free_commit_gib=28}
    try {$null=Assert-Service38Resources $guard 'startup' $reading; @{allowed=$true} | ConvertTo-Json}
    catch {@{allowed=$false} | ConvertTo-Json}
    """, case)
    assert result["allowed"] is (case in {"pass", "recovery"})


@WINDOWS
@pytest.mark.parametrize("case", ["fresh-luid", "ambiguous-gpu", "missing-counter"])
def test_gpu_identity_is_rebound_and_ambiguous_measurements_rejected(case):
    result = run_ps(r"""
    . (Join-Path $env:FT_TEST_MODULE 'service64k_paths.ps1')
    $labRoot=$env:FT_TEST_MODULE
    function Invoke-Service38Probe {
        $adapter=[pscustomobject]@{vendor_id=4098;device_id=30032;dedicated_video_bytes=15.8GB;flags=0;counter_instance='fresh-luid'}
        $adapters=@($adapter);$counters=@([pscustomobject]@{Name='fresh-luid'})
        $case=$env:FT_TEST_PAYLOAD | ConvertFrom-Json
        if ($case -eq 'ambiguous-gpu') {$adapters+= $adapter}
        if ($case -eq 'missing-counter') {$counters=@()}
        [pscustomobject]@{adapters=$adapters;counters=$counters}
    }
    try {$guard=Get-Service64kGuard; @{allowed=$true;instance=$guard.gpu_instance;total=$guard.gpu_total_gib} | ConvertTo-Json}
    catch {@{allowed=$false} | ConvertTo-Json}
    """, case)
    assert result["allowed"] is (case == "fresh-luid")
    if case == "fresh-luid":
        assert result["instance"] == "fresh-luid" and result["total"] == pytest.approx(15.8)


@WINDOWS
@pytest.mark.parametrize("case", ["pass", "runtime-tamper", "outside-path", "weak-startup", "nan-startup", "wrong-capacity"])
def test_release_rejects_changed_runtime_paths_and_policy(tmp_path, case):
    module = tmp_path / "portable module"
    shutil.copytree(MODULE, module)
    record_path = module / "reviewed-release.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if case == "runtime-tamper":
        with (module / "cached_stats_launcher.py").open("a", encoding="utf-8") as sink:
            sink.write("\nraise RuntimeError('tampered')\n")
    elif case == "outside-path":
        record["runtime_hashes"][0]["path"] = "../outside.ps1"
    elif case == "wrong-capacity":
        record["capacity"] = 8192
    elif case in {"weak-startup", "nan-startup"}:
        profile_path = module / "profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile["startup"]["free_commit_gib"] = 44 if case == "weak-startup" else "NaN"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        for entry in record["runtime_hashes"]:
            if entry["path"] == "profile.json":
                entry["sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record), encoding="utf-8")
    result = run_ps(r"""
    . (Join-Path $env:FT_TEST_MODULE 'service64k_release.ps1')
    function Read-Service64kEvidence($Release) {$Release}
    try {$null=Read-Service38WorkingRelease; @{allowed=$true} | ConvertTo-Json}
    catch {@{allowed=$false} | ConvertTo-Json}
    """, module=module)
    assert result["allowed"] is (case == "pass")


@WINDOWS
def test_failed_start_before_spawn_removes_pointer_without_creating_fault(tmp_path):
    fixture = tmp_path / "launcher fixture.ps1"
    fixture.write_text("param($RunDir,$PointerFile)\nthrow 'fixture failed before spawn'\n", encoding="utf-8")
    result = run_ps(r"""
    . (Join-Path $env:FT_TEST_MODULE 'service38_safety.ps1')
    . (Join-Path $env:FT_TEST_MODULE 'service64k_lifecycle.ps1')
    $payload=$env:FT_TEST_PAYLOAD | ConvertFrom-Json
    $pointerFile=Join-Path $payload.directory 'active-service.json'
    $faultFile=Join-Path $payload.directory 'recovery-required.json'
    $stopRequestFile=Join-Path $payload.directory 'stop-request.json'
    $launcher=$payload.launcher
    function Get-Listener {@()}
    try {Start-ManagedService @{RunDir=$payload.directory} 'fixture'}
    catch {if ($_.Exception.Message -notmatch 'fixture failed before spawn') {throw}}
    @{pointer=(Test-Path -LiteralPath $pointerFile);fault=(Test-Path -LiteralPath $faultFile)} | ConvertTo-Json
    """, {"directory": str(tmp_path), "launcher": str(fixture)})
    assert result == {"pointer": False, "fault": False}


@pytest.fixture
def affinity_module():
    spec = importlib.util.spec_from_file_location('backend_cpu_affinity', MODULE / 'backend_cpu_affinity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('cores,allowed,selected', [
    ([3, 12, 48, 192, 768, 3072, 12288, 49152], 65535, 21845),
    ([3, 12], 10, 10),
    ([3, 12], 3, 1),
    ([1, 2, 4, 8], 15, 15),
])
def test_cpu_affinity_honors_topology_and_inherited_mask(affinity_module, cores, allowed, selected):
    assert affinity_module.select_one_per_core(cores, allowed) == selected


@pytest.mark.parametrize('cores,allowed', [([], 15), ([3, 3], 3), ([3, 6], 7),
                                          ([0, 12], 15), ([3, 12], 0), ([3, 12], 16),
                                          ([True, 12], 15), ([3, 12], True)])
def test_cpu_affinity_rejects_invalid_topology(affinity_module, cores, allowed):
    with pytest.raises(ValueError):
        affinity_module.select_one_per_core(cores, allowed)


class FakeCpuAffinity:
    def __init__(self, reject=False, groups=1):
        self.mask = 15
        self.calls = []
        self.reject = reject
        self.groups = groups

    def masks(self):
        return self.mask, 15

    def topology(self):
        return {'group_count': self.groups, 'core_masks': [3, 12]}

    def created(self):
        return 42

    def set_mask(self, mask):
        self.calls.append(mask)
        if self.reject and mask == 5:
            raise OSError('affinity denied')
        self.mask = mask


@pytest.mark.parametrize('mode,expected,calls', [('windows', 15, []), ('one_per_core', 5, [5, 15])])
def test_cpu_affinity_changes_only_worker_scope_and_restores(affinity_module, tmp_path, mode, expected, calls):
    native = FakeCpuAffinity()
    output = tmp_path / 'backend-affinity.jsonl'
    seen = []

    def worker(args, ack):
        seen.append(native.mask)
        return 'worker-result'

    result = affinity_module.run_with_affinity(worker, 'args', 'ack', mode, output,
                                               native=native, gpu_initialized=lambda: False)
    assert result == 'worker-result' and seen == [expected]
    assert native.mask == 15 and native.calls == calls
    receipts = [json.loads(line) for line in output.read_text().splitlines()]
    assert receipts[0]['role'] == 'backend' and receipts[0]['applied_mask'] == expected
    assert receipts[0]['creation_filetime'] == 42 and receipts[0]['before_gpu_initialization']


@pytest.mark.parametrize('failure', ['native-denied', 'groups', 'late-gpu', 'receipt'])
def test_cpu_affinity_failure_prevents_worker_start(affinity_module, tmp_path, failure):
    native = FakeCpuAffinity(reject=failure == 'native-denied', groups=2 if failure == 'groups' else 1)
    output = tmp_path / ('missing/out.jsonl' if failure == 'receipt' else 'out.jsonl')
    worker = mock.Mock()
    with pytest.raises((OSError, RuntimeError)):
        affinity_module.run_with_affinity(worker, None, None, 'one_per_core', output,
                                         native=native, gpu_initialized=lambda: failure == 'late-gpu')
    worker.assert_not_called()
    assert native.mask == 15


def test_cpu_affinity_restores_on_worker_exception(affinity_module, tmp_path):
    native = FakeCpuAffinity()
    worker = mock.Mock(side_effect=RuntimeError('worker failed'))
    with pytest.raises(RuntimeError, match='worker failed'):
        affinity_module.run_with_affinity(worker, None, None, 'one_per_core', tmp_path / 'out.jsonl',
                                         native=native, gpu_initialized=lambda: False)
    assert native.mask == 15 and native.calls == [5, 15]


@WINDOWS
@pytest.mark.parametrize('mode,allowed', [('windows', True), ('one_per_core', True),
                                       ('ONE_PER_CORE', False), ('', False), (None, False), (True, False),
                                       ([], False), (['windows'], False)])
def test_cpu_affinity_configuration_validates_explicit_mode(tmp_path, mode, allowed):
    payload = {'schema': 3, 'runtime': 'rocm1010', 'backend_cpu_affinity': mode,
               'venv_root': str(tmp_path / 'venv'), 'model_path': str(tmp_path / 'model.gguf'),
               'state_root': str(tmp_path / 'state'), 'msvc_root': str(tmp_path / 'msvc'),
               'windows_sdk_root': str(tmp_path / 'sdk')}
    result = run_ps(r'''
    $source64k=$env:FT_TEST_MODULE; $labRoot=$env:FT_TEST_MODULE
    . (Join-Path $env:FT_TEST_MODULE 'service64k_paths.ps1')
    try {$c=Resolve-Service64kConfig ($env:FT_TEST_PAYLOAD | ConvertFrom-Json); @{allowed=$true;mode=$c.backend_cpu_affinity} | ConvertTo-Json}
    catch {@{allowed=$false} | ConvertTo-Json}
    ''', payload)
    assert result['allowed'] is allowed
    if allowed:
        assert result['mode'] == mode


def test_cpu_affinity_hook_is_importable_and_does_not_run_in_frontend(affinity_module, monkeypatch, tmp_path):
    import pickle

    def _run_scheduler(args, ack_queue):
        return 'scheduler'

    _run_scheduler.__module__ = 'freetoken.server.launch'
    token_worker = object()
    launch = SimpleNamespace(_run_scheduler=_run_scheduler, _run_tokenize_worker=token_worker,
                             _report_startup_error=mock.Mock())
    apply = mock.Mock()
    monkeypatch.setattr(affinity_module, 'run_with_affinity', apply)
    monkeypatch.setitem(sys.modules, 'backend_cpu_affinity', affinity_module)
    affinity_module.install_scheduler_hook(launch, 'one_per_core', tmp_path / 'out.jsonl', lambda: False)
    assert pickle.loads(pickle.dumps(launch._run_scheduler)) is affinity_module.scheduler_worker
    assert launch._run_tokenize_worker is token_worker
    apply.assert_not_called()
    args = SimpleNamespace(tp_info=SimpleNamespace(size=1, rank=0))
    monkeypatch.setattr(affinity_module.mp, 'current_process', lambda: SimpleNamespace(name='freetoken-detokenizer-0'))
    with pytest.raises(RuntimeError, match='scheduler worker'):
        launch._run_scheduler(args, None)
    apply.assert_not_called()


@WINDOWS
@pytest.mark.parametrize('case', ['stopped', 'active', 'fault', 'probe-failed'])
def test_cpu_affinity_setter_is_atomic_and_requires_stopped_service(tmp_path, case):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'schema': 3, 'backend_cpu_affinity': 'windows'}))
    result = run_ps(r'''
    . (Join-Path $env:FT_TEST_MODULE 'service38_safety.ps1')
    . (Join-Path $env:FT_TEST_MODULE 'service64k_paths.ps1')
    $p=$env:FT_TEST_PAYLOAD | ConvertFrom-Json
    $configPath=$p.path; $config=Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    function Assert-ProfileStartAllowed {if ($p.case -in @('active','fault')) {throw 'active or recovery-required'}}
    function Read-Service38WorkingRelease {@{valid=$true}}
    function Get-Service64kCpuAffinityProbe {if ($p.case -eq 'probe-failed') {throw 'probe failed'}; @{valid=$true}}
    try {$null=Set-Service64kCpuAffinity 'one_per_core'; $allowed=$true} catch {$allowed=$false}
    @{allowed=$allowed;mode=(Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).backend_cpu_affinity} | ConvertTo-Json
    ''', {'path': str(path), 'case': case})
    assert result == {'allowed': case == 'stopped', 'mode': 'one_per_core' if case == 'stopped' else 'windows'}


@WINDOWS
def test_cpu_affinity_windows_spawn_is_backend_only(tmp_path):
    fixture = tmp_path / 'spawn_affinity.py'
    output = tmp_path / 'affinity.jsonl'
    fixture.write_text(r'''
import json, multiprocessing as mp, pathlib, sys, types
sys.path.insert(0, sys.argv[1])
import backend_cpu_affinity as affinity
launch=types.ModuleType('freetoken.server.launch')
exec("""def _run_scheduler(args, ack_queue):
    ack_queue.put(("backend", __import__("backend_cpu_affinity").WindowsCpuAffinity().masks()[0], __import__("os").getpid()))
""", launch.__dict__)
launch._report_startup_error=lambda q,e:q.put(('error',str(e)))
affinity.install_scheduler_hook(launch, 'one_per_core', pathlib.Path(sys.argv[2]), lambda:False)
def tokenizer(q):
    q.put(('tokenizer',affinity.WindowsCpuAffinity().masks()[0]))
if __name__=='__main__':
    before=affinity.WindowsCpuAffinity().masks()[0]
    expected=affinity.probe_policy('one_per_core')['applied_mask']
    ctx=mp.get_context('spawn'); q=ctx.Queue()
    args=types.SimpleNamespace(tp_info=types.SimpleNamespace(size=1,rank=0))
    backend=ctx.Process(target=launch._run_scheduler,args=(args,q),name='freetoken-TP0-scheduler')
    token=ctx.Process(target=tokenizer,args=(q,),name='freetoken-detokenizer-0')
    try:
        backend.start(); token.start()
        received=[q.get(timeout=10),q.get(timeout=10)]
        backend.join(10); token.join(10)
        assert backend.exitcode==0 and token.exitcode==0
        after=affinity.WindowsCpuAffinity().masks()[0]
        print(json.dumps({'before':before,'after':after,'expected':expected,'received':received,'gpu_loaded':'torch' in sys.modules}))
    finally:
        for p in (backend,token):
            if p.pid and p.is_alive():p.terminate();p.join(5)
        q.close();q.join_thread()
''', encoding='utf-8')
    result = subprocess.run([sys.executable, '-B', str(fixture), str(MODULE), str(output)],
                            capture_output=True, text=True, encoding='utf-8', timeout=35,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    messages = {row[0]: row for row in record['received']}
    assert record['before'] == record['after'] == messages['tokenizer'][1]
    assert messages['backend'][1] == record['expected'] and not record['gpu_loaded']
    receipts = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row['event'] for row in receipts] == ['applied', 'restored']
    assert {row['pid'] for row in receipts} == {messages['backend'][2]}


@WINDOWS
@pytest.mark.parametrize('case', ['pass', 'foreign-pid', 'reused-pid', 'wrong-mask', 'late-gpu', 'overlap'])
def test_cpu_affinity_receipt_requires_live_owned_worker(tmp_path, case):
    result = run_ps(r'''
    . (Join-Path $env:FT_TEST_MODULE 'service64k_paths.ps1')
    $p=$env:FT_TEST_PAYLOAD | ConvertFrom-Json
    $started=[datetime]'2026-01-01T00:00:00Z'
    $r=[ordered]@{event='applied';role='backend';mode='one_per_core';process_name='freetoken-TP0-scheduler';
        before_gpu_initialization=$true;pid=20;creation_filetime=$started.ToFileTimeUtc();
        original_mask=15;applied_mask=5;group_count=1;core_masks=@(3,12)}
    if ($p.case -eq 'foreign-pid') {$r.pid=30}
    if ($p.case -eq 'reused-pid') {$r.creation_filetime-=1}
    if ($p.case -eq 'wrong-mask') {$r.applied_mask=10}
    if ($p.case -eq 'late-gpu') {$r.before_gpu_initialization=$false}
    if ($p.case -eq 'overlap') {$r.core_masks=@(3,3,12)}
    $r|ConvertTo-Json -Depth 4 -Compress|Set-Content -LiteralPath (Join-Path $p.directory 'backend-affinity.jsonl')
    function Get-Service38Tree {param($State);[pscustomobject]@{ProcessId=10};[pscustomobject]@{ProcessId=20}}
    function Get-Process {param($Id,$ErrorAction);[pscustomobject]@{StartTime=$started;ProcessorAffinity=[IntPtr]::new(5)}}
    try {$null=Assert-Service64kBackendCpuAffinity ([pscustomobject]@{pid=10;run_dir=$p.directory}) 'one_per_core';$allowed=$true}
    catch {$allowed=$false;$reason=$_.Exception.Message}
    @{allowed=$allowed;reason=$reason}|ConvertTo-Json
    ''', {'case': case, 'directory': str(tmp_path)})
    assert result['allowed'] is (case == 'pass'), result.get('reason')
