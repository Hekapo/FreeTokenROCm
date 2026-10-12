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


def run_ps(code, payload=None, module=MODULE):
    env = dict(os.environ, FT_TEST_MODULE=str(module), FT_TEST_PAYLOAD=json.dumps(payload), FT_TEST_PYTHON=sys.executable)
    result = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-Command", code], env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


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
