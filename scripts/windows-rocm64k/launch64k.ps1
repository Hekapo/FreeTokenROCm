#requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$Label,
    [Parameter(Mandatory)][string]$Src,
    [Parameter(Mandatory)][string]$RunDir,
    [ValidateSet(1919)][int]$Port = 1919,
    [Parameter(Mandatory)][string]$ModelPath,
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][string]$SdkDevel,
    [Parameter(Mandatory)][string]$TritonSite,
    [Parameter(Mandatory)][string]$StateManifest,
    [string]$PointerFile = '',
    [Parameter(Mandatory)][string]$StateRoot,
    [Parameter(Mandatory)][string]$ResourceGuard,
    [Parameter(Mandatory)][string]$MsvcRoot,
    [Parameter(Mandatory)][string]$WindowsSdkRoot,
    [ValidateSet('10.0.26100.0')][string]$WindowsSdkVersion,
    [switch]$PreflightOnly,
    [ValidateSet('rocm1000','rocm1010')][string]$Runtime = 'rocm1000',
    [ValidateSet('windows','one_per_core')][string]$BackendCpuAffinity = 'windows'
)
$ErrorActionPreference = 'Stop'
$stopRequestFile = Join-Path $StateRoot 'stop-request.json'
if (Test-Path -LiteralPath $stopRequestFile) { throw "Managed stop pending: $stopRequestFile" }
if (-not $PreflightOnly -and -not $PointerFile) { throw 'Use the managed controller for startup and recovery ownership' }
$KeepRunning = $true
$CacheStableTritonKnob = $true
$MaxRunningRequests = 1
$MaxSeqLenOverride = 65536
$NumTokens = 65536
$DefaultMaxOutputTokens = 2048
$ExpectedTritonVersion = '3.8.0'
$FrontendGenerationLimit = 1
$source64k = $Src
$labRoot = $PSScriptRoot
$venvRoot = Split-Path -Parent (Split-Path -Parent $Python)
$runtimeName = $Runtime
. (Join-Path $PSScriptRoot 'service38_safety.ps1')
. (Join-Path $PSScriptRoot 'service64k_paths.ps1')
. (Join-Path $PSScriptRoot 'service64k_release.ps1')
if ([IO.Path]::GetFullPath($TritonSite) -ine (Join-Path $venvRoot 'Lib\site-packages') -or
    [IO.Path]::GetFullPath($SdkDevel) -ine (Join-Path $venvRoot 'Lib\site-packages\_rocm_sdk_devel') -or
    [IO.Path]::GetFullPath($StateManifest) -ine (Get-Service64kSourceManifest) -or
    [IO.Path]::GetFullPath($ResourceGuard) -ine (Join-Path $RunDir 'resource-guard.json') -or
    (-not $PreflightOnly -and [IO.Path]::GetFullPath($PointerFile) -ine (Join-Path $StateRoot 'active-service.json')) -or
    -not [IO.Path]::GetFullPath($RunDir).StartsWith(
        ([IO.Path]::GetFullPath((Join-Path $StateRoot 'runs')).TrimEnd('\') + '\'), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Use the reviewed runtime and managed state paths'
}
$null = Read-Service38WorkingRelease
Assert-Service64kLocalPaths ([pscustomobject]@{
    venv_root=$venvRoot; model_path=$ModelPath; msvc_root=$MsvcRoot; windows_sdk_root=$WindowsSdkRoot; state_root=$StateRoot
})
$fingerprintScript = Join-Path $PSScriptRoot 'reviewed_source.py'
$env:FREETOKEN_TRITON_CACHE_AMD_KNOB = '0'
$env:AMDGCN_USE_BUFFER_OPS = '1'
$env:FREETOKEN_TRACE_TOKEN_IDS = '0'
$env:FT_BENCH_DISABLE_OVERLAP = '0'
$env:FT_DIAG_LOGIT_CHECK_EVERY = '64'
foreach ($name in @('FT_FINITE_DIAGNOSTICS_DIR','FT_DIAG_SCHED_STACK_INTERVAL_S',
    'FT_STEP1_PROFILE_OUTPUT','FT_STEP1_EVENT_OUTPUT','FT_STEP1_STATS_OUTPUT','FT_STEP1_STOP_FILE',
    'FT_SLOWDOWN_BACKEND_OUTPUT','FT_SLOWDOWN_PROFILE_CONTROL','FT_SLOWDOWN_KNOB_COMPARE',
    'FT_BACKEND_CPU_AFFINITY','FT_BACKEND_CPU_AFFINITY_OUTPUT')) {
    Remove-Item -LiteralPath ('Env:' + $name) -ErrorAction SilentlyContinue
}
$env:FT_BACKEND_CPU_AFFINITY = $BackendCpuAffinity
$env:FT_BACKEND_CPU_AFFINITY_OUTPUT = Join-Path $RunDir 'backend-affinity.jsonl'
$SDKDEVEL = $SdkDevel
$MSVC = $MsvcRoot
$SDK10 = $WindowsSdkRoot
foreach ($d in "server-logs", "temp", "tvm-ffi-cache", "triton-cache", "torch-extensions", "xdg-cache", "pycache", "hf-home", "miopen-db", "miopen-cache", "hip-cache") {
    New-Item -ItemType Directory -Force -Path "$RunDir\$d" | Out-Null
}

# environment mirrors the real service launch.json (q36service, 2026-09-21), caches isolated per run
$env:HIP_PATH = $SDKDEVEL; $env:ROCM_HOME = $SDKDEVEL; $env:ROCM_PATH = $SDKDEVEL; $env:HIPFIRE_ROCM_PATH = $SDKDEVEL
$env:HIP_PLATFORM = "amd"; $env:CC = "$SDKDEVEL\lib\llvm\bin\clang-cl.exe"; $env:DISTUTILS_USE_SDK = "1"; $env:MAX_JOBS = "1"
$env:PYTORCH_ROCM_ARCH = "gfx1201"; $env:FREETOKEN_ROCM_ARCH = "gfx1201"; $env:TVM_FFI_ROCM_ARCH_LIST = "gfx1201"
$env:TRITON_LIBHIP_PATH = "$SDKDEVEL\bin\amdhip64_7.dll"; $env:TVM_FFI_DISABLE_TORCH_C_DLPACK = "1"
$env:PYTHONNOUSERSITE = "1"; $env:PYTHONDONTWRITEBYTECODE = "1"; $env:FREETOKEN_MAMBA_SSM_DTYPE = "float32"
$env:FT_GGUF_BACKEND = "hip"; $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"; $env:FREETOKEN_DISABLE_OVERLAP_SCHEDULING = $(if ($env:FT_BENCH_DISABLE_OVERLAP) { $env:FT_BENCH_DISABLE_OVERLAP } else { "0" })
$env:INCLUDE = "$MSVC\include;$MSVC\..\..\..\Auxiliary\VS\include;$SDK10\include\$WindowsSdkVersion\ucrt;$SDK10\include\$WindowsSdkVersion\um;$SDK10\include\$WindowsSdkVersion\shared;$SDK10\include\$WindowsSdkVersion\winrt;$SDK10\include\$WindowsSdkVersion\cppwinrt"
$env:LIB = "$MSVC\lib\x64;$SDK10\lib\$WindowsSdkVersion\ucrt\x64;$SDK10\lib\$WindowsSdkVersion\um\x64"
$env:LIBPATH = "$MSVC\lib\x64;$MSVC\lib\x86\store\references;$SDK10\UnionMetadata\$WindowsSdkVersion;$SDK10\References\$WindowsSdkVersion"
$env:PATH = "$SDKDEVEL\bin;$SDKDEVEL\lib\llvm\bin;$MSVC\bin\HostX64\x64;$SDK10\bin\$WindowsSdkVersion\x64;$(Split-Path $Python);$env:PATH"
$env:TVM_FFI_CACHE_DIR = "$RunDir\tvm-ffi-cache"; $env:TRITON_CACHE_DIR = "$RunDir\triton-cache"; $env:TORCH_EXTENSIONS_DIR = "$RunDir\torch-extensions"
$env:XDG_CACHE_HOME = "$RunDir\xdg-cache"; $env:TEMP = "$RunDir\temp"; $env:TMP = "$RunDir\temp"; $env:PYTHONPYCACHEPREFIX = "$RunDir\pycache"
$env:HF_HOME = "$RunDir\hf-home"; $env:MIOPEN_USER_DB_PATH = "$RunDir\miopen-db"; $env:MIOPEN_CUSTOM_CACHE_DIR = "$RunDir\miopen-cache"; $env:HIP_CACHE_DIR = "$RunDir\hip-cache"
$env:PYTHONPATH = "$TritonSite;$Src\python"
foreach ($name in @('FT_ROCM1010_RANGE_SAFE','FT_ROCM1010_INIT_OUTPUT')) {
    Remove-Item -LiteralPath ('Env:' + $name) -ErrorAction SilentlyContinue
}
if ($Runtime -eq 'rocm1010') {
    $env:PYTHONPATH = "$PSScriptRoot\rocm1010;$PSScriptRoot;$TritonSite;$Src\python"
    $env:FT_ROCM1010_RANGE_SAFE = '1'
    $env:FT_ROCM1010_INIT_OUTPUT = Join-Path $RunDir 'compiler-init.jsonl'
}
$tritonProbe = @(& $Python -c 'import triton; print(triton.__version__); print(triton.__file__)')
$resolvedTritonSite = (Resolve-Path -LiteralPath $TritonSite).Path
if ($LASTEXITCODE -ne 0 -or $tritonProbe.Count -lt 2 -or $tritonProbe[0] -ne $ExpectedTritonVersion -or
    -not $tritonProbe[1].StartsWith($resolvedTritonSite, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Triton 3.8 overlay verification failed: $($tritonProbe -join '; ')"
}
$env:FREETOKEN_EXPERIMENT_BASELINE = "1"; $env:FREETOKEN_FUSED_COPY = "0"; $env:FREETOKEN_DISABLE_KERNEL_CACHE = "1"; $env:FREETOKEN_DISABLE_JIT = "0"
Remove-Item Env:FREETOKEN_KERNEL_CACHE_DIR -ErrorAction SilentlyContinue
foreach ($knob in @('GPU_MAX_HW_QUEUES', 'DEBUG_CLR_BATCH_CPU_SYNC_SIZE', 'GPU_FLUSH_ON_EXECUTION', 'GPU_CLR_PROFILE_OUTPUT', 'FT_STEP1_STATS_OUTPUT')) {
    Remove-Item -Path "Env:$knob" -ErrorAction SilentlyContinue
}

$serveArgs = @("-B", "-m", "freetoken.cli", "serve", "--model", $ModelPath,
    "--served-model-name", "qwen3.6-35b-a3b-q4_1", "--host", "127.0.0.1", "--port", "$Port", "--tensor-parallel-size", "1",
    "--dtype", "bfloat16", "--moe-backend", "offload", "--moe-cache-size", "2048", "--moe-cache-policy", "lru",
    "--expert-load", "serial", "--max-running-requests", "$MaxRunningRequests", "--max-seq-len-override", "$MaxSeqLenOverride", "--page-size", "1",
    "--num-tokens", "$NumTokens", "--max-prefill-length", "512", "--cuda-graph-max-bs", "0", "--cache-type", "naive",
    "--attention-backend", "triton", "--memory-ratio", "0.80", "--sampling-defaults", "none", "--max-output-tokens", "$DefaultMaxOutputTokens")
$serveArgs += @('--frontend-generation-limit', '1')
$env:FT_TRITON_STABLE_KNOB_OUTPUT = Join-Path $RunDir 'stable-knob-init.jsonl'
$serveArgs = @('-B', (Join-Path $PSScriptRoot 'cached_stats_launcher.py')) + @($serveArgs[3..($serveArgs.Count - 1)])
$guard = Get-Content -LiteralPath $ResourceGuard -Raw | ConvertFrom-Json
$currentGuard = Get-Service64kGuard
if ($guard.gpu_instance -cne $currentGuard.gpu_instance -or
    $null -eq $guard.gpu_total_gib -or -not [double]::IsFinite([double]$guard.gpu_total_gib) -or
    [Math]::Abs([double]$guard.gpu_total_gib - [double]$currentGuard.gpu_total_gib) -gt 0.000001) {
    throw 'Run resource guard does not identify the current reviewed GPU'
}
$profile = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'profile.json') -Raw | ConvertFrom-Json
foreach ($phase in @('startup','runtime','recovery')) {
    foreach ($name in @('free_physical_gib','free_commit_gib','gpu_local_free_gib')) {
        if ($null -eq $guard.$phase.$name -or -not [double]::IsFinite([double]$guard.$phase.$name) -or
            $guard.$phase.$name -lt $profile.$phase.$name) {
            throw "Run resource guard was weakened: $phase/$name"
        }
    }
}
$result = [ordered]@{
    label=$Label; runtime=$Runtime; compiler_workaround=($Runtime -eq 'rocm1010');
    backend_cpu_affinity=$BackendCpuAffinity; cpu_affinity_probe=(Get-Service64kCpuAffinityProbe $BackendCpuAffinity);
    src=$Src; head=(& git -C $Src rev-parse HEAD); port=$Port; python=$Python;
    model_path=$ModelPath; triton_site=$resolvedTritonSite; triton_version=$tritonProbe[0];
    args=$serveArgs; state_manifest=$StateManifest;
    launcher_sha256=(Get-FileHash -LiteralPath $MyInvocation.MyCommand.Path).Hash.ToLowerInvariant()
}

if ($PreflightOnly) {
    if ($guard) { $result.resources = Assert-Service38Resources $guard 'startup' }
    ($result | ConvertTo-Json -Depth 8) | Set-Content -Encoding utf8 "$RunDir\preflight.json"
    Write-Output "preflight passed: Triton $($tritonProbe[0]) from $($tritonProbe[1])"
    return
}
$listeners = @(Get-Service38Listener $Port)
if ($listeners.Count) { throw "Port $Port is already listening; refusing to start" }
if ($guard) { $result.resources_before = Assert-Service38Resources $guard 'startup' }
$started = Get-Date
$proc = $null
$keepOwned = $false
$stateFile = Join-Path $RunDir 'service-state.json'
if (Test-Path -LiteralPath $stateFile) { throw "Run directory already has service state: $stateFile" }
$faultFile = Join-Path $StateRoot 'recovery-required.json'
$observer = $null
try {
    if (Test-Path -LiteralPath $stopRequestFile) { throw 'Managed stop was requested before process creation' }
    # Written immediately before process creation. If preflight fails without this file,
    # the controller can discard its early pointer without treating it as an orphan risk.
    Write-Service38Json (Join-Path $RunDir 'spawn-intent.json') ([ordered]@{
        at = (Get-Date).ToString('o'); port = $Port; python = (Resolve-Path -LiteralPath $Python).Path })
    $proc = Start-Process -FilePath $Python -ArgumentList (Join-Service64kArguments $serveArgs) -WorkingDirectory $RunDir -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "$RunDir\server-logs\stdout.log" -RedirectStandardError "$RunDir\server-logs\stderr.log"
    # This is the first operation after Start-Process. The controller records this path before
    # waiting for health, so a failed/interrupted startup still has a recoverable identity.
    $state = [ordered]@{ status = 'STARTING'; pid = $proc.Id;
        root_start_filetime = $proc.StartTime.ToUniversalTime().ToFileTimeUtc();
        root_started_utc = $proc.StartTime.ToUniversalTime().ToString('o');
        python = (Resolve-Path -LiteralPath $Python).Path; port = $Port; profile = $Label;
        src = $Src; head = $result.head; args = $serveArgs; state_manifest = $StateManifest;
        resource_guard = $ResourceGuard; cache_stable_triton_knob = [bool]$CacheStableTritonKnob;
        backend_cpu_affinity = $BackendCpuAffinity; cpu_affinity_receipt = $null;
        started_at = (Get-Date).ToString('o'); run_dir = $RunDir }
    Write-Service38Json $stateFile $state
    if ($PointerFile) {
        Write-Service38Json $PointerFile ([ordered]@{
            state_file = $stateFile; started_at = $state.started_at; recovery_state = $state })
    }
    if (Test-Path -LiteralPath $stopRequestFile) { throw 'Managed stop was requested after process creation' }
    $result.pid = $proc.Id
    if ($guard) {
        $observerArgs = @('-NoProfile', '-File', (Join-Path $PSScriptRoot 'observe_service38.ps1'),
            '-StateFile', $stateFile, '-GuardFile', $ResourceGuard, '-FaultFile', $faultFile,
            '-EventsFile', (Join-Path $RunDir 'observer.jsonl'),
            '-ReadyFile', (Join-Path $RunDir 'observer-ready.json'))
        $observer = Start-Process -FilePath (Join-Path $PSHOME 'pwsh.exe') -ArgumentList (Join-Service64kArguments $observerArgs) `
            -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $RunDir 'observer.stdout.log') `
            -RedirectStandardError (Join-Path $RunDir 'observer.stderr.log')
        $state.observer_pid = $observer.Id
        $state.observer_start_filetime = $observer.StartTime.ToUniversalTime().ToFileTimeUtc()
        Write-Service38Json $stateFile $state
        if ($PointerFile) {
            Write-Service38Json $PointerFile ([ordered]@{
                state_file = $stateFile; started_at = $state.started_at; recovery_state = $state })
        }
        $observerReady = Join-Path $RunDir 'observer-ready.json'
        $observerDeadline = (Get-Date).AddSeconds(15)
        while (-not (Test-Path -LiteralPath $observerReady) -and (Get-Date) -lt $observerDeadline) {
            Start-Sleep -Milliseconds 250
            $observer.Refresh()
            if ($observer.HasExited -or (Test-Path -LiteralPath $faultFile)) {
                throw "Resource observer failed to start; inspect $RunDir and $faultFile"
            }
        }
        if (-not (Test-Path -LiteralPath $observerReady)) {
            throw "Resource observer did not confirm readiness; inspect $RunDir"
        }
    }
    $healthy = $false; $health = $null
    while (((Get-Date) - $started).TotalSeconds -lt 600) {
        Start-Sleep -Seconds 5
        $proc.Refresh()
        if ($proc.HasExited) { $result.exited_early = $proc.ExitCode; break }
        if (Test-Path -LiteralPath $stopRequestFile) { throw 'Managed stop was requested during startup' }
        if (Test-Path -LiteralPath $faultFile) { throw "Recovery required; inspect $faultFile" }
        if (-not (Assert-Service38PortOwner $state)) { continue }
        try { $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($health.status -eq 'ok') { $healthy = $true; break } } catch {}
    }
    $result.cold_start_s = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    $result.healthy = $healthy; $result.health = $health
    if ($healthy) {
        $state.cpu_affinity_receipt = Assert-Service64kBackendCpuAffinity $state $BackendCpuAffinity
        $result.cpu_affinity_receipt = $state.cpu_affinity_receipt
    }

function Complete([string]$prompt) {
    $body = @{ model = "qwen3.6-35b-a3b-q4_1"; prompt = $prompt; max_tokens = 64; temperature = 0 } | ConvertTo-Json
    $t0 = Get-Date
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 180
        return [ordered]@{ prompt = $prompt; latency_s = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2); finish_reason = $r.choices[0].finish_reason; usage = $r.usage; text = $r.choices[0].text; error = $null }
    } catch { return [ordered]@{ prompt = $prompt; error = $_.Exception.Message } }
}

if ($KeepRunning -and $healthy) {
    if (Test-Path -LiteralPath $stopRequestFile) { throw 'Managed stop was requested before readiness probe' }
    $result.probe = Complete 'Reply with one short sentence confirming that the service is ready.'
    if (Test-Path -LiteralPath $stopRequestFile) { throw 'Managed stop was requested during readiness probe' }
    if (-not $result.probe.error -and $result.probe.usage.completion_tokens -gt 0) {
        $state.status = 'RUNNING'
        Write-Service38Json $stateFile $state
        if ($PointerFile) {
            Write-Service38Json $PointerFile ([ordered]@{
                state_file = $stateFile; started_at = $state.started_at; recovery_state = $state })
        }
        Write-Service38Json (Join-Path $RunDir 'startup-result.json') $result
        $keepOwned = $true
        Write-Output "service running: pid=$($proc.Id) port=$Port Triton=$($tritonProbe[0])"
        return
    }
    $result.probe_failed = $true
    $healthy = $false
}

    throw 'Managed startup or readiness request failed'
} finally {
    if ($proc -and -not $keepOwned) {
        try {
            $proc.Refresh()
            $exitedBeforeStop = $proc.HasExited
            $exitCodeBeforeStop = if ($exitedBeforeStop) { $proc.ExitCode } else { $null }
            $stopWasRequested = Test-Path -LiteralPath $stopRequestFile
            Write-Service38StopIntent $RunDir 'launcher cleanup'
            $saved = if (Test-Path -LiteralPath $stateFile) {
                Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
            } else {
                # Covers interruption between Start-Process and the first state-file write.
                [pscustomobject]@{ pid = $proc.Id; port = $Port; python = (Resolve-Path $Python).Path;
                    root_start_filetime = $proc.StartTime.ToUniversalTime().ToFileTimeUtc();
                    root_started_utc = $proc.StartTime.ToUniversalTime().ToString('o'); status = 'STARTING' }
            }
            Stop-Service38Tree $saved
            if ($guard) {
                Wait-Service38ResourceRecovery $guard (Join-Path $RunDir 'resources-after-stop.json')
            }
            $saved.status = 'STOPPED'
            $saved | Add-Member -NotePropertyName stopped_at -NotePropertyValue (Get-Date).ToString('o') -Force
            Write-Service38Json $stateFile $saved
            if ($PointerFile) {
                Write-Service38Json $PointerFile ([ordered]@{
                    state_file = $stateFile; started_at = $saved.started_at; recovery_state = $saved })
            }
        } catch {
            Write-Service38Fault $faultFile "cleanup failed: $($_.Exception.Message)" $stateFile
            throw
        } finally {
            $observerCleanupError = $null
            if ($observer) {
                try {
                    Stop-Service38Observer ([pscustomobject]@{
                        observer_pid = $observer.Id
                        observer_start_filetime = $observer.StartTime.ToUniversalTime().ToFileTimeUtc()
                    })
                } catch {
                    $observerCleanupError = $_
                    Write-Service38Fault $faultFile "observer cleanup failed: $($_.Exception.Message)" $stateFile
                }
            }
            $fatal = @(Get-Service38FatalLogEvents $RunDir | Select-Object -First 1)
            if ($fatal.Count -and -not (Test-Path -LiteralPath $faultFile)) {
                Write-Service38Fault $faultFile "GPU fault in server log: $($fatal[0].Line)" $stateFile
            }
            if ($exitedBeforeStop -and -not $stopWasRequested) {
                Write-Service38Fault $faultFile "service exited before managed stop (code $exitCodeBeforeStop)" $stateFile
            }
            ($result | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath (Join-Path $RunDir 'startup-result.json') -Encoding utf8
            if ($observerCleanupError) { throw $observerCleanupError }
        }
    }
}
