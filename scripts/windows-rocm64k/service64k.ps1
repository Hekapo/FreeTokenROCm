#requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('Configure','SetCpuAffinity','Verify','Preflight','Start','Stop','Status','Health','CaptureDiagnostics')]
    [string]$Action = 'Status',
    [string]$ConfigFile = '',
    [string]$VenvRoot = '',
    [string]$ModelPath = '',
    [string]$StateRoot = '',
    [string]$MsvcRoot = '',
    [string]$WindowsSdkRoot = '',
    [ValidateSet('rocm1000','rocm1010')][string]$Runtime = 'rocm1010',
    [ValidateSet('windows','one_per_core')][string]$BackendCpuAffinity = 'windows',
    [switch]$ProgressDiagnostics
)
$ErrorActionPreference = 'Stop'
if (-not $IsWindows) { throw 'This managed profile requires Windows and PowerShell 7' }
$source64k = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$labRoot = $PSScriptRoot
$port = 1919
$Profile = '64k'
. (Join-Path $labRoot 'service38_safety.ps1')
. (Join-Path $labRoot 'service64k_paths.ps1')
. (Join-Path $labRoot 'service64k_release.ps1')
$configPath = if ($ConfigFile) { Resolve-Service64kPath $ConfigFile }
    else { Join-Path $source64k '.local\windows-rocm64k\config.json' }
$controlMutex = $null
$controlMutexOwned = $false
try {
    if ($PSBoundParameters.ContainsKey('ProgressDiagnostics') -and $Action -notin @('Start','Preflight')) {
        throw 'ProgressDiagnostics is accepted only with Start or Preflight'
    }
    if ($Action -eq 'Configure') {
        $controlMutex = [Threading.Mutex]::new($false, 'Local\FreeTokenROCm-Service38-1919')
        try { $controlMutexOwned = $controlMutex.WaitOne(5000) }
        catch [Threading.AbandonedMutexException] { throw 'Previous control action was interrupted; inspect its state before configuring' }
        if (-not $controlMutexOwned) { throw 'Another service control action is in progress' }
        if (Test-Path -LiteralPath $configPath) { throw "Configuration already exists: $configPath; preserve it and its service state" }
        if (@(Get-Service38Listener $port).Count) { throw 'Port 1919 is occupied; configuration cannot replace an active service' }
        $profile = Get-Content -LiteralPath (Join-Path $labRoot 'profile.json') -Raw | ConvertFrom-Json
        $programFilesX86 = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
        $candidate = [pscustomobject]@{
            schema = 3; runtime = $Runtime; backend_cpu_affinity = $BackendCpuAffinity;
            venv_root = $VenvRoot; model_path = $ModelPath;
            state_root = $(if ($StateRoot) { $StateRoot } else { Split-Path -Parent $configPath });
            msvc_root = $(if ($MsvcRoot) { $MsvcRoot } else {
                Join-Path $programFilesX86 ("Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\" + $profile.msvc_version) });
            windows_sdk_root = $(if ($WindowsSdkRoot) { $WindowsSdkRoot } else { Join-Path $programFilesX86 'Windows Kits\10' })
        }
        $config = Resolve-Service64kConfig $candidate
        $runtimeName = $config.runtime
        $venvRoot = $config.venv_root
        $null = Read-Service38WorkingRelease
        Assert-Service64kLocalPaths $config
        $guard = Get-Service64kGuard
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $configPath), $config.state_root | Out-Null
        Write-Service38Json $configPath $config
        [ordered]@{ configured = $true; config = $configPath; source_root = $source64k;
            state_root = $config.state_root; gpu_instance = $guard.gpu_instance; model_started = $false } |
            ConvertTo-Json -Depth 5
        return
    }
    if ($Action -eq 'SetCpuAffinity' -and -not $PSBoundParameters.ContainsKey('BackendCpuAffinity')) {
        throw 'SetCpuAffinity requires an explicit -BackendCpuAffinity windows or one_per_core'
    }
    if ($Action -notin @('Configure','SetCpuAffinity') -and $PSBoundParameters.ContainsKey('BackendCpuAffinity')) {
        throw 'BackendCpuAffinity is accepted only with Configure or SetCpuAffinity'
    }
    if ($VenvRoot -or $ModelPath -or $StateRoot -or $MsvcRoot -or $WindowsSdkRoot -or $PSBoundParameters.ContainsKey('Runtime')) {
        throw 'Local path parameters are accepted only with -Action Configure'
    }
    $config = Read-Service64kConfig $configPath
    $runtimeName = $config.runtime
    $venvRoot = $config.venv_root
    $stateRoot = $config.state_root
    New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
    $launcher = Join-Path $labRoot 'launch64k.ps1'
    $pointerFile = Join-Path $stateRoot 'active-service.json'
    $faultFile = Join-Path $stateRoot 'recovery-required.json'
    $stopRequestFile = Join-Path $stateRoot 'stop-request.json'
    $quarantine64k = Join-Path $labRoot 'quarantine-history.json'
    . (Join-Path $labRoot 'service64k_lifecycle.ps1')
    function Assert-ProfileStartAllowed {
        if (Test-Path -LiteralPath $faultFile) { throw "Recovery required; inspect $faultFile before another start" }
        if (Test-Path -LiteralPath $stopRequestFile) { throw "Pending stop request; inspect $stopRequestFile" }
        if ((Get-Listener).Count) { throw 'Port 1919 is occupied; refusing to replace its owner' }
        Assert-NoRecordedService
    }
    if ($Action -eq 'Stop') {
        Write-Service38Json $stopRequestFile ([ordered]@{ at = (Get-Date).ToString('o'); reason = 'controller stop' })
    }
    if ($Action -in @('Start','Stop','SetCpuAffinity','CaptureDiagnostics')) {
        $controlMutex = [Threading.Mutex]::new($false, 'Local\FreeTokenROCm-Service38-1919')
        $waitMs = if ($Action -eq 'Stop') { 300000 } else { 5000 }
        try { $controlMutexOwned = $controlMutex.WaitOne($waitMs) }
        catch [Threading.AbandonedMutexException] {
            $controlMutexOwned = $true
            Write-Service38Fault $faultFile "Previous service control process exited during a state change; inspect $pointerFile"
        }
        if (-not $controlMutexOwned) { throw 'Another service control action is in progress' }
    }
    switch ($Action) {
        'CaptureDiagnostics' {
            $state = Get-State
            if (-not $state) { throw 'No managed service is recorded' }
            Save-Service64kProgressCapture $state 'explicit diagnostic capture' | ConvertTo-Json -Depth 8
        }
        'SetCpuAffinity' {
            Set-Service64kCpuAffinity $BackendCpuAffinity | ConvertTo-Json -Depth 6
        }
        'Verify' {
            $release = Read-Service38WorkingRelease
            Assert-Service64kLocalPaths $config
            $guard = Get-Service64kGuard
            [ordered]@{ verified = $true; source_root = $source64k; venv_root = $venvRoot;
                runtime = $runtimeName; compiler_workaround = ($runtimeName -eq 'rocm1010');
                backend_cpu_affinity = $config.backend_cpu_affinity;
                cpu_affinity_probe = (Get-Service64kCpuAffinityProbe $config.backend_cpu_affinity);
                source_files = 476; packages = 74; native_files = 5;
                gpu_instance = $guard.gpu_instance; resource_guard = $guard;
                model_started = $false; repository_launcher_model_smoke_performed = ($release.repository_launcher_model_smoke_performed -eq $true) } |
                ConvertTo-Json -Depth 6
        }
        'Start' {
            Assert-ProfileStartAllowed
            Start-ManagedService (Get-ProfileLaunchArgs) 'Reviewed 64K service'
        }
        'Preflight' {
            Assert-ProfileStartAllowed
            $launchArgs = Get-ProfileLaunchArgs
            $launchArgs.PreflightOnly = $true
            & $launcher @launchArgs
            Write-Output "64K preflight evidence: $($launchArgs.RunDir)"
        }
        'Stop' {
            if (Test-Path -LiteralPath $pointerFile) { Stop-TrialService }
            elseif ((Get-Listener).Count) { throw 'Port 1919 is occupied without a recorded managed service' }
            else { Write-Output 'No managed service is recorded; port 1919 is free' }
            try { Remove-Item -LiteralPath $stopRequestFile -ErrorAction Stop }
            catch [System.Management.Automation.ItemNotFoundException] { }
        }
    'Status' {
        # Startup already leaves spawn-intent.json before Start-Process and service-state.json
        # after PID registration. Surface both markers without changing recovery behavior.
        $startupEvidence = [ordered]@{
            pointer_present = (Test-Path -LiteralPath $pointerFile -PathType Leaf)
            spawn_intent_present = $false
            state_file_present = $false
            recovery_identity_present = $false
        }
        if ($startupEvidence.pointer_present) {
            try {
                $auditPointer = Get-Content -LiteralPath $pointerFile -Raw | ConvertFrom-Json
                $auditStateFile = [string]$auditPointer.state_file
                if ($auditStateFile) {
                    $startupEvidence.state_file_present = Test-Path -LiteralPath $auditStateFile -PathType Leaf
                    $startupEvidence.spawn_intent_present = Test-Path -LiteralPath (
                        Join-Path (Split-Path -Parent $auditStateFile) 'spawn-intent.json') -PathType Leaf
                }
                $startupEvidence.recovery_identity_present = $null -ne $auditPointer.recovery_state
            } catch {
                $startupEvidence.read_error = $_.Exception.Message
            }
        }
        $state = $null
        $stateError = $null
        try { $state = Get-State }
        catch { $stateError = $_.Exception.Message }
        $owned = $false
        $observerOwned = $false
        $identityError = $null
        $remainingTree = @()
        if ($null -ne $state -and $state.status -in @('RUNNING', 'STARTING')) {
            try {
                $remainingTree = @(Get-Service38Tree $state)
                $owned = $null -ne (Get-OwnedRoot $state)
            } catch { $identityError = $_.Exception.Message }
        }
        if ($null -ne $state -and $state.observer_pid) {
            try { $observerOwned = $null -ne (Get-Service38Observer $state) }
            catch { $identityError = $_.Exception.Message }
        }
        $health = $null
        $listeners = @()
        try { $listeners = @(Get-Listener) }
        catch { $identityError = $_.Exception.Message }
        $listenerOwned = $false
        if ($listeners.Count -and $null -ne $state) {
            try { $listenerOwned = Assert-Service38PortOwner $state }
            catch { $identityError = $_.Exception.Message }
        }
        if ($listeners.Count -and $owned -and $listenerOwned) {
            try { $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 5 }
            catch { $health = @{ error = $_.Exception.Message } }
        }
        $stopRequested = Test-Path -LiteralPath $stopRequestFile
        $workingReleaseValid=$false; $workingReleaseError=$null; $repositoryModelSmoke=$false
        if (Test-Path -LiteralPath (Join-Path $labRoot 'reviewed-release.json')) {
            try {
                $checkedRelease=Read-Service38WorkingRelease; $workingReleaseValid=$true
                $repositoryModelSmoke=($checkedRelease.repository_launcher_model_smoke_performed -eq $true)
            }
            catch { $workingReleaseError=$_.Exception.Message }
        }
        $recoveryRequired = (Test-Path -LiteralPath $faultFile) -or $stopRequested
        $status = if ($recoveryRequired -or $stateError -or $identityError -or
            ($state -and $state.status -eq 'RUNNING' -and $health -and $health.status -ne 'ok') -or
            ($state -and $state.status -eq 'RUNNING' -and -not $listenerOwned) -or
            ($state -and $state.observer_pid -and $owned -and -not $observerOwned) -or
            ($observerOwned -and -not $owned) -or
            (-not $owned -and ($remainingTree.Count -or $listeners.Count -or
                ($state -and $state.status -in @('RUNNING', 'STARTING'))))) {
            'RECOVERY_REQUIRED'
        } elseif ($owned -and $state.status -eq 'STARTING') { 'STARTING'
        } elseif ($owned) { 'RUNNING' } else { 'STOPPED' }
        [ordered]@{ status = $status; root_owned = $owned; observer_owned = $observerOwned;
            runtime = $runtimeName; compiler_workaround = ($runtimeName -eq 'rocm1010');
            configured_backend_cpu_affinity = $config.backend_cpu_affinity;
            listener_owned = $listenerOwned;
            process_tree_count = $remainingTree.Count;
            listeners = $listeners.Count; recovery_required = $recoveryRequired;
            ctx64k_quarantined = ((Test-Path -LiteralPath $quarantine64k) -and -not $workingReleaseValid);
            ctx64k_quarantine_history_preserved = (Test-Path -LiteralPath $quarantine64k);
            ctx64k_working_release_valid = $workingReleaseValid; ctx64k_working_release_error = $workingReleaseError;
            repository_launcher_model_smoke_performed = $repositoryModelSmoke;
            stop_requested = $stopRequested;
            state_error = $stateError; identity_error = $identityError;
            health = $health; startup_evidence = $startupEvidence; state = $state } |
            ConvertTo-Json -Depth 8
    }
        'Health' {
            $state = Get-State
            if ($null -eq (Get-OwnedRoot $state) -or -not (Assert-Service38PortOwner $state)) {
                throw 'No running service with a verified listener owner'
            }
            Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 10 | ConvertTo-Json
        }
    }
} finally {
    if ($controlMutex) {
        if ($controlMutexOwned) { $controlMutex.ReleaseMutex() }
        $controlMutex.Dispose()
    }
}
