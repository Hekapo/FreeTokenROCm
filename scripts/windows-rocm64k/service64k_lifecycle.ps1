function Get-Listener {
    @(Get-Service38Listener $port)
}

function Get-State {
    if (-not (Test-Path -LiteralPath $pointerFile -PathType Leaf)) { return $null }
    $pointer = Get-Content -LiteralPath $pointerFile -Raw | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath $pointer.state_file -PathType Leaf)) {
        if ($pointer.recovery_state) { return $pointer.recovery_state }
        throw "service state file missing without recorded PID/time: $($pointer.state_file)"
    }
    try { return Get-Content -LiteralPath $pointer.state_file -Raw | ConvertFrom-Json }
    catch {
        if ($pointer.recovery_state) { return $pointer.recovery_state }
        throw "service state is unreadable without recorded PID/time: $($pointer.state_file)"
    }
}

function Get-OwnedRoot($state) {
    if ($null -eq $state -or $state.status -notin @('RUNNING', 'STARTING')) { return $null }
    $rootProcess = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
    if ($null -eq $rootProcess) { return $null }
    $identity = Get-Service38ProcessSnapshot -Filter "ProcessId=$($state.pid)" | Select-Object -First 1
    $created = $rootProcess.StartTime.ToUniversalTime().ToFileTimeUtc()
    if ($created -ne [long]$state.root_start_filetime -or
        $identity.ExecutablePath -ine $state.python -or
        -not (Test-Service38CommandLine $identity.CommandLine ([int]$state.port))) {
        throw "PID $($state.pid) does not match the recorded managed service identity"
    }
    return $rootProcess
}

function Set-StoppedState($state) {
    $pointer = Get-Content -LiteralPath $pointerFile -Raw | ConvertFrom-Json
    $state.status = 'STOPPED'
    $state | Add-Member -NotePropertyName stopped_at -NotePropertyValue (Get-Date).ToString('o') -Force
    Write-Service38Json $pointer.state_file $state
    $pointer | Add-Member -NotePropertyName recovery_state -NotePropertyValue $state -Force
    Write-Service38Json $pointerFile $pointer
}

function Stop-TrialService {
    $state = Get-State
    if ($null -eq $state) { throw 'No Triton 3.8 service state is recorded' }
    $activePointer = Get-Content -LiteralPath $pointerFile -Raw | ConvertFrom-Json
    if ($state.status -eq 'STOPPED') {
        try {
            if (@(Get-Service38Tree $state).Count) {
                Write-Service38Fault $faultFile 'Service tree remained after STOPPED state' $activePointer.state_file
                Write-Service38StopIntent (Split-Path -Parent $activePointer.state_file) 'repeat cleanup'
                Stop-Service38Tree $state
            }
            if ((Get-Listener).Count) { throw 'Port 1919 is occupied while the recorded service is stopped' }
            Stop-Service38Observer $state
        } catch {
            Write-Service38Fault $faultFile "repeat cleanup failed: $($_.Exception.Message)" $activePointer.state_file
            throw
        }
        Write-Output 'Managed service is already stopped'
        return
    }
    $capture = Save-Service64kProgressCapture $state 'controller stop'
    if ($capture.enabled) {
        if ($capture.directory) { Write-Output ("Backend diagnostics saved: " + $capture.directory) }
        else { Write-Output ("Backend diagnostics unavailable: " + $capture.error) }
    }
    $treeBeforeStop = @(Get-Service38Tree $state)
    if (-not $treeBeforeStop.Count -and $state.status -eq 'RUNNING') {
        Write-Service38Fault $faultFile 'Running service exited before managed stop' $activePointer.state_file
    }
    if ($state.observer_pid -and -not (Get-Service38Observer $state)) {
        Write-Service38Fault $faultFile 'Resource observer exited before managed stop' $activePointer.state_file
    }
    Write-Service38StopIntent (Split-Path -Parent $activePointer.state_file) 'controller stop'
    try { Stop-Service38Tree $state }
    catch {
        Write-Service38Fault $faultFile "cleanup failed: $($_.Exception.Message)" $activePointer.state_file
        throw
    }
    try { Stop-Service38Observer $state }
    catch {
        Write-Service38Fault $faultFile "observer cleanup failed: $($_.Exception.Message)" $activePointer.state_file
        throw
    }
    $fatal = @(Get-Service38FatalLogEvents (Split-Path -Parent $activePointer.state_file) | Select-Object -First 1)
    if ($fatal.Count) {
        Write-Service38Fault $faultFile "fatal server event: $($fatal[0].Line)" $activePointer.state_file
    }
    if ($state.resource_guard) {
        try {
            $guard = Get-Content -LiteralPath $state.resource_guard -Raw | ConvertFrom-Json
            Wait-Service38ResourceRecovery $guard (Join-Path (Split-Path -Parent $activePointer.state_file) 'resources-after-stop.json')
        } catch {
            Write-Service38Fault $faultFile "resource recovery failed: $($_.Exception.Message)" $activePointer.state_file
            throw
        }
    }
    Set-StoppedState $state
    Write-Output "managed service stopped; port $port is free"
}

function Assert-NoRecordedService {
    $oldState = Get-State
    if ($null -eq $oldState) { return }
    if (@(Get-Service38Tree $oldState).Count) {
        throw 'Managed service or its descendants are already running'
    }
    if (Get-Service38Observer $oldState) { throw 'Managed observer is still running' }
    if ($oldState.status -in @('RUNNING', 'STARTING')) {
        throw 'Recorded service ended without managed stop; recovery review required'
    }
}

function Start-ManagedService([hashtable]$LauncherArgs, [string]$Description) {
    if (Test-Path -LiteralPath $faultFile) { throw "Recovery required; inspect $faultFile before another start" }
    if (Test-Path -LiteralPath $stopRequestFile) { throw "Pending stop request; inspect $stopRequestFile before another start" }
    $LauncherArgs.PointerFile = $pointerFile
    $stateFile = Join-Path $LauncherArgs.RunDir 'service-state.json'
    $spawnIntent = Join-Path $LauncherArgs.RunDir 'spawn-intent.json'
    Write-Service38Json $pointerFile ([ordered]@{
        state_file = $stateFile; started_at = (Get-Date).ToString('o') })
    try {
        & $launcher @LauncherArgs
        if (-not (Test-Path -LiteralPath $stateFile -PathType Leaf)) {
            throw "$Description did not reach ready state; inspect $($LauncherArgs.RunDir)"
        }
        $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
        if ($state.status -ne 'RUNNING' -or $null -eq (Get-OwnedRoot $state)) {
            throw "$Description root exited during startup"
        }
        if (-not (Assert-Service38PortOwner $state)) {
            throw "$Description has no listener owned by its process tree"
        }
        if ($state.observer_pid -and -not (Get-Service38Observer $state)) {
            throw "$Description resource observer exited during startup"
        }
        if (Test-Path -LiteralPath $stopRequestFile) { throw 'Managed stop was requested during startup' }
        if (Test-Path -LiteralPath $faultFile) { throw "Recovery required; inspect $faultFile" }
    } catch {
        $cause = $_
        if (-not (Test-Path -LiteralPath $stateFile) -and
            -not (Test-Path -LiteralPath $spawnIntent) -and
            -not (Get-Listener).Count) {
            # The launcher never reached Start-Process, so there is no owned PID to recover.
            Remove-Item -LiteralPath $pointerFile -ErrorAction SilentlyContinue
            throw $cause
        }
        try {
            $failedState = Get-State
            if ($failedState -and $failedState.status -ne 'STOPPED') {
                Stop-TrialService
            } elseif (-not $failedState -and -not (Get-Listener).Count) {
                Remove-Item -LiteralPath $pointerFile -ErrorAction SilentlyContinue
            } elseif ((Get-Listener).Count) {
                throw 'Port is occupied after launcher failure without an active recorded service'
            }
        } catch {
            Write-Service38Fault $faultFile "startup cleanup failed: $($_.Exception.Message)" $stateFile
            throw "${Description} failed and requires recovery; original error: $($cause.Exception.Message); cleanup error: $($_.Exception.Message)"
        }
        throw $cause
    }
    Write-Service38Json $pointerFile ([ordered]@{
        state_file = $stateFile; started_at = (Get-Date).ToString('o'); recovery_state = $state })
    Write-Output "$Description active at http://127.0.0.1:$port; run=$($LauncherArgs.RunDir)"
}
