# Each system-provider query lives in its own hidden process. Its Windows job
# closes on normal disposal OR parent termination, so stuck probes cannot leak.
$script:service38ProbeRoot = $PSScriptRoot
function Initialize-Service38Native {
    if (-not ('Service38Native' -as [type])) {
        Add-Type -Path (Join-Path $script:service38ProbeRoot 'service38_probe_native.cs')
    }
}

function Start-Service38Probe {
    param([ValidateSet('processes', 'resources', 'listeners', 'gpu-identity')][string]$Kind, $Payload,
        [ValidateRange(100, 30000)][int]$TimeoutMilliseconds = 5000,
        [string]$WorkerPath = (Join-Path $script:service38ProbeRoot 'query_service38.ps1'))
    Initialize-Service38Native
    $info = [Diagnostics.ProcessStartInfo]::new((Join-Path $PSHOME 'pwsh.exe'))
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($Payload | ConvertTo-Json -Compress -Depth 7)))
    foreach ($arg in @('-NoProfile', '-NonInteractive', '-File', $WorkerPath, '-Kind', $Kind, '-PayloadBase64', $encoded)) {
        $info.ArgumentList.Add($arg)
    }
    $job = [Service38Native]::CreateProbeJob()
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    $started = $false
    try {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $null = $process.Start()
        $started = $true
        [Service38Native]::Attach($job, $process.Handle)
        $process.StandardInput.WriteLine('probe-ready')
        $process.StandardInput.Close()
        return [pscustomobject]@{ kind = $Kind; process = $process; job = $job; clock = $clock;
            timeout_ms = $TimeoutMilliseconds; output = $process.StandardOutput.ReadToEndAsync();
            errors = $process.StandardError.ReadToEndAsync(); closed = $false }
    } catch {
        $job.Dispose()
        if ($started -and -not $process.HasExited) { $process.Kill() }
        $process.Dispose()
        throw
    }
}

function Close-Service38Probe($Probe) {
    if ($null -eq $Probe -or $Probe.closed) { return }
    $Probe.closed = $true
    $Probe.job.Dispose() # Kills only this probe and its children; no PID lookup.
    $Probe.process.Dispose()
}

function Receive-Service38Probe($Probe) {
    if ($Probe.closed) { throw "Probe $($Probe.kind) was already closed" }
    $Probe.process.Refresh()
    if ($Probe.clock.ElapsedMilliseconds -ge $Probe.timeout_ms) {
        $elapsed = $Probe.clock.ElapsedMilliseconds
        Close-Service38Probe $Probe
        throw "System probe '$($Probe.kind)' timed out after ${elapsed}ms (limit $($Probe.timeout_ms)ms)"
    }
    # Pipes must also reach EOF: an orphaned child retaining stdout is a timeout.
    if ($Probe.process.HasExited -and $Probe.output.IsCompleted -and $Probe.errors.IsCompleted) {
        try {
            if ($Probe.process.ExitCode -ne 0) {
                throw "Probe $($Probe.kind) exited $($Probe.process.ExitCode): $($Probe.errors.GetAwaiter().GetResult())"
            }
            $result = $Probe.output.GetAwaiter().GetResult() | ConvertFrom-Json -ErrorAction Stop
            if ($null -eq $result -or $result.ok -isnot [bool] -or
                ($result.ok -and ($null -eq $result.PSObject.Properties['value'] -or
                    ($Probe.kind -eq 'resources' -and $null -eq $result.value)))) {
                throw "Probe $($Probe.kind) returned an invalid envelope"
            }
            if (-not $result.ok) { throw "Probe $($Probe.kind) failed: $($result.error)" }
            return [pscustomobject]@{ pending = $false; value = $result.value }
        } finally { Close-Service38Probe $Probe }
    }
    return [pscustomobject]@{ pending = $true }
}

function Invoke-Service38Probe([string]$Kind, $Payload) {
    $probe = Start-Service38Probe $Kind $Payload
    try {
        do {
            $result = Receive-Service38Probe $probe
            if ($result.pending) { Start-Sleep -Milliseconds 50 }
        } while ($result.pending)
        return $result.value
    } finally { Close-Service38Probe $probe }
}

function Get-Service38ProcessSnapshot([string]$Filter = '') {
    @(Invoke-Service38Probe 'processes' @{ filter = $Filter })
}

function Get-Service38NativeMemory {
    Initialize-Service38Native
    $memory = [Service38Native+MemoryStatus]::new()
    if (-not [Service38Native]::GlobalMemoryStatusEx($memory)) {
        throw [ComponentModel.Win32Exception]::new([Runtime.InteropServices.Marshal]::GetLastWin32Error())
    }
    [ordered]@{ free_physical_gib = [math]::Round($memory.AvailablePhysical / 1GB, 3);
        available_page_file_gib = [math]::Round($memory.AvailablePageFile / 1GB, 3) }
}
