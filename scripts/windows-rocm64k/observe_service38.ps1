param(
    [Parameter(Mandatory)][string]$StateFile,
    [Parameter(Mandatory)][string]$GuardFile,
    [Parameter(Mandatory)][string]$FaultFile,
    [Parameter(Mandatory)][string]$EventsFile,
    [Parameter(Mandatory)][string]$ReadyFile
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'service38_safety.ps1')
$state = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
$guard = Get-Content -LiteralPath $GuardFile -Raw | ConvertFrom-Json
$lastLogLength = @{}
$pendingLogLine = @{}
$pendingLogOffset = @{}
$fatalPattern = Get-Service38FatalLogPattern
$eventPattern = 'Prefill batch|\bNaN\b|\bInf\b|' + $fatalPattern
$lastMemorySample = [datetime]::MinValue
$lastHeartbeat = [datetime]::MinValue
$lastProbeStart = [datetime]::MinValue
$treeProbe = $null
$resourceProbe = $null
$tree = @()
$treeSampled = $false
$resources = $null
$progressWatch = [pscustomobject]@{sequence=$null;clock=[Diagnostics.Stopwatch]::StartNew()}
function Record-ProcessMemory($Tree) {
    $ownedIds = @($Tree | ForEach-Object { [int]$_.ProcessId })
    $processes = @(Get-Process -ErrorAction SilentlyContinue)
    $owned = @($processes | Where-Object { $ownedIds -contains $_.Id } | ForEach-Object {
        [pscustomobject]@{ pid = $_.Id; name = $_.ProcessName;
            started = $_.StartTime.ToUniversalTime().ToString('o');
            private_bytes = $_.PrivateMemorySize64; working_set_bytes = $_.WorkingSet64 }
    })
    $others = @($processes | Where-Object { $ownedIds -notcontains $_.Id } |
        Group-Object ProcessName | ForEach-Object {
            [pscustomobject]@{ name = $_.Name; count = $_.Count;
                private_bytes = ($_.Group | Measure-Object PrivateMemorySize64 -Sum).Sum;
                working_set_bytes = ($_.Group | Measure-Object WorkingSet64 -Sum).Sum }
        } | Sort-Object private_bytes -Descending | Select-Object -First 15)
    [ordered]@{ kind = 'process_memory'; at = (Get-Date).ToString('o'); owned = $owned;
        engine_private_bytes = ($owned | Measure-Object private_bytes -Sum).Sum;
        other_processes_private_bytes = ($processes | Where-Object { $ownedIds -notcontains $_.Id } |
            Measure-Object PrivateMemorySize64 -Sum).Sum; top_other_groups = $others } |
        ConvertTo-Json -Compress -Depth 6 | Add-Content -LiteralPath $EventsFile
}
function Record-ServerLine([string]$Line, [string]$LogName, [long]$LineOffset) {
    if ($Line -match $eventPattern) {
        [ordered]@{ kind = 'server_log'; at = (Get-Date).ToString('o'); line = $Line.Trim() } |
            ConvertTo-Json -Compress | Add-Content -LiteralPath $EventsFile
    }
    if (Test-Service38FatalLine $Line (Split-Path -Parent $StateFile) $LogName $LineOffset) {
        throw "Fatal server event: $Line"
    }
}
try {
while ($true) {
    try {
        # Drain logs before checking provider results, including their failures.
        # A pending GPU or WMI query never blocks the loop below.
        foreach ($logName in @('stdout.log', 'stderr.log')) {
            $log = Join-Path (Split-Path $StateFile) "server-logs\$logName"
            if (-not (Test-Path -LiteralPath $log)) { continue }
            $info = Get-Item -LiteralPath $log
            $offset = [long]($lastLogLength[$logName] ?? 0)
            if ($info.Length -lt $offset) {
                $offset = 0
                $pendingLogLine[$logName] = ''
                $pendingLogOffset[$logName] = 0L
            }
            if ($info.Length -gt $offset) {
                $stream = [IO.File]::Open($log, 'Open', 'Read', 'ReadWrite')
                try {
                    $null = $stream.Seek($offset, 'Begin')
                    $reader = [IO.StreamReader]::new($stream)
                    $fresh = $reader.ReadToEnd()
                    $lastLogLength[$logName] = $stream.Position
                } finally { $reader.Dispose() }
                $lineOffset = $offset - [Text.Encoding]::UTF8.GetByteCount(($pendingLogLine[$logName] ?? ''))
                $lines = (($pendingLogLine[$logName] ?? '') + $fresh) -split "`n"
                $pendingLogLine[$logName] = $lines[-1]
                for ($i = 0; $i -lt $lines.Count - 1; $i++) {
                    Record-ServerLine $lines[$i] $logName $lineOffset
                    $lineOffset += [Text.Encoding]::UTF8.GetByteCount($lines[$i]) + 1
                }
                $pendingLogOffset[$logName] = $lineOffset
            }
        }
        $now = Get-Date
        if (($now - $lastHeartbeat).TotalSeconds -ge 3) {
            [ordered]@{ kind = 'heartbeat'; at = $now.ToString('o'); memory = Get-Service38NativeMemory;
                tree_probe_pending = ($null -ne $treeProbe); resource_probe_pending = ($null -ne $resourceProbe) } |
                ConvertTo-Json -Compress -Depth 5 | Add-Content -LiteralPath $EventsFile
            $lastHeartbeat = $now
        }
        if (($now - $lastMemorySample).TotalSeconds -ge 15) {
            Record-ProcessMemory $tree
            $lastMemorySample = $now
        }
        $root = Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue
        if ($root -and $root.StartTime.ToUniversalTime().ToFileTimeUtc() -ne [long]$state.root_start_filetime) {
            throw "PID $($state.pid) no longer matches the recorded service identity"
        }
        if ($null -eq $root) {
            foreach ($name in $pendingLogLine.Keys) {
                if ($pendingLogLine[$name]) { Record-ServerLine $pendingLogLine[$name] $name $pendingLogOffset[$name] }
            }
            $stopIntent = Join-Path (Split-Path -Parent $StateFile) 'stop-intent.json'
            if (-not (Test-Path -LiteralPath $stopIntent)) {
                throw 'Service tree exited without a managed stop; recovery review required'
            }
            break
        }
        if ($state.progress_diagnostics -eq $true -and -not (Test-Path -LiteralPath (
            Join-Path (Split-Path -Parent $StateFile) 'stop-intent.json'))) {
            if (-not $state.cpu_affinity_receipt) {
                $acceptedState = Read-Service64kProgressFile $StateFile
                if ($acceptedState.pid -ne $state.pid -or
                    $acceptedState.root_start_filetime -ne $state.root_start_filetime -or
                    $acceptedState.run_dir -cne $state.run_dir) { throw 'Diagnostic state changed its root identity' }
                if ($acceptedState.cpu_affinity_receipt) { $state.cpu_affinity_receipt = $acceptedState.cpu_affinity_receipt }
            }
            $progress = Read-Service64kBackendProgress $state
            if (Test-Service64kProgressStall $progress $progressWatch) {
                throw "Backend made no stage progress for 45s: $($progress.phase)"
            }
        }
        if ($treeProbe) {
            $sample = Receive-Service38Probe $treeProbe
            if (-not $sample.pending) {
                $treeProbe = $null
                $tree = @(Select-Service38Tree $state @($sample.value))
                $treeSampled = $true
                if (-not $tree.Count) {
                    if (Get-Process -Id ([int]$state.pid) -ErrorAction SilentlyContinue) {
                        throw 'Live service root is absent from the process provider snapshot'
                    }
                    continue # root exited during the snapshot; inspect final logs next iteration
                }
            }
        }
        if ($resourceProbe) {
            $sample = Receive-Service38Probe $resourceProbe
            if (-not $sample.pending) {
                $resourceProbe = $null
                $resources = Assert-Service38Resources $guard 'runtime' $sample.value
                [ordered]@{ kind = 'resources'; at = (Get-Date).ToString('o'); resources = $resources } |
                    ConvertTo-Json -Compress -Depth 5 | Add-Content -LiteralPath $EventsFile
            }
        }
        if ($treeSampled -and $tree.Count -and $resources -and -not (Test-Path -LiteralPath $ReadyFile)) {
            Write-Service38Json $ReadyFile ([ordered]@{
                at = (Get-Date).ToString('o'); root_pid = $state.pid; resources = $resources })
        }
        if (-not $treeProbe -and -not $resourceProbe -and ($now - $lastProbeStart).TotalSeconds -ge 3) {
            $treeProbe = Start-Service38Probe 'processes' @{}
            $resourceProbe = Start-Service38Probe 'resources' $guard
            $lastProbeStart = Get-Date
        }
    } catch {
        Write-Service38Fault $FaultFile $_.Exception.Message $StateFile
        # Close probes first; do not consult the failed provider for diagnostics.
        Close-Service38Probe $treeProbe
        Close-Service38Probe $resourceProbe
        # Critical resource/provider cleanup never waits for a Python-thread reply.
        $capture = Save-Service64kProgressCapture $state 'observer fault cleanup' -WaitMilliseconds 0
        if ($capture.enabled) {
            [ordered]@{kind='progress_capture';at=(Get-Date).ToString('o');capture=$capture} |
                ConvertTo-Json -Compress -Depth 8 | Add-Content -LiteralPath $EventsFile
        }
        try { Record-ProcessMemory $tree } catch {
            [ordered]@{ kind = 'memory_sample_error'; message = $_.Exception.Message } |
                ConvertTo-Json -Compress | Add-Content -LiteralPath $EventsFile
        }
        Write-Service38StopIntent (Split-Path -Parent $StateFile) 'observer fault cleanup'
        try { Stop-Service38Tree $state } catch {
            [ordered]@{ kind = 'cleanup_error'; at = (Get-Date).ToString('o'); message = $_.Exception.Message } |
                ConvertTo-Json -Compress | Add-Content -LiteralPath $EventsFile
        }
        break
    }
    Start-Sleep -Milliseconds 250
}
} finally {
    Close-Service38Probe $treeProbe
    Close-Service38Probe $resourceProbe
}
