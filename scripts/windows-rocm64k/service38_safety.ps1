# Shared by the launcher, controller and external observer. No GPU runtime is imported here.
. (Join-Path $PSScriptRoot 'service38_probes.ps1')
function Write-Service38Json([string]$Path, $Value) {
    $directory = Split-Path -Parent $Path
    $temporary = Join-Path $directory ('.' + (Split-Path -Leaf $Path) + '.' + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        $json = $Value | ConvertTo-Json -Depth 8
        [IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        [IO.File]::Move($temporary, $Path, $true)
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -ErrorAction SilentlyContinue
        }
    }
}

function Write-Service38Fault([string]$Path, [string]$Reason, [string]$StateFile = '') {
    # Preserve the first failure; a later cleanup error must not erase its cause.
    $directory = Split-Path -Parent $Path
    $temporary = Join-Path $directory ('.' + (Split-Path -Leaf $Path) + '.' + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        $fault = [ordered]@{
            at = (Get-Date).ToString('o'); reason = $Reason; state_file = $StateFile }
        $json = $fault | ConvertTo-Json -Depth 8
        [IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        try { [IO.File]::Move($temporary, $Path) }
        catch [IO.IOException] {
            if (-not (Test-Path -LiteralPath $Path)) { throw }
        }
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -ErrorAction SilentlyContinue
        }
    }
}

function Write-Service38StopIntent([string]$RunDir, [string]$Reason) {
    $offsets = [ordered]@{}
    foreach ($name in @('stdout.log', 'stderr.log')) {
        $path = Join-Path $RunDir "server-logs\$name"
        $offsets[$name] = if (Test-Path -LiteralPath $path) { (Get-Item -LiteralPath $path).Length } else { 0L }
    }
    Write-Service38Json (Join-Path $RunDir 'stop-intent.json') ([ordered]@{
        at = (Get-Date).ToString('o'); reason = $Reason; log_offsets = $offsets })
}

function Test-Service38FatalLine([string]$Line, [string]$RunDir, [string]$LogName, [long]$LineOffset) {
    $pattern = Get-Service38FatalLogPattern
    if ($Line -notmatch $pattern) { return $false }
    $workerExit = 'Backend supervisor: .*exited'
    if ($Line -notmatch $workerExit -or $Line -match $pattern.Replace("$workerExit|", '')) { return $true }
    $intentPath = Join-Path $RunDir 'stop-intent.json'
    if (-not (Test-Path -LiteralPath $intentPath)) { return $true }
    $intent = Get-Content -LiteralPath $intentPath -Raw | ConvertFrom-Json
    # Keep failures from before Stop. Only worker-exit messages appended after
    # the recorded boundary are expected while killing the owned process tree.
    if ($null -eq $intent.log_offsets) { return $true }
    $property = $intent.log_offsets.PSObject.Properties[$LogName]
    return ($null -eq $property -or $LineOffset -lt [long]$property.Value)
}

function Get-Service38FatalLogEvents([string]$RunDir) {
    foreach ($name in @('stdout.log', 'stderr.log')) {
        $path = Join-Path $RunDir "server-logs\$name"
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $stream = [IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
        try {
            $reader = [IO.StreamReader]::new($stream)
            $text = $reader.ReadToEnd()
        } finally { $reader.Dispose() }
        $offset = 0L
        foreach ($line in ($text -split "`n")) {
            if (Test-Service38FatalLine $line $RunDir $name $offset) {
                [pscustomobject]@{ Line = $line; LogName = $name; Offset = $offset }
            }
            $offset += [Text.Encoding]::UTF8.GetByteCount($line) + 1
        }
    }
}

function Get-Service38Observer($State) {
    if (-not $State.observer_pid) { return $null }
    $observerId = [int]$State.observer_pid
    $observerStarted = [long]$State.observer_start_filetime
    $observer = Get-Process -Id $observerId -ErrorAction SilentlyContinue
    if ($null -eq $observer -or $observer.StartTime.ToUniversalTime().ToFileTimeUtc() -ne $observerStarted) { return $null }
    return $observer
}

function Stop-Service38Observer($State) {
    $observer = Get-Service38Observer $State
    if ($null -eq $observer) { return }
    $observerId = $observer.Id
    $observerStarted = [long]$State.observer_start_filetime
    try { Stop-Process -Id $observerId -Force -ErrorAction Stop }
    catch {
        if (Get-Service38Observer $State) { throw }
        return
    }
    $deadline = (Get-Date).AddSeconds(10)
    do {
        $observer = Get-Process -Id $observerId -ErrorAction SilentlyContinue
        if ($null -eq $observer -or $observer.StartTime.ToUniversalTime().ToFileTimeUtc() -ne $observerStarted) {
            return
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    throw "observer PID $observerId remained after managed stop"
}

function Get-Service38Listener([int]$Port) {
    @(Invoke-Service38Probe 'listeners' @{ port = $Port })
}

function Test-Service38CommandLine([string]$CommandLine, [int]$Port) {
    $portPattern = '(?:^|\s)--port\s+' + [regex]::Escape([string]$Port) + '(?:\s|$)'
    return ($CommandLine -match 'freetoken\.cli|stats_launcher\.py') -and
        ($CommandLine -match $portPattern)
}

function Get-Service38FatalLogPattern {
    'non-finite logits|HIP error|hipError|device error|device-side assert|GPU fault|\bTDR\b|out of memory|\bOOM\b|Backend supervisor: .*exited|Exception Code: 0xC0000005'
}

function Get-Service38Resources($Guard) {
    Invoke-Service38Probe 'resources' $Guard
}

function Assert-Service38Resources($Guard, [string]$Phase, $Reading = $null) {
    if ($PSBoundParameters.ContainsKey('Reading') -and $null -eq $Reading) {
        throw "Resource guard ($Phase) has an invalid null measurement"
    }
    $reading = if ($null -eq $Reading) { Get-Service38Resources $Guard } else { $Reading }
    foreach ($field in @('free_physical_gib', 'free_commit_gib', 'gpu_local_used_gib', 'gpu_local_free_gib')) {
        if ($null -eq $reading.$field -or -not [double]::IsFinite([double]$reading.$field)) {
            throw "Resource guard ($Phase) has an invalid measurement: $field"
        }
    }
    $limits = $Guard.$Phase
    if ($reading.free_physical_gib -lt $limits.free_physical_gib -or
        $reading.free_commit_gib -lt $limits.free_commit_gib -or
        $reading.gpu_local_free_gib -lt $limits.gpu_local_free_gib) {
        throw "Resource guard ($Phase) failed: $($reading | ConvertTo-Json -Compress); minimum=$($limits | ConvertTo-Json -Compress)"
    }
    return $reading
}

function Get-Service38RecoveryGuard($Guard) {
    $effectiveGuard = $Guard | ConvertTo-Json -Depth 6 | ConvertFrom-Json
    if ($effectiveGuard.PSObject.Properties.Name -contains 'recovery') {
        if ($null -eq $effectiveGuard.recovery) { throw 'Resource recovery limits are missing' }
        $effectiveGuard.startup = $effectiveGuard.recovery
    }
    return $effectiveGuard
}

function Wait-Service38ResourceRecovery($Guard, [string]$EvidenceFile) {
    $recoveryGuard = Get-Service38RecoveryGuard $Guard
    $deadline = (Get-Date).AddSeconds(30)
    do {
        try {
            $reading = Assert-Service38Resources $recoveryGuard 'startup'
            Write-Service38Json $EvidenceFile $reading
            return
        } catch {
            $lastError = $_.Exception.Message
            if ((Get-Date) -lt $deadline) { Start-Sleep -Seconds 2 }
        }
    } while ((Get-Date) -lt $deadline)
    throw "Resources did not return to the recovery guard after stop: $lastError"
}

function Get-Service38Tree($State) {
    # Avoid any provider call for an already absent or reused root PID.
    $root = Get-Process -Id ([int]$State.pid) -ErrorAction SilentlyContinue
    if ($null -eq $root) { return @() }
    if ($root.StartTime.ToUniversalTime().ToFileTimeUtc() -ne [long]$State.root_start_filetime) {
        throw "PID $($State.pid) no longer matches the recorded service identity"
    }
    Select-Service38Tree $State @(Get-Service38ProcessSnapshot)
}

function Select-Service38Tree($State, $All) {
    $rootId = [int]$State.pid
    # ConvertFrom-Json may turn an ISO timestamp into DateTime; reparsing its
    # culture-formatted string fails under a different Windows locale.
    $rootStarted = [datetime]::FromFileTimeUtc([long]$State.root_start_filetime)
    $all = @($All)
    $root = @($all | Where-Object { $_.ProcessId -eq $rootId })
    if (-not $root.Count) {
        # Parent PID alone cannot prove ownership after the root exits: Windows may reuse it.
        # A vanished root is handled as recovery-required by the caller; never kill guessed children.
        return @()
    }
    $process = Get-Process -Id $rootId -ErrorAction SilentlyContinue
    # The root can exit between the CIM snapshot and the live process lookup.
    # Report an absent tree so the caller can distinguish managed Stop from an
    # unexpected exit and still inspect final logs. Never infer child ownership.
    if ($null -eq $process) { return @() }
    $created = $process.StartTime.ToUniversalTime().ToFileTimeUtc()
    if ($created -ne [long]$State.root_start_filetime -or
        $root[0].ExecutablePath -ine $State.python -or
        -not (Test-Service38CommandLine $root[0].CommandLine ([int]$State.port))) {
        throw "PID $rootId no longer matches the recorded service identity"
    }
    $tree = @($root)
    $frontier = @($rootId)
    while ($frontier.Count) {
        $next = @($all | Where-Object { $frontier -contains $_.ParentProcessId -and
            $_.CreationDate.ToUniversalTime() -ge $rootStarted })
        $tree += $next
        $frontier = @($next | ForEach-Object { [int]$_.ProcessId })
    }
    return $tree
}

function Assert-Service38PortOwner($State) {
    $listeners = @(Get-Service38Listener ([int]$State.port))
    if (-not $listeners.Count) { return $false }
    $tree = @(Get-Service38Tree $State)
    $ownedIds = @($tree | ForEach-Object { [int]$_.ProcessId })
    foreach ($listener in $listeners) {
        if ($ownedIds -notcontains [int]$listener.OwningProcess) {
            throw "Port $($State.port) is listening under foreign PID $($listener.OwningProcess)"
        }
    }
    return $true
}

function Stop-Service38Tree($State) {
    $tree = @(Get-Service38Tree $State)
    if (-not $tree.Count -and $State.status -in @('RUNNING', 'STARTING')) {
        throw 'Managed root is absent; descendants cannot be identified safely after PID reuse'
    }
    [array]::Reverse($tree)
    foreach ($item in $tree) {
        $current = Get-Service38ProcessSnapshot -Filter "ProcessId=$($item.ProcessId)" | Select-Object -First 1
        if ($null -ne $current -and $current.CreationDate -eq $item.CreationDate) {
            try { Stop-Process -Id $item.ProcessId -Force -ErrorAction Stop }
            catch {
                $again = Get-Service38ProcessSnapshot -Filter "ProcessId=$($item.ProcessId)" | Select-Object -First 1
                if ($again -and $again.CreationDate -eq $item.CreationDate) { throw }
            }
        }
    }
    $deadline = (Get-Date).AddSeconds(60)
    do {
        $snapshot = @(Get-Service38ProcessSnapshot)
        $remaining = @($tree | Where-Object {
            $item = $_
            $p = $snapshot | Where-Object { $_.ProcessId -eq $item.ProcessId } | Select-Object -First 1
            $null -ne $p -and $p.CreationDate -eq $_.CreationDate
        })
        if ($remaining.Count) { Start-Sleep -Seconds 2 }
    } while ($remaining.Count -and (Get-Date) -lt $deadline)
    $listeners = @(Get-Service38Listener ([int]$State.port))
    if ($remaining.Count -or $listeners.Count) {
        throw "service cleanup incomplete: processes=$($remaining.Count), listeners=$($listeners.Count)"
    }
}
