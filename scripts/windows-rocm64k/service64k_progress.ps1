function Read-Service64kProgressFile([string]$Path, [int]$Maximum = 65536) {
    $stream = [IO.File]::Open($Path, 'Open', 'Read', [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
    try {
        if ($stream.Length -gt $Maximum) { throw 'Diagnostic JSON exceeds its size bound' }
        $reader = [IO.StreamReader]::new($stream)
        return $reader.ReadToEnd() | ConvertFrom-Json -ErrorAction Stop
    } finally { $stream.Dispose() }
}

function Read-Service64kBackendProgress($State) {
    if ($State.progress_diagnostics -ne $true -or -not $State.cpu_affinity_receipt) { return $null }
    $receipt = $State.cpu_affinity_receipt
    foreach ($identity in @(@($State.pid,$State.root_start_filetime), @($receipt.pid,$receipt.creation_filetime))) {
        $process = Get-Process -Id ([int]$identity[0]) -ErrorAction SilentlyContinue
        if (-not $process -or $process.StartTime.ToUniversalTime().ToFileTimeUtc() -ne [long]$identity[1]) {
            throw 'Diagnostic process identity no longer matches its recorded PID/time'
        }
    }
    $path = Join-Path $State.run_dir 'backend-progress\latest.json'
    $progress = Read-Service64kProgressFile $path
    if ($progress.schema -ne 1 -or $progress.pid -ne $receipt.pid -or
        $progress.creation_filetime -ne $receipt.creation_filetime -or
        $progress.busy -isnot [bool] -or $progress.sequence -isnot [long] -and $progress.sequence -isnot [int]) {
        throw 'Backend progress record does not match the reviewed worker'
    }
    return $progress
}

function Copy-Service64kDiagnosticTail([string]$Source, [string]$Destination, [int]$Maximum) {
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) { return }
    $inputFile = [IO.File]::Open($Source, 'Open', 'Read', [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
    try {
        $length = $inputFile.Length
        $offset = [Math]::Max(0L, $length - $Maximum)
        $null = $inputFile.Seek($offset, 'Begin')
        $bytes = [byte[]]::new([int]($length - $offset))
        $total = 0
        while ($total -lt $bytes.Length) {
            $read = $inputFile.Read($bytes,$total,$bytes.Length-$total)
            if ($read -eq 0) { break }
            $total += $read
        }
        if ($total -ne $bytes.Length) { throw 'Diagnostic file changed while copying' }
        [IO.File]::WriteAllBytes($Destination, $bytes)
        return [ordered]@{source_bytes=$length;copied_bytes=$total;offset=$offset;truncated=($offset -gt 0)}
    } finally { $inputFile.Dispose() }
}

function Save-Service64kProgressCapture($State, [string]$Reason,
    [ValidateRange(0,1500)][int]$WaitMilliseconds = 1500) {
    $result = [ordered]@{enabled=($State.progress_diagnostics -eq $true);stack_acknowledged=$false;error=$null}
    if (-not $result.enabled) { return $result }
    $clock = [Diagnostics.Stopwatch]::StartNew()
    try {
        $id = [guid]::NewGuid().ToString('N')
        $directory = Join-Path $State.run_dir ('diagnostic-captures\' + $id)
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        $result.directory = $directory
        $result.request_id = $id
        $result.reason = $Reason.Substring(0,[Math]::Min($Reason.Length,160))
        $backend = Join-Path $State.run_dir 'backend-progress'
        try {
            $progress = Read-Service64kBackendProgress $State
            if ($progress) {
                $request = [ordered]@{schema=1;request_id=$id;pid=$progress.pid;creation_filetime=$progress.creation_filetime}
                Write-Service38Json (Join-Path $backend 'capture-request.json') $request
                $deadline = $clock.ElapsedMilliseconds + $WaitMilliseconds
                do {
                    $ackPath = Join-Path $backend 'capture-ack.json'
                    if (Test-Path -LiteralPath $ackPath -PathType Leaf) {
                        $ack = Read-Service64kProgressFile $ackPath 4096
                        if ($ack.schema -eq 1 -and $ack.request_id -ceq $id -and
                            $ack.pid -eq $progress.pid -and $ack.creation_filetime -eq $progress.creation_filetime) {
                            $result.stack_acknowledged = $true
                            break
                        }
                    }
                    if ($clock.ElapsedMilliseconds -ge $deadline) { break }
                    Start-Sleep -Milliseconds 25
                } while ($true)
            } else { $result.error = 'Backend identity has not been accepted during startup' }
        } catch { $result.error = $_.Exception.Message }
        $result.latest_copy = Copy-Service64kDiagnosticTail (Join-Path $backend 'latest.json') (Join-Path $directory 'latest.json') 65536
        $result.stack_copy = Copy-Service64kDiagnosticTail (Join-Path $backend 'stacks.log') (Join-Path $directory 'stacks.log') 1048576
        $result.at = (Get-Date).ToString('o')
        $result.elapsed_ms = $clock.ElapsedMilliseconds
        Write-Service38Json (Join-Path $directory 'capture.json') $result
    } catch { $result.error = $_.Exception.Message }
    return $result
}

function Test-Service64kProgressStall($Progress, $Watch) {
    if (-not $Progress -or -not $Progress.busy) {
        $Watch.sequence = $null
        $Watch.clock.Restart()
        return $false
    }
    if ($Watch.sequence -ne $Progress.sequence) {
        $Watch.sequence = $Progress.sequence
        $Watch.clock.Restart()
    }
    return $Watch.clock.Elapsed.TotalSeconds -ge 45
}
