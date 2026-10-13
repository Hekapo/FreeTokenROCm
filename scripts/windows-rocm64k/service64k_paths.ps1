function Resolve-Service64kPath([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { throw 'A configured path is empty' }
    if ([IO.Path]::IsPathFullyQualified($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $source64k $Value))
}

function Read-Service64kConfig([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Configure the service first: service64k.ps1 -Action Configure -VenvRoot <venv> -ModelPath <GGUF>; config=$Path"
    }
    $value = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    return Resolve-Service64kConfig $value
}

function Resolve-Service64kConfig($Value) {
    $value = $Value
    $required = @('schema','venv_root','model_path','state_root','msvc_root','windows_sdk_root')
    if ($value.schema -in @(2,3)) { $required += 'runtime' }
    if ($value.schema -eq 3) { $required += 'backend_cpu_affinity' }
    if ($value.schema -notin @(1,2,3) -or @($value.PSObject.Properties).Count -ne $required.Count) {
        throw 'Unexpected local 64K configuration'
    }
    foreach ($name in $required) {
        if (-not $value.PSObject.Properties[$name]) { throw "Missing configuration field: $name" }
    }
    if ($value.schema -eq 1) {
        $value | Add-Member -NotePropertyName runtime -NotePropertyValue 'rocm1000'
    }
    if ($value.schema -ne 3) {
        $value | Add-Member -NotePropertyName backend_cpu_affinity -NotePropertyValue 'windows'
        $value.schema = 3
    }
    if ($value.runtime -cnotin @('rocm1000','rocm1010')) { throw 'Unknown pinned 64K runtime' }
    if ($value.backend_cpu_affinity -isnot [string] -or
        $value.backend_cpu_affinity -cnotin @('windows','one_per_core')) { throw 'Unknown backend CPU affinity mode' }
    foreach ($name in @('venv_root','model_path','state_root','msvc_root','windows_sdk_root')) {
        $value.$name = Resolve-Service64kPath $value.$name
    }
    $modulePrefix = $labRoot.TrimEnd('\') + '\'
    if ($value.state_root -ieq $source64k -or $value.state_root -ieq $labRoot -or
        $value.state_root.StartsWith($modulePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Mutable state must be stored outside the launcher directory and repository root'
    }
    return $value
}

function Get-Service64kRuntime {
    if (-not $runtimeName) { return 'rocm1000' }
    if ($runtimeName -cnotin @('rocm1000','rocm1010')) { throw 'Unknown pinned 64K runtime' }
    return $runtimeName
}

function Get-Service64kSourceManifest {
    $name = if ((Get-Service64kRuntime) -eq 'rocm1010') { 'accepted-source.rocm1010.json' } else { 'accepted-source.json' }
    return Join-Path $labRoot $name
}

function Assert-Service64kJitPaths([string]$StatePath) {
    $name = 'freetoken__fast_index_copy_1310720_32_2048_1024_4_1'
    $dll = Join-Path $StatePath ('runs\ctx64k-20261013-000000-00000000\tvm-ffi-cache\' + $name + '_0000000000000000\' + $name + '.dll')
    if ($dll.Length -ge 260) {
        throw "MSVC JIT DLL output would be $($dll.Length) characters; configure a shorter state root"
    }
}

function Assert-Service64kLocalPaths($Config) {
    if ($Config.state_root) { Assert-Service64kJitPaths $Config.state_root }
    $profile = Get-Content -LiteralPath (Join-Path $labRoot 'profile.json') -Raw | ConvertFrom-Json
    if ((Split-Path -Leaf $Config.model_path) -ine $profile.model_filename -or
        -not (Test-Path -LiteralPath $Config.model_path -PathType Leaf)) {
        throw "The reviewed profile requires $($profile.model_filename); configure its local path"
    }
    $files = @((Join-Path $Config.venv_root 'Scripts\python.exe'),
        (Join-Path $Config.venv_root 'Lib\site-packages\triton\__init__.py'))
    foreach ($path in $files) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required runtime file is missing: $path" }
    }
    $version = $profile.windows_sdk_version
    $directories = @((Join-Path $Config.msvc_root 'include'), (Join-Path $Config.msvc_root 'lib\x64'),
        (Join-Path $Config.windows_sdk_root "include\$version\ucrt"),
        (Join-Path $Config.windows_sdk_root "include\$version\um"),
        (Join-Path $Config.windows_sdk_root "lib\$version\ucrt\x64"),
        (Join-Path $Config.windows_sdk_root "lib\$version\um\x64"))
    foreach ($path in $directories) {
        if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "Required build directory is missing: $path" }
    }
}

function Get-Service64kGuard {
    $profile = Get-Content -LiteralPath (Join-Path $labRoot 'profile.json') -Raw | ConvertFrom-Json
    $identity = Invoke-Service38Probe 'gpu-identity' @{}
    $adapters = @($identity.adapters | Where-Object {
        $_.vendor_id -eq $profile.gpu_vendor_id -and $_.device_id -eq $profile.gpu_device_id -and
        $_.dedicated_video_bytes -ge 15.5GB -and ($_.flags -band 2) -eq 0
    })
    if ($adapters.Count -ne 1) { throw 'The reviewed RX 9070 XT GPU must be identified unambiguously' }
    $adapter = $adapters[0]
    $counters = @($identity.counters | Where-Object { $_.Name -ceq $adapter.counter_instance })
    if ($counters.Count -ne 1) { throw 'The current GPU LUID has no unambiguous memory counter' }
    return [pscustomobject]@{
        gpu_instance = $adapter.counter_instance
        gpu_total_gib = [Math]::Min([double]$profile.gpu_total_gib, [double]$adapter.dedicated_video_bytes / 1GB)
        startup = $profile.startup; runtime = $profile.runtime; recovery = $profile.recovery
    }
}

function Get-ProfileLaunchArgs {
    $null = Read-Service38WorkingRelease
    Assert-Service64kLocalPaths $config
    $runDir = Join-Path $stateRoot ('runs\ctx64k-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' +
        [guid]::NewGuid().ToString('N').Substring(0, 8))
    New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null
    $guardFile = Join-Path $runDir 'resource-guard.json'
    Write-Service38Json $guardFile (Get-Service64kGuard)
    return @{
        Label = 'reviewed-64k'; Runtime = (Get-Service64kRuntime); Src = $source64k; RunDir = $runDir; Port = $port;
        BackendCpuAffinity = $config.backend_cpu_affinity;
        ModelPath = $config.model_path; Python = (Join-Path $venvRoot 'Scripts\python.exe');
        SdkDevel = (Join-Path $venvRoot 'Lib\site-packages\_rocm_sdk_devel');
        TritonSite = (Join-Path $venvRoot 'Lib\site-packages'); StateRoot = $stateRoot;
        StateManifest = (Get-Service64kSourceManifest); ResourceGuard = $guardFile;
        MsvcRoot = $config.msvc_root; WindowsSdkRoot = $config.windows_sdk_root;
        WindowsSdkVersion = '10.0.26100.0'
    }
}

function Get-Service64kCpuAffinityProbe([string]$Mode) {
    if ($Mode -cnotin @('windows','one_per_core')) { throw 'Unknown backend CPU affinity mode' }
    $result = & (Join-Path $venvRoot 'Scripts\python.exe') -B (Join-Path $labRoot 'backend_cpu_affinity.py') --probe $Mode
    if ($LASTEXITCODE -ne 0) { throw 'Backend CPU affinity preflight failed' }
    return $result | ConvertFrom-Json
}

function Set-Service64kCpuAffinity([string]$Mode) {
    Assert-ProfileStartAllowed
    $null = Read-Service38WorkingRelease
    $probe = Get-Service64kCpuAffinityProbe $Mode
    $config.backend_cpu_affinity = $Mode
    Write-Service38Json $configPath $config
    return [ordered]@{configured=$true;config=$configPath;backend_cpu_affinity=$Mode;cpu=$probe;model_started=$false}
}

function Assert-Service64kBackendCpuAffinity($State, [string]$Mode) {
    $path = Join-Path $State.run_dir 'backend-affinity.jsonl'
    $records = @(Get-Content -LiteralPath $path | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.event -eq 'applied' })
    if ($records.Count -ne 1) { throw 'Missing or ambiguous backend CPU affinity receipt' }
    $record = $records[0]
    $owned = @(Get-Service38Tree $State | ForEach-Object { [int]$_.ProcessId })
    if ($record.role -cne 'backend' -or $record.mode -cne $Mode -or
        $record.process_name -cne 'freetoken-TP0-scheduler' -or
        $record.before_gpu_initialization -ne $true -or
        $owned -notcontains [int]$record.pid -or [int]$record.pid -eq [int]$State.pid) {
        throw 'Backend CPU affinity receipt does not match the managed worker'
    }
    $backend = Get-Process -Id ([int]$record.pid) -ErrorAction Stop
    $actualMask = [BitConverter]::ToUInt64([BitConverter]::GetBytes($backend.ProcessorAffinity.ToInt64()), 0)
    if ($backend.StartTime.ToUniversalTime().ToFileTimeUtc() -ne [long]$record.creation_filetime -or
        $actualMask -ne [uint64]$record.applied_mask) {
        throw 'Backend PID/time or live CPU affinity differs from its receipt'
    }
    if ($Mode -eq 'windows' -and $record.applied_mask -ne $record.original_mask) {
        throw 'Windows CPU placement was changed'
    }
    if ($Mode -eq 'one_per_core') {
        if ($record.group_count -ne 1 -or -not @($record.core_masks).Count -or
            (([uint64]$record.applied_mask -band [uint64]$record.original_mask) -ne [uint64]$record.applied_mask)) {
            throw 'Incorrect backend CPU affinity topology or inherited mask'
        }
        $covered = [uint64]0
        foreach ($core in $record.core_masks) {
            if ([uint64]$core -eq 0 -or ([uint64]$core -band $covered) -ne 0) {
                throw 'Invalid or overlapping physical core masks in backend receipt'
            }
            $covered = $covered -bor [uint64]$core
            $eligible = [uint64]$core -band [uint64]$record.original_mask
            $chosen = [uint64]$core -band [uint64]$record.applied_mask
            if (($eligible -gt 0 -and ($chosen -eq 0 -or ($chosen -band ($chosen - 1)) -ne 0)) -or
                ($eligible -eq 0 -and $chosen -ne 0)) { throw 'Backend affinity does not choose one LP per eligible core' }
        }
        if (($covered -band [uint64]$record.original_mask) -ne [uint64]$record.original_mask) {
            throw 'Backend receipt topology does not cover its inherited affinity'
        }
    }
    return $record
}

function Join-Service64kArguments([string[]]$Arguments) {
    $quoted = foreach ($argument in $Arguments) {
        if ($argument -ne '' -and $argument -notmatch '[\s"]') { $argument; continue }
        $escaped = [regex]::Replace($argument, '(\\*)"', {
            param($match)
            ('\' * (2 * $match.Groups[1].Length + 1)) + '"'
        })
        $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
        '"' + $escaped + '"'
    }
    return $quoted -join ' '
}
