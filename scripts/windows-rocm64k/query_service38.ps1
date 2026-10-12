param(
    [Parameter(Mandatory)][ValidateSet('processes', 'resources', 'listeners', 'gpu-identity')][string]$Kind,
    [Parameter(Mandatory)][string]$PayloadBase64
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try {
    # Parent assigns our job before releasing this gate. No provider or child
    # process is touched during the start/assignment window.
    if ([Console]::In.ReadLine() -ne 'probe-ready') { throw 'Probe parent did not confirm job ownership' }
    $payload = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($PayloadBase64)) | ConvertFrom-Json
    switch ($Kind) {
        'gpu-identity' {
            Add-Type -Path (Join-Path $PSScriptRoot 'service38_gpu_identity.cs')
            $value = [ordered]@{
                adapters = @([Service38GpuIdentity]::Read())
                counters = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPULocalAdapterMemory -OperationTimeoutSec 2 |
                    Select-Object Name, LocalUsage)
                boot_utc = (Get-CimInstance Win32_OperatingSystem -OperationTimeoutSec 2).LastBootUpTime.ToUniversalTime().ToString('o')
            }
        }
        'processes' {
            $argsForQuery = @{ ClassName = 'Win32_Process'; OperationTimeoutSec = 2; ErrorAction = 'Stop' }
            if ($payload.filter) { $argsForQuery.Filter = [string]$payload.filter }
            $value = @(Get-CimInstance @argsForQuery | Select-Object ProcessId, ParentProcessId, ExecutablePath, CommandLine, CreationDate)
        }
        'resources' {
            # Preserve the existing guard's CIM memory semantics and GPU LUID.
            $os = Get-CimInstance Win32_OperatingSystem -OperationTimeoutSec 2
            $gpu = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPULocalAdapterMemory -OperationTimeoutSec 2 |
                Where-Object { $_.Name -eq $payload.gpu_instance })
            if ($gpu.Count -ne 1) { throw "Cannot measure configured GPU instance $($payload.gpu_instance)" }
            $value = [ordered]@{
                at = (Get-Date).ToString('o')
                free_physical_gib = [math]::Round($os.FreePhysicalMemory * 1KB / 1GB, 3)
                free_commit_gib = [math]::Round($os.FreeVirtualMemory * 1KB / 1GB, 3)
                gpu_local_used_gib = [math]::Round([double]$gpu[0].LocalUsage / 1GB, 3)
                gpu_local_free_gib = [math]::Round([double]$payload.gpu_total_gib - [double]$gpu[0].LocalUsage / 1GB, 3)
            }
        }
        'listeners' {
            # Connection tables can also wait on a system provider; isolate them too.
            $value = @(Get-NetTCPConnection -LocalPort ([int]$payload.port) -State Listen -ErrorAction Stop |
                Select-Object LocalPort, OwningProcess)
        }
    }
    [ordered]@{ ok = $true; value = $value } | ConvertTo-Json -Compress -Depth 7
} catch {
    # No matching listener is a normal empty result, unlike a provider failure.
    if ($Kind -eq 'listeners' -and $_.FullyQualifiedErrorId -like 'CmdletizationQuery_NotFound*') {
        [ordered]@{ ok = $true; value = @() } | ConvertTo-Json -Compress
    } else {
        [ordered]@{ ok = $false; error = $_.Exception.Message } | ConvertTo-Json -Compress
    }
}
