#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$VenvRoot,
    [Parameter(Mandatory)][string]$Output,
    [switch]$ProjectOperators,
    [ValidateSet('core','q4-attention')][string]$Suite = 'core',
    [switch]$AllowUnqualifiedHost
)
$ErrorActionPreference = 'Stop'
if (-not $IsWindows) { throw 'This diagnostic requires Windows' }
if (-not $AllowUnqualifiedHost) { throw 'Explicit authorization for the current unqualified host is required' }
$source64k = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$labRoot = Join-Path $source64k 'scripts\windows-rocm64k'
. (Join-Path $labRoot 'service38_safety.ps1')
. (Join-Path $labRoot 'service64k_paths.ps1')
$config = Read-Service64kConfig (Join-Path $source64k '.local\windows-rocm64k\config.json')
$candidate = [IO.Path]::GetFullPath($VenvRoot)
if ($candidate -ieq $config.venv_root) { throw 'Use an isolated candidate environment' }
$python = Join-Path $candidate 'Scripts\python.exe'
$sdk = Join-Path $candidate 'Lib\site-packages\_rocm_sdk_devel'
foreach ($required in @($python, (Join-Path $sdk 'lib\llvm\bin\clang-cl.exe'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing candidate file: $required" }
}
$runRoot = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $runRoot) { throw 'Use a fresh diagnostic output directory' }
$mutex = [Threading.Mutex]::new($false, 'Local\FreeTokenROCm-Service38-1919')
$owned = $false
$process = $null
$stdout = $null; $stderr = $null
$result = [ordered]@{ model_loaded=$false; host_vendor_matrix_qualified=$false; completed=$false }
try {
    try { $owned = $mutex.WaitOne(5000) }
    catch [Threading.AbandonedMutexException] {
        $owned = $true
        throw 'The previous controller was interrupted; inspect its state before GPU diagnostics'
    }
    if (-not $owned) { throw 'The managed controller is busy' }
    if (@(Get-Service38Listener 1919).Count) { throw 'A service is already listening on port 1919' }
    foreach ($name in @('active-service.json','recovery-required.json','stop-request.json')) {
        if (Test-Path -LiteralPath (Join-Path $config.state_root $name)) { throw "Managed state requires attention: $name" }
    }
    $guard = Get-Service64kGuard
    $before = Assert-Service38Resources $guard 'startup'
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    Write-Service38Json (Join-Path $runRoot 'resource-guard.json') $guard
    $result.resources_before = $before
    $result.host_os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber
    $result.host_driver = @(Get-CimInstance Win32_PnPSignedDriver -Filter "DeviceClass='DISPLAY'" |
        Select-Object DeviceName,DriverVersion)
    $version = '10.0.26100.0'
    $msvc = $config.msvc_root
    $windowsSdk = $config.windows_sdk_root
    $env:HIP_PATH = $env:ROCM_HOME = $env:ROCM_PATH = $sdk
    $env:HIP_PLATFORM = 'amd'
    $env:CC = Join-Path $sdk 'lib\llvm\bin\clang-cl.exe'
    $env:DISTUTILS_USE_SDK = '1'; $env:MAX_JOBS = '1'
    $env:PYTORCH_ROCM_ARCH = $env:FREETOKEN_ROCM_ARCH = $env:TVM_FFI_ROCM_ARCH_LIST = 'gfx1201'
    $env:TRITON_LIBHIP_PATH = Join-Path $sdk 'bin\amdhip64_7.dll'
    $env:TVM_FFI_DISABLE_TORCH_C_DLPACK = '1'
    $env:PYTHONNOUSERSITE = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:FREETOKEN_MAMBA_SSM_DTYPE = 'float32'; $env:FT_GGUF_BACKEND = 'hip'
    $env:FREETOKEN_TRITON_CACHE_AMD_KNOB = '0'; $env:AMDGCN_USE_BUFFER_OPS = '1'
    $env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'
    $env:CUDA_VISIBLE_DEVICES = $env:HIP_VISIBLE_DEVICES = $env:ROCR_VISIBLE_DEVICES = '0'
    $env:INCLUDE = "$msvc\include;$msvc\..\..\..\Auxiliary\VS\include;$windowsSdk\include\$version\ucrt;$windowsSdk\include\$version\um;$windowsSdk\include\$version\shared"
    $env:LIB = "$msvc\lib\x64;$windowsSdk\lib\$version\ucrt\x64;$windowsSdk\lib\$version\um\x64"
    $env:PATH = "$sdk\bin;$sdk\lib\llvm\bin;$msvc\bin\HostX64\x64;$windowsSdk\bin\$version\x64;$(Split-Path $python);$env:PATH"
    $env:PYTHONPATH = "$candidate\Lib\site-packages;$source64k\python"
    $start = [Diagnostics.ProcessStartInfo]::new($python)
    $start.UseShellExecute = $false; $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
    $start.WorkingDirectory = $source64k
    foreach ($argument in @('-X','utf8','-B',(Join-Path $PSScriptRoot 'qualify_gpu.py'),
                            '--output',(Join-Path $runRoot 'checks'))) { $start.ArgumentList.Add($argument) }
    if ($ProjectOperators) { $start.ArgumentList.Add('--project-operators') }
    $start.ArgumentList.Add('--suite'); $start.ArgumentList.Add($Suite)
    $process = [Diagnostics.Process]::Start($start)
    $result.pid = $process.Id
    $result.process_started = $process.StartTime.ToString('o')
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    $deadline = [DateTime]::UtcNow.AddSeconds(240)
    Write-Output "GPU diagnostic started: PID $($process.Id), timeout 240 s"
    while (-not $process.WaitForExit(1000)) {
        if (Test-Path -LiteralPath (Join-Path $config.state_root 'stop-request.json')) {
            throw 'Managed stop requested during GPU diagnostics'
        }
        if ([DateTime]::UtcNow -gt $deadline) { throw 'GPU diagnostic exceeded its timeout' }
        $reading = Assert-Service38Resources $guard 'runtime'
        [IO.File]::AppendAllText((Join-Path $runRoot 'resources.jsonl'),
            ($reading | ConvertTo-Json -Compress) + [Environment]::NewLine)
    }
    [IO.File]::WriteAllText((Join-Path $runRoot 'stdout.txt'), $stdout.GetAwaiter().GetResult())
    [IO.File]::WriteAllText((Join-Path $runRoot 'stderr.txt'), $stderr.GetAwaiter().GetResult())
    $result.exit_code = $process.ExitCode
    $result.completed = $true
    if ($process.ExitCode -ne 0) { throw "GPU diagnostic failed with exit code $($process.ExitCode); inspect $runRoot" }
} catch {
    $result.error = $_.Exception.Message
    if ($process -and -not $process.HasExited) { $process.Kill($true); $process.WaitForExit() }
    if ($stdout) { [IO.File]::WriteAllText((Join-Path $runRoot 'stdout.txt'), $stdout.GetAwaiter().GetResult()) }
    if ($stderr) { [IO.File]::WriteAllText((Join-Path $runRoot 'stderr.txt'), $stderr.GetAwaiter().GetResult()) }
    throw
} finally {
    if (Test-Path -LiteralPath $runRoot) {
        try { $result.resources_after = Assert-Service38Resources $guard 'recovery' }
        catch { $result.resources_after_error = $_.Exception.Message }
        Write-Service38Json (Join-Path $runRoot 'process-result.json') $result
    }
    if ($process) { $process.Dispose() }
    if ($owned) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
$result | ConvertTo-Json -Depth 6
