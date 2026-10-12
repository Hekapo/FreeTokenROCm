function Get-Service64kRuntimeNames {
    @('service64k.ps1','service64k_lifecycle.ps1','service64k_paths.ps1','service64k_release.ps1','launch64k.ps1','service38_safety.ps1','service38_probes.ps1','service38_probe_native.cs','service38_gpu_identity.cs','query_service38.ps1','observe_service38.ps1','stable_triton_knob.py','cached_stats_launcher.py','reviewed_source.py','accepted-source.json','profile.json','quarantine-history.json')
}

function Read-Service38WorkingRelease {
    $release = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'reviewed-release.json') -Raw | ConvertFrom-Json
    if ($release.schema -ne 3 -or $release.mode -ne 'repository-64k-20261012' -or
        $release.profile -ne '64k' -or $release.capacity -ne 65536 -or $release.kv_capacity -ne 65536 -or
        $release.max_running_requests -ne 1 -or $release.cache_stable_triton_knob -ne $true -or
        $release.user_policy -ne 'reuse verified stages; no repeated full qualification') {
        throw 'Unexpected repository 64K release record'
    }
    $required = @(Get-Service64kRuntimeNames)
    $seen = @()
    foreach ($entry in $release.runtime_hashes) {
        if ($entry.path -notin $required -or $entry.path -in $seen -or
            (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $entry.path)).Hash -ine $entry.sha256) {
            throw "Reviewed launcher input changed: $($entry.path)"
        }
        $seen += $entry.path
    }
    if ($seen.Count -ne $required.Count) { throw 'Incomplete repository launcher record' }
    $profile = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'profile.json') -Raw | ConvertFrom-Json
    $scalars = @{
        schema=1; profile='64k'; model='qwen3.6-35b-a3b-q4_1'; model_filename='Qwen_Qwen3.6-35B-A3B-Q4_1.gguf';
        port=1919; context_capacity=65536; kv_capacity=65536; max_running_requests=1;
        frontend_generation_limit=1; default_max_output_tokens=2048; logit_check_every=64;
        prefill_overlap=$true; source_knob_enabled=$false; moe_prefill_demand_tokens=0;
        cache_stable_triton_knob=$true; amd_buffer_ops=$true; arch='gfx1201';
        gpu_vendor_id=4098; gpu_device_id=30032; gpu_total_gib=16;
        msvc_version='14.44.35207'; windows_sdk_version='10.0.26100.0'
    }
    foreach ($name in $scalars.Keys) {
        if ($profile.$name -ne $scalars[$name]) { throw "Unexpected profile setting: $name" }
    }
    $floors = @{
        startup = @{free_physical_gib=38; free_commit_gib=45; gpu_local_free_gib=14};
        runtime = @{free_physical_gib=12; free_commit_gib=4; gpu_local_free_gib=3};
        recovery = @{free_physical_gib=38; free_commit_gib=28; gpu_local_free_gib=14}
    }
    foreach ($phase in $floors.Keys) {
        foreach ($name in $floors[$phase].Keys) {
            if ($null -eq $profile.$phase.$name -or -not [double]::IsFinite([double]$profile.$phase.$name) -or
                $profile.$phase.$name -lt $floors[$phase][$name]) {
                throw "Resource threshold weakened: $phase/$name"
            }
        }
    }
    return Read-Service64kEvidence $release
}

function Assert-Service38RecordedStop($Status) {
    if ($Status.status -ne 'STOPPED' -or $Status.root_owned -ne $false -or $Status.observer_owned -ne $false -or
        $Status.process_tree_count -ne 0 -or $Status.listeners -ne 0 -or $Status.recovery_required -ne $false) {
        throw 'Integration evidence has no verified clean Stop'
    }
}


function Read-Service64kEvidence($Release) {
    $source = $source64k
    $venv = $venvRoot
    $evidence = @{}
    foreach ($entry in $Release.evidence_hashes) {
        if ($entry.kind -notin @('capacity','source','verification','long','long-stop','ordinary') -or
            $evidence.ContainsKey($entry.kind) -or $entry.path -cne ('evidence/' + $entry.kind + '.json') -or
            (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot $entry.path)).Hash -ine $entry.sha256) {
            throw 'Integration evidence changed or duplicated'
        }
        $evidence[$entry.kind] = Get-Content -LiteralPath (Join-Path $PSScriptRoot $entry.path) -Raw | ConvertFrom-Json
    }
    foreach ($kind in @('capacity','source','verification','long','long-stop','ordinary')) {
        if (-not $evidence.ContainsKey($kind)) { throw "Missing integration evidence: $kind" }
    }
    $capacity = $evidence.capacity
    if ($capacity.successful_requests -ne 9 -or $capacity.measured_requests -ne 6 -or
        $capacity.all_outputs_identical -ne $true -or $capacity.actual_prompt_tokens -ne 24 -or
        $capacity.completion_tokens -ne 391 -or $capacity.prefill_overlap -ne $true -or
        $capacity.source_knob_enabled -ne $false -or $capacity.moe_prefill_demand_tokens -ne 0 -or
        $capacity.lab_stable_knob -ne $true -or $capacity.amd_buffer_ops -ne $true -or
        $capacity.logit_check_every -ne 64) { throw 'Missing matched capacity acceptance' }
    $sizes = @{'4k'=4096; '8k'=8192; '64k'=65536}
    $contents = @()
    foreach ($phase in @('4k','8k','64k')) {
        $p = $capacity.phases.$phase
        if ($p.configured_capacity -ne $sizes[$phase] -or $p.kv_capacity -ne $sizes[$phase] -or
            $p.summary.all_pass -ne $true -or $p.summary.requests -ne 3 -or $p.summary.measured -ne 2 -or
            $p.resource_minima_gib.free_physical_gib -lt 12 -or $p.resource_minima_gib.free_commit_gib -lt 4 -or
            $p.resource_minima_gib.gpu_local_free_gib -lt 3 -or @($p.summary.records).Count -ne 3) {
            throw "Incorrect capacity evidence: $phase"
        }
        Assert-Service38RecordedStop $p.status
        foreach ($record in $p.summary.records) {
            if ($record.pass -ne $true -or $record.prompt_tokens -ne 24 -or $record.completion_tokens -ne 391 -or
                $record.first_output_text -ne '1' -or $record.reasoning -ne '' -or $record.finish_reason -ne 'stop') {
                throw 'Invalid capacity output evidence'
            }
            $contents += $record.content_sha256
        }
    }
    if (@($contents | Select-Object -Unique).Count -ne 1) { throw 'Capacity output hashes disagree' }
    $verified = $evidence.verification
    if ($verified.stage -ne 8 -or $verified.requests_passed -ne 9 -or $verified.model_stopped -ne $true -or
        $verified.candidate_package_versions -ne 74 -or $verified.candidate_native_files_match_control -ne 21790 -or
        $verified.control_files_unchanged -ne 33718 -or $verified.git_diff_check -ne $true) {
        throw 'Missing source/package integration verification'
    }
    $manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'accepted-source.json') -Raw | ConvertFrom-Json
    $code = $evidence.source.accepted_code_sha256
    if (@($manifest.python_files.PSObject.Properties).Count -ne 476 -or
        @($code.PSObject.Properties).Count -ne 476) { throw 'Incorrect reviewed source inventory' }
    foreach ($entry in $code.PSObject.Properties) {
        if ($manifest.python_files.($entry.Name) -ine $entry.Value) { throw "Source differs from capacity acceptance: $($entry.Name)" }
    }
    $numerical = $evidence.source.numerical_files_identical_to_after_stage2_long
    if (@($numerical.PSObject.Properties).Count -ne 14) { throw 'Missing inherited numerical source proof' }
    foreach ($entry in $numerical.PSObject.Properties) {
        if ($manifest.python_files.($entry.Name) -ine $entry.Value) { throw 'Inherited numerical path changed' }
    }
    $long = $evidence.long
    if ($long.all_pass -ne $true -or $long.requests -ne 3 -or $long.finite_diagnostics -ne $false -or
        @($long.records).Count -ne 3 -or @($long.records | Where-Object {$_.pass -ne $true}).Count -ne 0 -or
        @($long.records | Where-Object {$_.prompt_tokens -eq 63488 -and $_.completion_tokens -eq 391 -and
            $_.content_sha256 -eq $contents[0] -and $_.finish_reason -eq 'stop'}).Count -ne 1) {
        throw 'Missing prior ordinary long-prefix acceptance'
    }
    Assert-Service38RecordedStop $evidence.'long-stop'
    if ($evidence.ContainsKey('ordinary')) {
        $ordinary = $evidence.ordinary
        if ($ordinary.all_pass -ne $true -or $ordinary.requests -ne 1 -or $ordinary.normal_start -ne $true -or
            $ordinary.record.pass -ne $true -or $ordinary.record.prompt_tokens -ne 24 -or
            $ordinary.record.completion_tokens -ne 391 -or $ordinary.record.content_sha256 -ne $contents[0]) {
            throw 'Invalid ordinary launch acceptance'
        }
        Assert-Service38RecordedStop $ordinary.status
    }
    $probe = & (Join-Path $venv 'Scripts\python.exe') -B (Join-Path $PSScriptRoot 'reviewed_source.py') --repo $source --venv-root $venv --verify (Join-Path $PSScriptRoot 'accepted-source.json')
    if ($LASTEXITCODE -ne 0 -or ($probe | ConvertFrom-Json).reviewed_source_verified -ne $true) {
        throw 'Reviewed source or environment differs from integration acceptance'
    }
    return $Release
}
