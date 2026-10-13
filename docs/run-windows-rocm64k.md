# Managed Windows 64K service

The repository launcher is `scripts/windows-rocm64k/service64k.ps1`. It uses this
checkout as the engine source and serves the reviewed Qwen3.6-35B-A3B GGUF Q4_1
profile on `http://127.0.0.1:1919` with context and KV capacity 65536.

## Requirements and local configuration

Use Windows, PowerShell 7, CPython3.12.0 and the RX9070XT / gfx1201.
Configuration schema2 records the chosen native runtime:

| Runtime | AMD Torch | SDK | Triton Windows |
|---|---|---|---|
| `rocm1010` (new configurations) | 2.14.0+rocm10.1.0 | 10.1.0 | 3.8.0.post29 |
| `rocm1000` (rollback) | 2.13.0+rocm10.0.0 | 10.0.0 | 3.8.0.post28 |

Both retain the recorded74 packages. Legacy schema1 configurations select
`rocm1000`; reading them does not rewrite the file. `-Runtime` is accepted only
by `Configure`, and later commands use the persisted choice.
The [10.1 constraints](../constraints/windows-rocm1010-candidate-20261012.txt)
and [upgrade report](rocm1010-upgrade-20261012.md) record its exact environment.
The [10.0 constraints](../constraints/windows-rocm1000-pydeps-20261012.txt)
and [installation guide](install-windows-rocm10.md) describe the rollback environment.

ROCm10.1 requires the reviewed range-fold compiler workaround. The controller
verifies its SHA256 and the short64K/operator/Stop proof. The bootstrap installs
the hook separately before GPU initialization in every spawned kernel process;
failure terminates that process. Installed Triton files are unchanged. Old10.0
long64K evidence is not treated as a qualification of the new native stack.

The [2026-10-13 full64K series](rocm1010-full-series-20261013.md) passed353requests,
the exact context boundary, long decode, queue/cancel,30-minute load and Stop
on the new native stack. Post-soak control speed declined50.02→39.06tokens/s;
the cause remains open. The release reader retains its immutable short acceptance
proof; the full-series manifest records later functional results and performance
limitations separately.

From the checkout root, configure the paths to the existing environment and
model. The model and environment stay in their current locations.

```powershell
pwsh -NoProfile -File .\scripts\windows-rocm64k\service64k.ps1 -Action Configure `
  -Runtime rocm1010 `
  -VenvRoot 'D:\FreeToken\venv' `
  -ModelPath 'D:\Models\Qwen_Qwen3.6-35B-A3B-Q4_1.gguf' `
  -StateRoot 'D:\FT-state\rocm1010'
```

The command checks the environment and creates
`.local/windows-rocm64k/config.json`. This directory is ignored by Git and
contains local settings; the selected state root holds service state, run logs
and caches. Use a short state root: the controller rejects a model JIT DLL path
of260 or more characters before loading the model. The default build
paths are Visual Studio 2022 Build Tools / MSVC 14.44.35207 and Windows SDK
10.0.26100.0. `Configure` accepts `-MsvcRoot`, `-WindowsSdkRoot`, `-StateRoot`,
and `-ConfigFile` for different directory locations. Relative paths are resolved
against the checkout root. Pass `-ConfigFile` to subsequent commands when using
a nondefault configuration file. An existing configuration is preserved.

On this workspace the default configuration now selects `rocm1010`, state in
`C:\Users\Iskandar\FreeTokenROCm\.local\service64k-rocm1010`. The original
configuration is preserved as `config.before-rocm1010-20261013.json`, and
`config-rocm1000.json` uses the old environment with a separate short state root.
To use the rollback configuration, first Stop the active service, then:

```powershell
pwsh -NoProfile -File .\scripts\windows-rocm64k\service64k.ps1 -Action Start `
  -ConfigFile .local/windows-rocm64k/config-rocm1000.json
```

Use the same `-ConfigFile` for its Status/Stop. Both runtimes share the existing
mutex, port1919, process identity checks, resource observer and Stop implementation.

## Daily commands

```powershell
pwsh -NoProfile -File .\scripts\windows-rocm64k\service64k.ps1 -Action Verify
pwsh -NoProfile -File .\scripts\windows-rocm64k\service64k.ps1 -Action Preflight
pwsh -NoProfile -File .\scripts\windows-rocm64k\service64k.ps1 -Action Start
pwsh -NoProfile -File .\scripts\windows-rocm64k\service64k.ps1 -Action Status
pwsh -NoProfile -File .\scripts\windows-rocm64k\service64k.ps1 -Action Stop
```

`Verify` checks source, packages, native binaries, launcher files, prior
acceptance evidence, GPU identity, and configured build paths. `Preflight`
also checks current available memory and writes the effective launch arguments.
Both run without loading the model. `Start` loads the model and sends the usual
short readiness request. `Health` checks the live service after verifying its
listener owner. `Stop` also works while startup is in progress.

The fixed profile uses BF16 KV, FP32 GDN state, naive cache, page size 1,
prefill chunks 512, MoE LRU cache 2048, serial expert loading, ordinary prefill
overlap, one backend/frontend request, default output budget 2048, and logits
checks every 64 tokens. CUDA graphs, diagnostic tracing, source knob opt-in,
and routed prefill opt-in remain disabled. The verified process-start Triton
buffer-ops cache is retained, with Torch loaded before Triton.

## Resource guard and recovery

All figures are GiB of available memory.

| Phase | RAM | Commit | Local VRAM |
|---|---:|---:|---:|
| Before Start | 38 | 45 | 14 |
| During service | 12 | 4 | 3 |
| After Stop | 38 | 28 | 14 |

The GPU counter is resolved from the current adapter LUID at each launch.
Its total memory is capped at the smaller of the recorded 16 GiB and actual
DXGI dedicated memory. Reboots do not require editing a saved counter name.

The observer retains bounded system probes, process ownership checks, fatal
log detection, and cleanup. Start and Stop share the laboratory controller's
port-specific mutex. An occupied port is not replaced. If `Status` reports
`RECOVERY_REQUIRED`, inspect `recovery-required.json` and the latest run logs;
preserve that evidence while resolving the cause. Stop operates only on the
recorded process identities.

The earlier laboratory scripts and their quarantine history remain available.
For rollback, stop the repository service first, then use that recorded
launcher and its qualified environment. Keep the repository configuration and
run logs for comparison.

## Acceptance scope

The bundled `reviewed-release.json` and `evidence/` preserve the successful
[October integration](integration-research-stage8-20261012.md), prior ordinary
63488-token long-prefix result, numerical-source inheritance, and the
[ordinary launch check](integration-release-prep-20261012.md). Historical paths
inside evidence are recorded context; verification reads the bundled files.

The repository launcher was checked on 2026-10-12 without loading the model:
23 targeted tests passed, `Configure`, `Verify`, and `Preflight` passed, and
`Status` reported `STOPPED` with no owned processes or listeners. Source
identity was also verified from both existing checkouts. A new model smoke or
full benchmark series was not performed for this packaging change.

The verifier checks all 476 engine files, two build files, 74 package versions,
and five native/compiler binaries. Engine text uses CRLF-to-LF normalization;
other bytes are preserved. Launcher line endings are fixed through
`.gitattributes`; historical evidence keeps its original bytes. Documentation
and Git metadata changes do not invalidate engine identity. Changed engine,
packages, or native binaries require updated acceptance evidence.
