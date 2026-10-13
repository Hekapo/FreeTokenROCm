# Windows ROCm dependency snapshots

These constraints describe Windows x64 / CPython 3.12 environments for the
Qwen3.6 GGUF Q4_1 profile. They pin the installed dependency graph, including
the AMD Torch/SDK distributions. They do not select a wheel source or install
packages by themselves.

| File | Environment |
|---|---|
| `windows-rocm1000-control-20261012.txt` | Original 73-package control, Transformers5.16.1 / Hub1.29.0 / FastAPI0.141.1 / Pydantic2.13.5 |
| `windows-rocm1000-pydeps-20261012.txt` | Qualified 74-package candidate, Transformers5.19.0 / Hub1.33.0 / FastAPI0.143.0 / Pydantic2.14.1 |
| `windows-rocm1010-candidate-20261012.txt` | 74-package Torch2.14 / SDK10.1 / Triton post29; selected GPU, Q4_1/attention and short64K/Stop pass with the explicit range-fold workaround; selected by the working64K config on2026-10-13 |
| `windows-rocm1010-profile-20261013.json` | Current working10.1 profile; full64K functional/resource/lifecycle evidence and open performance observation |
| `windows-rocm1010-full-series-20261013.json` | 353requests, context boundary, long decode, queue/cancel and30-minute load; post-soak control speed drops21.91% |

The two ROCm10.0 snapshots retain AMD Torch2.13.0+rocm10.0.0, SDK10.0.0, Triton Windows3.8.0.post28,
TVM-FFI0.1.13.post3, OpenAI2.54.0, NumPy2.4.6 and flashlib0.3.0.

Create a separate venv from the same Python, then use independent copies of the
qualified control's `Lib/site-packages` or install the exact AMD artifacts from
their recorded vendor sources. A generic PyPI Torch or rocm placeholder is not
a replacement for these artifacts. Use explicit `--python` on every install.

On that prepared candidate, install the selected PyPI group with the candidate
constraint file, for example:

```powershell
uv --no-config pip install --python <candidate-python> --constraint constraints/windows-rocm1000-pydeps-20261012.txt transformers==5.19.0 huggingface-hub==1.33.0 fastapi==0.143.0 pydantic==2.14.1
uv --no-config pip check --python <candidate-python>
```

For the recorded run, all six changed/new wheels were downloaded separately,
verified against publisher SHA256 values, and installed from local files.
The native runtime files were copied independently and compared by hash.

Run the checks in the [stage4 report](../docs/integration-research-stage4-20261012.md)
before adopting any changed combination. Reusing these constraints on a
different platform does not qualify that platform. ModelScope is an optional
extra and is absent from these snapshots.

Rollback selects the original control Python and its snapshot. Keep native/JIT
caches separate for each environment. At stage4 the managed launcher used
the control and the candidate ran through a bounded diagnostic plan. The later
ordinary64K activation is recorded below.

## ROCm10.1 candidate

The candidate changes nine vendor packages together. Its 65 other packages match
the qualified Python dependency environment. Runtime and build-system Torch
bounds now both allow `>=2.11,<2.15`; this allows the isolated candidate and does
not qualify every platform or select a vendor wheel automatically.

[The artifact manifest](windows-rocm1010-artifacts-20261012.json) records exact
URLs and SHA256 values for all nine downloaded artifacts. AMD's index did not
publish independent hashes; these AMD hashes identify the downloaded HTTPS
bytes. The Triton wheel hash matches the publisher's PyPI hash.

`rocm==10.1.0` was supplied as an sdist. Its local wheel was built offline with
CPython3.12.0 / setuptools84.0.0 and `ROCM_SDK_TARGET_FAMILY=gfx1201`. Retain the
source hash, build inputs, generated metadata and local wheel hash together.
Use the exact local wheels with the candidate constraints, explicit `--python`,
`--no-config`, and independent native/JIT caches.

All installed requirements, CPU regressions, native build/load and CPU TVM-FFI
tensor interop passed. Unmodified Triton post29 still fails the histogram and
loop range-analysis reproducers. The explicit process-only compiler hook now
passes four full-pipeline reproducers, five small GPU reference checks and six
selected project GPU tests. Q4_1/attention and one short model request at
configured65536 context/KV capacity have also passed with a clean managed Stop.
The short request produced the reference391-token answer at36.731 tokens/s.
The subsequent ordinary64K series passed353requests including the exact context
boundary, repeated long inputs, queue/cancel and30-minute varied load. Stop was
clean and resource floors held. The matched control after varied load fell
50.02→39.06tokens/s; its cause and isolated SDK improvement remain unestablished.
See the [full-series report](../docs/rocm1010-full-series-20261013.md).
Installed Triton files remain unchanged. The shared managed launcher now supports
both pinned stacks; the working configuration selects10.1 and preserves a10.0
rollback configuration. The2026-10-12 qualification manifest remains a historical
record of the separate trial. See the [managed launch guide](../docs/run-windows-rocm64k.md).

[The qualification manifest](windows-rocm1010-qualification-20261012.json)
records the pinned environment, compiler hook, native hashes, resource minima
and exact check scope. See the [upgrade report](../docs/rocm1010-upgrade-20261012.md)
for repeatable commands and remaining gates.

The host also differs from AMD's ROCm10.1 support row: Windows11 24H2 and
DriverStore32.0.31041.1004 versus 25H2 / 32.0.31041.5005. The candidate is prepared
for further qualification. See the [stage5 report](../docs/integration-research-stage5-20261012.md)
for evidence and remaining gates.

## Final integration profile (2026-10-12)

[The recorded profile](windows-rocm1000-profile-20261012.json) selects the qualified
74-package Python dependency environment with the unchanged ROCm10.0/Torch2.13/
Triton post28 native stack. It records compiler/DLL hashes, exact model settings,
memory gates and the source working-tree fingerprint; HEAD alone does not identify
the uncommitted source tested here.

[Stage8](../docs/integration-research-stage8-20261012.md) passed one warmup and two
measured short requests at each matched context/KV capacity4096/8192/65536. Source
optimization flags stay off; ordinary prefill overlap and the existing lab stable
knob are retained. Long-prefix evidence is scoped to stage2; the final full tree
was not rerun with a long prompt. This record does not change the managed Start
environment, remove quarantine, or promote the legacy working-release marker.

## Ordinary managed 64K launch (2026-10-12)

The later [release preparation](../docs/integration-release-prep-20261012.md)
connected this qualified Python74/native10.0 environment to ordinary Start and
made64K the default controller profile. A schema2 reviewed record binds source,
build inputs, packages, selected binaries and24 runtime files to the stage8 and
inherited stage2 evidence. One application request plus the built-in readiness
probe and cleanStop passed. The stage8 profileJSON above remains a snapshot of
that earlier acceptance; its managed-Start flags describe stage8, before this
later activation. The current record is in the external lab/audit snapshot.

## CPU affinity diagnostic (2026-10-13)

[windows-rocm1010-cpu-affinity-diagnostic-20261013.json](windows-rocm1010-cpu-affinity-diagnostic-20261013.json)
records the matched SDK comparison and two reversible backend affinity changes.
All 84 short requests passed with 65536 KV capacity. Selecting one logical CPU
per physical core improved decode by 25-28%; SDK version speedup was not shown.
This diagnostic is not default-launch, long-prefill, or mixed-load acceptance.
The earlier long probe stalled and its fault evidence is preserved. See the
[investigation report](../docs/rocm1010-speed-investigation-20261013.md).

## Backend affinity in the managed launcher (2026-10-13)

[windows-rocm1010-cpu-affinity-launcher-20261013.json](windows-rocm1010-cpu-affinity-launcher-20261013.json)
records the later repository implementation and short feature check: 96 CPU
tests, readiness, four exact 24/391 counting requests, verified backend-only
placement, and clean Stop with no remaining recorded processes. Median decode
was45.88tokens/s. The startup commit exception applied only to a single-use
frozen launcher copy; the production floor remains45GiB. Local ROCm10.1
configurations now select `one_per_core`; ROCm10.0 rollback retains `windows`.
Long-prefill and mixed-load behavior with this mode remain unqualified. See
the [implementation report](../docs/rocm1010-cpu-affinity-launcher-20261013.md).
