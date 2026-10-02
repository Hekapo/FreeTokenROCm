# 64K migration checkpoint - 2026-10-02

This document records the source checkpoint merged from `64K-Migration` into
the project's `main` branch in `Hekapo/FreeTokenROCm`. The merge preserves work;
it does not qualify a new model runtime or enable the quarantined accelerated
profile. The first parent before this checkpoint was
`a50984cf2d5fa8acbe0f1d91cb8dfe74b508aae8`.

## Source changes in this checkpoint

- Stage query offsets for chunked Triton prefill in fenced, reusable pinned
  buffers, separately from accumulated KV offsets.
- Pass the host-known maximum query chunk length to the Qwen GDN causal
  convolution grid, avoiding per-layer device scalar reads.
- Include post-drain KV, recurrent and sliding-window occupancy in terminal
  abort acknowledgements, so frontend statistics reflect released resources.
- Use a Python thread for optional scheduler stack diagnostics and release its
  frame references between snapshots.
- Cover allocated context budgets, 65536-token admission boundaries, abort
  accounting, query metadata and the GDN convolution call path.
- Keep the shell's configured model ceiling separate from its current allocated
  KV budget, allowing the displayed budget to grow after a cache rebuild.

The process-start cached Triton AMD knob optimization was measured in the
laboratory harness. It has not been integrated into production source by this
checkpoint. Published numbers below therefore do not describe a fresh inference
run from this commit.

## Local runtime and historical evidence

The tested laboratory profile used Windows 11 build 26100, RX 9070 XT / gfx1201,
Python 3.12.0, AMD Torch 2.13.0+rocm10.0.0, ROCm SDK 10.0.0 and Triton Windows
3.8.0.post28. Model: `Qwen_Qwen3.6-35B-A3B-Q4_1.gguf`. Weight quantization is
Q4_1; KV dtype is BF16. The 64K profile requests 65536 tokens, naive cache,
512-token prefill chunks, serial expert execution, MoE cache 2048 / LRU, one
frontend/backend and CUDA graphs disabled.

Historical results are separate evidence for their recorded runtimes:

| Saved run | Result and scope |
|---|---|
| `ctx64k-stage3-20260930/execution10-short-path` | 602/602 checks and 8 clean Stops, including the 65280 + 256 boundary |
| Historical short comparison | 4K 40.05, 8K 40.27, original 64K 41.40 tokens/s; accelerated 4K/8K were not measured |
| `ctx64k-knob-compare-20261002` | Instrumented A/B: 62/62 checks; short median 40.7111 -> 51.8806 tokens/s, +27.44%; long 63488-token prefill passed |
| `ctx64k-working-smoke-20261002` | Triton-first import selected an incorrect HIP DLL; corrected by importing Torch first in the launcher |
| `ctx64k-working-smoke-20261002-torch-first` | Ordinary short request 51.975 tokens/s; long prefill failed with NaN/Inf after 4096 completed tokens; third request not run |

The latest ordinary smoke is **FAIL**. No working release record was produced;
its clean Stop was not credited. The fault/recovery markers remain in place.
The cause of the non-finite logits and the earlier whole-PC freeze is unknown.
Memory guard thresholds were not breached in this last failure. Identical
shared HSACO files and passing isolated GDN configurations do not establish
full-model correctness.

Logs and reports remain outside Git in the local workspace under
`rocm10-20260926/results/` and `rocm10-20260926/triton38-lab-20260928/`.
Large weights, wheels, binaries, caches and raw logs are not part of the source
checkpoint. The research snapshot is
`rocm10-20260926/research-refresh-20261002/SUMMARY_RU.md`.

## Merge verification

The selected CPU-only regression set is listed below. GPU visibility is disabled
in the test subprocess, Torch is imported before pytest collection, the existing
AMD environment is reused, and no packages are installed. GPU kernel cases in
the last two files are excluded by selecting the named CPU nodes.

```text
tests/scheduler/test_abort_inflight_prefill.py
tests/scheduler/test_cost_accounting_core.py
tests/scheduler/test_scheduler_kv_usage.py
tests/server/test_backend_death.py
tests/server/test_openai_api.py
tests/shell/test_token_limit.py
tests/models/test_qwen35_gdn_prefill.py::test_gdn_prefill_uses_host_query_lengths
tests/kernels/test_triton_attention.py::test_chunked_prefill_query_metadata_avoids_blocking_device_tensor
```

The initial selected run passed **133 tests in 13.54 seconds** before the shell
cache-growth correction. Its new regression then failed against the previous
implementation (`8192 != 65536`) and passed after the correction. The complete
shell test file passed **12 tests in 6.15 seconds** after the fix, including
8K -> 64K -> 4K -> 128K live KV updates with the model ceiling held at 64K and
metadata fallback cases. These checks validate CPU logic and mocked device
interfaces; they are not a GPU benchmark or model qualification.

The test subprocess used the existing ROCm 10 / Triton 3.8 Python interpreter,
`PYTHONPATH=<checkout>/python`, `PYTHONDONTWRITEBYTECODE=1`, and
`HIP_VISIBLE_DEVICES=ROCR_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES=-1`. Both calls
used `pytest.main` with `-q -o addopts= -p no:cacheprovider`; the initial call
selected the targets above. The final shell-only command was:

```powershell
python.exe -B -c 'import torch,pytest,sys; sys.exit(pytest.main(["-q","-o","addopts=","-p","no:cacheprovider","tests/shell/test_token_limit.py"]))'
```

## Next branch

The next source branch is `integration/research-refresh-20261002`, created from
the merged and pushed `main`. Its integration plan will identify the exact
baseline commit, source donors, acceptance gates and rollback boundaries.
Package upgrades, native kernel ports, driver changes and model runs are separate
future implementation steps.
