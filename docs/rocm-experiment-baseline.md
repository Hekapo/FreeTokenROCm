# ROCm experiment baseline (P candidate)

This is preparation, not an executed stack migration. The source base is
`4c2da356d4daf36c86e76cfff091e32b86e44081`. Package versions, driver settings,
model loaders and copy kernels are unchanged. GPU/JIT/serving validation on the
user's working stack is still required before accepting P.

## Changes

`kernel/_toolchain.py` consults an already-loaded GPU assignment, falling back
to the current device rather than device 0. It remains loadable by file path in
build environments, without importing FreeToken. Explicit architecture overrides
are preserved. Discovery logs the selected visible ordinal, detected architecture
and configured architecture variables; conflicting overrides are not silently
replaced. All-explicit cross-compilation settings do not require a GPU probe.

For a Linux/WSL SDK with only versioned HIP libraries, `kernel/utils.py` places the
compatibility link under `$XDG_CACHE_HOME/freetoken/rocm-lib/<digest>/` (default
`~/.cache`). The digest covers the canonical library path and its bytes. Changing
SDKs or replacing a runtime at the same path produces a different namespace on
fresh lookup. Aliases of one real file are deduplicated; several distinct runtime
files without an authoritative SDK `libamdhip64.so` alias cause an error instead
of a guessed version selection. Existing legacy aliases are not modified.
Windows `amdhip64.lib` handling and the TVM-FFI unknown-template guard remain.

This does not preserve old library bytes after an in-place SDK replacement and
is not support for hot-swapping runtimes in a running process. Keep separate SDKs
and restart Python. Full library hashing is a cold lookup cost, not a claimed
inference optimization. Compiler/bitcode/native artifacts still require the
separate environments and build directories described below.

## Opt-in strict configuration

Set these variables before starting a new Python/server process. This profile
validates configuration; it never installs packages or changes global settings.

```text
FREETOKEN_EXPERIMENT_BASELINE=1
FREETOKEN_FUSED_COPY=0
FREETOKEN_SKIP_FAST_INDEX_COPY=0
FREETOKEN_SKIP_BANK_PIN=0
FREETOKEN_DISABLE_KERNEL_CACHE=1
FREETOKEN_DISABLE_KERNEL_CACHE_VERSION_CHECK=0
FREETOKEN_DISABLE_JIT=0
```

Set `TVM_FFI_CACHE_DIR`, `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR` and
`XDG_CACHE_HOME` to distinct absolute directories reserved for this experiment.
For example, use separate subdirectories under an absolute `E0-P-candidate`
cache root. Use another root for E1; do not reuse the user's existing cache.
Do not set `FREETOKEN_KERNEL_CACHE_DIR` to reuse unverified binaries.

Run the existing verified serve command with an explicit MoE strategy, real
weights, `--cuda-graph-max-bs 0` and no explicit graph batch sizes. Do not change
the checkpoint, tokenizer, quantization, cache sizes or workload for the control.
This is not a new installer or an automatically selected serving configuration.

When enabled, `EngineConfig` rejects unknown booleans, fused-copy, skip-copy,
skip-pin, disabled JIT/version checks, reused prebuilt kernel-cache mode, dummy
weights, automatic MoE strategy, relative/identical cache directories, nonzero
or implicit graph maxima and explicit nonempty graph batch lists. It also
rejects an already-imported offload module with fused-copy enabled: changing the
environment after import is not enough. Without the new opt-in flag, existing
configuration behavior and other platforms' defaults are unchanged.

`FREETOKEN_BASELINE_CONFIG=<JSON>` in the local log records validated configuration
intent, requested SDK/architecture paths, source/Python locations and copy mode.
It explicitly says `hardware_validation=not_performed`. This is not a runtime
mapping probe or proof that a compiler, library, GPU or checkpoint is compatible.
Keep the log local and review private paths before sharing it.

## Hardware gate

Use the existing, unchanged collector from `chore/rocm-migration-prep` at
`4d5f83624a72373052f32b456c16e4d5d3f5a20d`. Run its `collect_rocm_env.py` with the
working FreeToken interpreter and `--repo` pointing at the actual checkout.
First collect without `--probe-torch`; the GPU probe remains a separate opt-in.
Attach that JSON to the configuration record; do not treat unknown fields as
verified. Record the real local SHA/dirty state separately from the published base.

Use a separate checkout/worktree and separately recreated Python environment;
never overwrite native extensions in the original checkout. Verify the imported
FreeToken path. No driver or global PATH changes. Venvs do not isolate the driver.
Do not run two competing test servers on the same GPU.

On the old working stack, check native/JIT loading, actual selected GPU/runtime,
`graph_runner.graph_bs_list == []`, and `_copy_fused_ok == False` when an offload
cache exists. Per-bank fast-copy still runs; this profile does not replace it
with staged PyTorch safe-copy. Check load, real output correctness, memory and
single/concurrent serving against the original workload. Preserve full local
logs. No speed claim follows from the configuration guard or unit tests.

After that gate, freeze P and create `build/rocm-stack-upgrade` from its exact SHA.
Do not update versions here or continue functional integration before preserving
that branch point. No merge to main, PR, or graph/fused activation is part of P.

## Assistant-side validation

Source-loaded tests import whole preparation modules by file path. GPU/package
boundaries are mocked; runtime files are temporary byte fixtures, not ELF/DLLs.
The real `EngineConfig` and `GraphRunner` bodies are exercised with dependency
stubs, including the no-capture early return. Existing full-engine tests are not
replaced by these checks.

```text
python -m pytest -q -o addopts= --confcutdir=tests/utils tests/utils/test_rocm_preparation.py
python -m pytest -q -o addopts= --confcutdir=tests/engine tests/engine/test_rocm_experiment.py
```

Initial candidate validation in the assistant's Linux/Python 3.13.5 environment:
71 tests passed. Five selected regression cases failed against the original
source base and passed after the fixes. The modified existing
`tests/utils/test_rocm_arch.py` was compile-checked but not run as a full module:
its imports require the FreeToken/Triton environment absent from the assistant.
No Windows execution, GPU tests, native compilation, model inference, A/B or soak
was performed. The full repository was not cloned; connector-exported source
blobs were checked against their Git SHA before the isolated tests.

## References

- Approved workflow: shared P, independent code and version branches, hardware gate.
- Source base: https://github.com/Hekapo/FreeTokenROCm/tree/4c2da356d4daf36c86e76cfff091e32b86e44081
- PyTorch current device: https://docs.pytorch.org/docs/main/generated/torch.cuda.current_device.html
- Extension build directory controls: https://docs.pytorch.org/docs/main/cpp_extension.html
- TVM-FFI cache control: https://github.com/apache/tvm-ffi/blob/main/python/tvm_ffi/cpp/extension.py
- Triton cache controls: https://github.com/triton-lang/triton/blob/main/python/triton/knobs.py

These references explain interfaces, not the compatibility of a future installed
stack. No external fork files were copied wholesale in this preparation step.
