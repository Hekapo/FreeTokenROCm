"""Opt-in baseline validation; no GPU imports, installations or environment writes."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

BASELINE_ENV = "FREETOKEN_EXPERIMENT_BASELINE"
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})
_CACHE_VARS = ("TVM_FFI_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME")
_SDK_VARS = ("HIP_PATH", "ROCM_HOME", "ROCM_PATH", "HIP_DEVICE_LIB_PATH")
_ARCH_VARS = ("FREETOKEN_ROCM_ARCH", "PYTORCH_ROCM_ARCH", "TVM_FFI_ROCM_ARCH_LIST",
              "TRITON_OVERRIDE_ARCH", "ROCM_SDK_TARGET_FAMILY")


def _boolean(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} requires an explicit boolean (0/1, false/true, no/yes, off/on)")


def validate_baseline(config: Any) -> dict[str, Any] | None:
    """Return a configuration manifest, not a hardware-validation result."""
    if not _boolean(BASELINE_ENV):
        return None
    if "FREETOKEN_FUSED_COPY" not in os.environ or _boolean("FREETOKEN_FUSED_COPY"):
        raise ValueError("Baseline requires FREETOKEN_FUSED_COPY=0 before starting Python")
    for name in ("FREETOKEN_SKIP_FAST_INDEX_COPY", "FREETOKEN_SKIP_BANK_PIN",
                 "FREETOKEN_DISABLE_KERNEL_CACHE_VERSION_CHECK", "FREETOKEN_DISABLE_JIT"):
        if _boolean(name):
            raise ValueError(f"Baseline forbids {name}=1")
    if not _boolean("FREETOKEN_DISABLE_KERNEL_CACHE"):
        raise ValueError("Baseline requires FREETOKEN_DISABLE_KERNEL_CACHE=1 and fresh JIT caches")
    if config.cuda_graph_max_bs != 0 or config.cuda_graph_bs not in (None, []):
        raise ValueError("Baseline requires cuda_graph_max_bs=0 and no explicit graph batch sizes")
    if config.use_dummy_weight:
        raise ValueError("Baseline requires real weights, not use_dummy_weight")
    if config.moe_strategy == "auto":
        raise ValueError("Baseline requires an explicit moe_strategy for comparable runs")

    # The offload module caches the flag at import; changing the environment is too late.
    offload = sys.modules.get("freetoken.moe.offload_cache")
    fused = getattr(offload, "_FUSED_COPY", None)
    if offload is not None and fused is not False:
        raise ValueError("Baseline found fused-copy enabled or unknown after import; restart Python")
    cache_dirs = {}
    for name in _CACHE_VARS:
        value = os.environ.get(name, "")
        if not value.strip() or not pathlib.Path(value).is_absolute():
            raise ValueError(f"Baseline requires an absolute, experiment-specific {name}")
        cache_dirs[name] = str(pathlib.Path(value).resolve())
    if len(set(cache_dirs.values())) != len(cache_dirs):
        raise ValueError("Baseline cache directories must be distinct")

    return {
        "schema_version": 1,
        "kind": "baseline_configuration",
        "hardware_validation": "not_performed",
        "python_executable": sys.executable,
        "source_directory": str(pathlib.Path(__file__).resolve().parents[2]),
        "model_path": config.model_path,
        "dtype": str(config.dtype),
        "moe_strategy": config.moe_strategy,
        "cuda_graph_max_bs": config.cuda_graph_max_bs,
        "cuda_graph_bs": config.cuda_graph_bs,
        "fused_copy_at_config": fused if offload is not None else "module_not_loaded",
        "required_copy_mode": "per_bank",
        "required_fused_environment": False,
        "cache_directories": cache_dirs,
        "requested_sdk": {name: os.environ[name] for name in _SDK_VARS if name in os.environ},
        "requested_arch": {name: os.environ[name] for name in _ARCH_VARS if name in os.environ},
        "note": "Configuration only. Attach collector JSON and real GPU/JIT/serving results.",
    }
