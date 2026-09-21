#!/usr/bin/env python3
"""Collect a local FreeToken/ROCm inventory without installing or updating anything.

Only --out writes a file (exclusive creation). --probe-torch is opt-in because
importing PyTorch and enumerating GPUs can initialize the GPU runtime.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

ENV_NAMES = (
    "HIP_PATH", "ROCM_HOME", "ROCM_PATH", "HIP_DEVICE_LIB_PATH",
    "ROCM_SDK_TARGET_FAMILY", "FREETOKEN_ROCM_ARCH", "PYTORCH_ROCM_ARCH",
    "TVM_FFI_ROCM_ARCH_LIST", "TRITON_OVERRIDE_ARCH", "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF", "TVM_FFI_CACHE_DIR", "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR", "FREETOKEN_FUSED_COPY", "FT_GGUF_BACKEND",
)
PACKAGE_NAMES = {
    "freetoken", "torch", "torchvision", "torchaudio", "triton",
    "triton-windows", "pytorch-triton-rocm", "apache-tvm-ffi", "flashlib",
    "gguf", "transformers", "numpy", "safetensors", "flash-linear-attention",
    "setuptools", "ninja", "packaging",
}
TORCH_PROBE = r'''
import json
try:
    import torch
    info = {
        "torch_version": str(torch.__version__),
        "torch_file": str(torch.__file__),
        "hip_build_version": getattr(torch.version, "hip", None),
        "cuda_build_version": getattr(torch.version, "cuda", None),
        "gpu_available": bool(torch.cuda.is_available()),
        "devices": [],
    }
    if info["gpu_available"]:
        info["device_count"] = torch.cuda.device_count()
        for index in range(min(info["device_count"], 8)):
            prop = torch.cuda.get_device_properties(index)
            info["devices"].append({
                "index": index, "name": prop.name,
                "gcn_arch_name": getattr(prop, "gcnArchName", None),
                "vram_bytes": prop.total_memory,
            })
    print("FREETOKEN_ENV_JSON=" + json.dumps(info))
except Exception as exc:
    print("FREETOKEN_ENV_JSON=" + json.dumps({
        "error_type": type(exc).__name__, "error": str(exc)
    }))
    raise SystemExit(1)
'''


def normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def relevant_package(value: str) -> bool:
    name = normalize_name(value)
    return name in PACKAGE_NAMES or name.startswith(("rocm-", "-rocm-", "amd-"))


def redact_home(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact_home(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_home(item) for item in value]
    if isinstance(value, str):
        home = str(Path.home())
        if home not in ("/", "\\", "."):
            flags = re.IGNORECASE if os.name == "nt" else 0
            for variant in {home, home.replace("\\", "/")}:
                value = re.sub(re.escape(variant), "<HOME>", value, flags=flags)
    return value


def run_command(argv: list[str], timeout: int = 15) -> dict[str, Any]:
    try:
        env = dict(os.environ)
        env["GIT_OPTIONAL_LOCKS"] = "0"
        proc = subprocess.run(argv, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              check=False, env=env)
        return {"returncode": proc.returncode,
                "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {exc}"}


def git_inventory(repo: Path) -> dict[str, Any]:
    git = shutil.which("git")
    if git is None:
        return {"available": False}
    prefix = [git, "--no-optional-locks", "-C", str(repo)]
    head = run_command(prefix + ["rev-parse", "--verify", "HEAD"])
    if head.get("returncode") != 0:
        return {"available": True, "head_unavailable": True, "detail": head}
    branch = run_command(prefix + ["symbolic-ref", "--quiet", "--short", "HEAD"])
    status = run_command(prefix + ["status", "--porcelain=v1", "--untracked-files=normal"])
    rows = status.get("stdout", "").splitlines()
    return {
        "available": True, "head_sha": head["stdout"],
        "branch": branch.get("stdout") or None,
        "detached_head": branch.get("returncode") == 1,
        "dirty": bool(rows) if status.get("returncode") == 0 else None,
        "status_entry_count": len(rows) if status.get("returncode") == 0 else None,
        "status_error": status.get("stderr") or status.get("error"),
        "note": "No fetch; remote URLs, diffs and file names are not collected.",
    }


def ram_bytes() -> int | None:
    try:
        if sys.platform == "linux":
            return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        "total_phys", "avail_phys", "total_page", "avail_page",
                        "total_virtual", "avail_virtual", "avail_extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            fn = ctypes.windll.kernel32.GlobalMemoryStatusEx
            fn.argtypes, fn.restype = [ctypes.POINTER(MemoryStatus)], wintypes.BOOL
            if fn(ctypes.byref(status)):
                return int(status.total_phys)
    except (OSError, ValueError, AttributeError):
        return None
    return None


def package_inventory() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "")
        if name and relevant_package(name):
            result.setdefault(name, []).append(dist.version)
    return dict(sorted(result.items(), key=lambda pair: pair[0].lower()))


def torch_inventory() -> dict[str, Any]:
    result = run_command([sys.executable, "-c", TORCH_PROBE], timeout=45)
    prefix = "FREETOKEN_ENV_JSON="
    for line in reversed(result.get("stdout", "").splitlines()):
        if line.startswith(prefix):
            try:
                payload = json.loads(line[len(prefix):])
                payload["probe_returncode"] = result.get("returncode")
                return payload
            except json.JSONDecodeError:
                break
    return {"probe_failed": True, "returncode": result.get("returncode"),
            "error": result.get("error"),
            "stderr_tail": result.get("stderr", "")[-2000:]}


def collect(repo: Path, label: str, probe_torch: bool) -> dict[str, Any]:
    return redact_home({
        "schema_version": 1, "label": label,
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "system": {"os": platform.system(), "release": platform.release(),
                   "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                   "physical_ram_bytes": ram_bytes()},
        "python": {"version": platform.python_version(), "executable": sys.executable,
                   "prefix": sys.prefix, "base_prefix": sys.base_prefix},
        "git": git_inventory(repo), "packages": package_inventory(),
        "environment": {key: os.environ[key] for key in ENV_NAMES if key in os.environ},
        "tool_paths": {name: shutil.which(name) for name in ("hipcc", "clang", "nvcc", "uv")},
        "torch_probe": torch_inventory() if probe_torch else {"not_run": True},
        "limitations": [
            "Driver version and loaded HIP DLL/SO paths are not measured.",
            "Package/build versions do not prove the runtime libraries actually loaded.",
            "No FreeToken tests, model generation or benchmarks were run.",
            "Review this report before sharing; home paths are redacted, not every possible local path.",
        ],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--out", type=Path, help="Create a NEW JSON file; refuses overwrite.")
    parser.add_argument("--probe-torch", action="store_true",
                        help="Import torch in a subprocess and enumerate GPUs (may initialize HIP).")
    args = parser.parse_args()
    if not args.repo.is_dir():
        parser.error("--repo must point to an existing directory")
    if args.out is not None and args.out.exists():
        parser.error("--out already exists; choose a new file name")
    if args.probe_torch:
        print("Optional probe: PyTorch may initialize the GPU runtime; no model is loaded.",
              file=sys.stderr)
    payload = json.dumps(collect(args.repo, args.label, args.probe_torch), indent=2,
                         ensure_ascii=False) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        try:
            with args.out.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        except OSError as exc:
            parser.error(str(exc))
        print(f"Created report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
