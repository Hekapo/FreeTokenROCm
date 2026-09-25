"""Source-loaded HIP depfile rule tests: no FreeToken install, GPU, tvm-ffi or native compiler."""

import importlib.util
import os
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Shape of a build.ninja written by the pinned tvm-ffi 0.1.13 for a Windows HIP JIT module
# (paths shortened); only the compile_cuda command line may change.
PINNED_HIP_NINJA = """ninja_required_version = 1.3
cxx = cl
cxxflags = /std:c++17 /MD /EHsc
nvcc = hipcc
cuda_cflags = -Xcompiler /std:c++17 /O2 -std=c++20 -O3
ldflags = /DLL tvm_ffi.lib amdhip64.lib

rule compile
  command = $cxx /showIncludes $cxxflags -c $in /Fo$out
  deps = msvc

rule compile_cuda
  depfile = $out.d
  deps = gcc
  command = $nvcc $cuda_cflags -c $in -o $out

rule link
  command = $cxx $in /link $ldflags /out:$out

build cuda_0.o: compile_cuda C$:\\cache\\tvm-ffi\\freetoken__index\\cuda.cu
build freetoken__index.dll: link cuda_0.o

default freetoken__index.dll
"""

DEPFILE_COMMAND = "  command = $nvcc -MD -MF $out.d $cuda_cflags -c $in -o $out"


@pytest.fixture
def kernel_utils():
    spec = importlib.util.spec_from_file_location(
        "prep_kernel_utils_depfile", ROOT / "python/freetoken/kernel/utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.os = SimpleNamespace(name="posix", environ={}, fsencode=os.fsencode)
    return module


@pytest.fixture
def fake_extension(monkeypatch):
    def install(generate):
        extension = SimpleNamespace(_generate_ninja_build=generate)
        monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "tvm_ffi.cpp", SimpleNamespace(extension=extension))
        return extension

    return install


def test_pinned_hip_rule_gains_depfile_and_nothing_else_changes(kernel_utils):
    rewritten = kernel_utils._add_hip_depfile(PINNED_HIP_NINJA)
    assert DEPFILE_COMMAND in rewritten
    assert "  command = $nvcc $cuda_cflags -c $in -o $out" not in rewritten
    before, after = PINNED_HIP_NINJA.splitlines(), rewritten.splitlines()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after)
    assert changed == [("  command = $nvcc $cuda_cflags -c $in -o $out", DEPFILE_COMMAND)]


def test_depfile_path_matches_the_declared_depfile(kernel_utils):
    rewritten = kernel_utils._add_hip_depfile(PINNED_HIP_NINJA)
    rule = rewritten.split("rule compile_cuda\n", 1)[1].split("\n\n", 1)[0]
    assert "  depfile = $out.d" in rule
    assert "  deps = gcc" in rule
    assert "-MF $out.d" in rule


def test_rewrite_is_idempotent(kernel_utils):
    once = kernel_utils._add_hip_depfile(PINNED_HIP_NINJA)
    assert kernel_utils._add_hip_depfile(once) == once


def test_build_without_device_sources_is_untouched(kernel_utils):
    cpp_only = PINNED_HIP_NINJA.split("rule compile_cuda", 1)[0]
    assert kernel_utils._add_hip_depfile(cpp_only) == cpp_only


@pytest.mark.parametrize(
    "command",
    [
        "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d $cuda_cflags -c $in -o $out",
        "  command = $nvcc $cuda_cflags -x hip -c $in -o $out",
    ],
)
def test_unknown_compile_rule_is_refused(kernel_utils, command):
    changed = PINNED_HIP_NINJA.replace("  command = $nvcc $cuda_cflags -c $in -o $out", command)
    with pytest.raises(RuntimeError, match="refusing an unverified depfile rewrite"):
        kernel_utils._add_hip_depfile(changed)


def test_disabled_guard_does_not_touch_tvm_ffi(kernel_utils, fake_extension):
    original = Mock(return_value=PINNED_HIP_NINJA)
    extension = fake_extension(original)
    with kernel_utils._hip_depfile_tvm_ffi_rule(False):
        assert extension._generate_ninja_build is original
    original.assert_not_called()


def test_guard_rewrites_and_restores_after_caller_failure(kernel_utils, fake_extension):
    original = Mock(return_value=PINNED_HIP_NINJA)
    extension = fake_extension(original)
    try:
        with kernel_utils._hip_depfile_tvm_ffi_rule(True):
            assert DEPFILE_COMMAND in extension._generate_ninja_build("name", sources=[])
            raise LookupError("simulate caller failure")
    except LookupError:
        pass
    assert extension._generate_ninja_build is original
    original.assert_called_once_with("name", sources=[])


def test_guard_composes_with_windows_flag_guard(kernel_utils, fake_extension):
    original = Mock(return_value=PINNED_HIP_NINJA)
    extension = fake_extension(original)
    kernel_utils.os.name = "nt"
    with kernel_utils._windows_hip_tvm_ffi_flags(True), kernel_utils._hip_depfile_tvm_ffi_rule(True):
        rewritten = extension._generate_ninja_build()
    assert "cuda_cflags = -std=c++17 -O2 -std=c++20 -O3" in rewritten
    assert DEPFILE_COMMAND in rewritten
    assert extension._generate_ninja_build is original
