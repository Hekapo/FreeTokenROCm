"""Source-loaded preparation tests: no FreeToken install, GPU or native compiler."""

import importlib.util
import os
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARCH_VARS = ("TVM_FFI_ROCM_ARCH_LIST", "PYTORCH_ROCM_ARCH", "TRITON_OVERRIDE_ARCH",
             "ROCM_SDK_TARGET_FAMILY")


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def toolchain(monkeypatch):
    module = load_source("prep_toolchain", "python/freetoken/kernel/_toolchain.py")
    module.os = SimpleNamespace(environ={}, path=os.path)
    for name in ("freetoken.gpu_select", "_rocm_sdk_core"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    cuda = SimpleNamespace(
        is_available=Mock(return_value=True), current_device=Mock(return_value=1),
        get_device_properties=Mock(side_effect=lambda i: SimpleNamespace(
            gcnArchName={0: "gfx1036", 1: "gfx1201:xnack-"}[i])),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, version=SimpleNamespace(hip="test-hip", cuda=None)))
    return module, cuda


@pytest.mark.parametrize("assigned", [0, 1])
def test_arch_uses_published_assignment(toolchain, monkeypatch, assigned):
    module, cuda = toolchain
    monkeypatch.setitem(sys.modules, "freetoken.gpu_select",
                        SimpleNamespace(assigned_visible_gpu=lambda: assigned))
    module._ensure_rocm_arch()
    cuda.get_device_properties.assert_called_once_with(assigned)
    cuda.current_device.assert_not_called()
    expected = "gfx1201" if assigned == 1 else "gfx1036"
    assert all(module.os.environ[name] == expected for name in ARCH_VARS)


@pytest.mark.parametrize("published", [False, True])
def test_arch_falls_back_to_current_not_zero(toolchain, monkeypatch, published):
    module, cuda = toolchain
    if published:
        monkeypatch.setitem(sys.modules, "freetoken.gpu_select",
                            SimpleNamespace(assigned_visible_gpu=lambda: None))
    module._ensure_rocm_arch()
    cuda.get_device_properties.assert_called_once_with(1)
    assert module.os.environ["PYTORCH_ROCM_ARCH"] == "gfx1201"


def test_arch_preserves_explicit_override(toolchain):
    module, _ = toolchain
    module.os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100;gfx1201"
    module._ensure_rocm_arch()
    assert module.os.environ["PYTORCH_ROCM_ARCH"] == "gfx1100;gfx1201"
    assert module.os.environ["TVM_FFI_ROCM_ARCH_LIST"] == "gfx1201"


def test_complete_arch_override_does_not_import_torch(toolchain, monkeypatch):
    module, _ = toolchain
    module.os.environ.update(dict.fromkeys(ARCH_VARS, "gfx1201"))
    monkeypatch.setitem(sys.modules, "torch", None)
    module._ensure_rocm_arch()


def test_unavailable_gpu_does_not_guess_arch(toolchain):
    module, cuda = toolchain
    cuda.is_available.return_value = False
    module._ensure_rocm_arch()
    assert module.os.environ == {}
    cuda.get_device_properties.assert_not_called()


def test_failed_assignment_does_not_fall_back_to_gpu_zero(toolchain, monkeypatch):
    module, cuda = toolchain
    monkeypatch.setitem(sys.modules, "freetoken.gpu_select", SimpleNamespace(
        assigned_visible_gpu=Mock(side_effect=RuntimeError("UUID unavailable"))))
    module._ensure_rocm_arch()
    cuda.get_device_properties.assert_not_called()
    assert not module.os.environ


def test_sdk_detection_remains_standalone(toolchain, monkeypatch, tmp_path):
    module, _ = toolchain
    sdk = tmp_path / "sdk"
    compiler = sdk / "lib/llvm/bin/clang"
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    monkeypatch.setitem(sys.modules, "freetoken", None)
    monkeypatch.setitem(sys.modules, "_rocm_sdk_core",
                        SimpleNamespace(__file__=str(sdk / "__init__.py")))
    assert module.ensure_rocm_env() == str(sdk)
    assert module.os.environ["HIP_PATH"] == str(sdk)
    assert module.os.environ["ROCM_HOME"] == str(sdk)
    assert "ROCM_PATH" not in module.os.environ
    assert module.os.environ["PYTORCH_ROCM_ARCH"] == "gfx1201"


def test_cuda_sdk_environment_is_unchanged(toolchain, monkeypatch):
    module, cuda = toolchain
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, version=SimpleNamespace(hip=None)))
    assert module.ensure_rocm_env() is None
    assert not module.os.environ
    cuda.is_available.assert_not_called()


@pytest.fixture
def kernel_utils(monkeypatch, tmp_path):
    module = load_source("prep_kernel_utils", "python/freetoken/kernel/utils.py")
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    env = {"XDG_CACHE_HOME": str(tmp_path / "cache")}
    module.os = SimpleNamespace(name="posix", environ=env, fsencode=os.fsencode)
    return module


def make_runtime(root, name="libamdhip64.so.7", content=b"fake runtime"):
    path = root / "lib" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def link_from(flags):
    return pathlib.Path(next(x[2:] for x in flags if x.startswith("-L"))) / "libamdhip64.so"


def test_different_sdks_never_share_compatibility_alias(kernel_utils, tmp_path):
    a = make_runtime(tmp_path / "a", content=b"A")
    b = make_runtime(tmp_path / "b", content=b"B")
    kernel_utils._rocm_candidates = lambda: [a.parent.parent]
    first = link_from(kernel_utils._rocm_link_flags())
    kernel_utils._rocm_link_flags.cache_clear()
    kernel_utils._rocm_candidates = lambda: [b.parent.parent]
    second = link_from(kernel_utils._rocm_link_flags())
    assert first != second
    assert first.resolve() == a.resolve()
    assert second.resolve() == b.resolve()


def test_in_place_runtime_change_uses_a_new_namespace(kernel_utils, tmp_path):
    runtime = make_runtime(tmp_path / "sdk", content=b"version-A")
    kernel_utils._rocm_candidates = lambda: [runtime.parent.parent]
    first = link_from(kernel_utils._rocm_link_flags())
    runtime.write_bytes(b"version-B")  # same path and size; test a fresh process's lookup
    kernel_utils._rocm_link_flags.cache_clear()
    second = link_from(kernel_utils._rocm_link_flags())
    assert first != second
    assert first.is_symlink() and second.is_symlink()


def test_existing_legacy_alias_is_not_overwritten(kernel_utils, tmp_path):
    old = make_runtime(tmp_path / "old", content=b"old")
    new = make_runtime(tmp_path / "new", content=b"new")
    legacy = tmp_path / "cache/freetoken/rocm-lib/libamdhip64.so"
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to(old)
    kernel_utils._rocm_candidates = lambda: [new.parent.parent]
    assert link_from(kernel_utils._rocm_link_flags()).resolve() == new.resolve()
    assert legacy.resolve() == old.resolve()


def test_ambiguous_runtimes_fail_instead_of_sorting_versions(kernel_utils, tmp_path):
    sdk = tmp_path / "sdk"
    make_runtime(sdk, "libamdhip64.so.9", b"9")
    make_runtime(sdk, "libamdhip64.so.10", b"10")
    kernel_utils._rocm_candidates = lambda: [sdk]
    with pytest.raises(RuntimeError, match="Multiple HIP runtimes"):
        kernel_utils._rocm_link_flags()


def test_versioned_symlink_aliases_are_deduplicated(kernel_utils, tmp_path):
    runtime = make_runtime(tmp_path / "sdk", "libamdhip64.so.7.2")
    (runtime.parent / "libamdhip64.so.7").symlink_to(runtime)
    kernel_utils._rocm_candidates = lambda: [runtime.parent.parent]
    assert link_from(kernel_utils._rocm_link_flags()).resolve() == runtime.resolve()


def test_authoritative_unversioned_sdk_alias_wins(kernel_utils, tmp_path):
    sdk = tmp_path / "sdk"
    runtime = make_runtime(sdk, "libamdhip64.so.7")
    make_runtime(sdk, "libamdhip64.so.9", b"unused")
    (runtime.parent / "libamdhip64.so").symlink_to(runtime)
    kernel_utils._rocm_candidates = lambda: [sdk]
    assert kernel_utils._rocm_link_flags() == [f"-L{sdk / 'lib'}", f"-Wl,-rpath,{sdk / 'lib'}"]


@pytest.mark.parametrize("replacement", ["file", "wrong_link", "dangling_link"])
def test_wrong_namespaced_cache_entry_is_not_overwritten(kernel_utils, tmp_path, replacement):
    runtime = make_runtime(tmp_path / "sdk")
    directory = kernel_utils._rocm_compat_link_dir(runtime)
    link = directory / "libamdhip64.so"
    link.unlink()
    other = tmp_path / "other.so"
    if replacement == "file":
        link.write_bytes(b"do not overwrite")
    else:
        if replacement == "wrong_link":
            other.touch()
        link.symlink_to(other)
    original_target = link.readlink() if replacement != "file" else None
    with pytest.raises((RuntimeError, FileNotFoundError)):
        kernel_utils._rocm_compat_link_dir(runtime)
    if replacement == "file":
        assert link.read_bytes() == b"do not overwrite"
    else:
        assert link.is_symlink()
        assert link.readlink() == original_target


def test_repeated_link_creation_is_idempotent(kernel_utils, tmp_path):
    runtime = make_runtime(tmp_path / "sdk")
    assert kernel_utils._rocm_compat_link_dir(runtime) == kernel_utils._rocm_compat_link_dir(runtime)


def test_windows_import_library_path_is_preserved(kernel_utils, tmp_path):
    sdk = tmp_path / "sdk"
    make_runtime(sdk, "amdhip64.lib")
    kernel_utils.os.name = "nt"
    kernel_utils._rocm_candidates = lambda: [sdk]
    assert kernel_utils._rocm_link_flags() == [f"/LIBPATH:{sdk / 'lib'}", "amdhip64.lib"]


@pytest.mark.parametrize("changed_template", [False, True])
def test_tvm_guard_restores_original_and_rejects_unknown_template(kernel_utils, monkeypatch, changed_template):
    original = Mock(return_value="new unknown flags" if changed_template else "-Xcompiler /std:c++17 /O2")
    extension = SimpleNamespace(_generate_ninja_build=original)
    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tvm_ffi.cpp", SimpleNamespace(extension=extension))
    kernel_utils.os.name = "nt"
    try:
        with kernel_utils._windows_hip_tvm_ffi_flags(True):
            if changed_template:
                with pytest.raises(RuntimeError, match="refusing an unverified"):
                    extension._generate_ninja_build()
            else:
                assert extension._generate_ninja_build() == "-std=c++17 -O2"
            raise LookupError("simulate caller failure")
    except LookupError:
        pass
    assert extension._generate_ninja_build is original


def test_parallel_link_creation_agrees_on_one_runtime(kernel_utils, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    runtime = make_runtime(tmp_path / "sdk")
    with ThreadPoolExecutor(max_workers=4) as pool:
        directories = list(pool.map(lambda _: kernel_utils._rocm_compat_link_dir(runtime), range(12)))
    assert len(set(directories)) == 1
    assert (directories[0] / "libamdhip64.so").resolve() == runtime.resolve()
