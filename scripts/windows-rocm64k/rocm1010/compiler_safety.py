"""Explicit, experimental compiler qualification for the pinned ROCm 10.1 stack."""

from __future__ import annotations

import ast
import hashlib
import importlib
from importlib import metadata
import inspect
from pathlib import Path
import sys
import textwrap


COMPILER_SHA256 = '50dceba3abd768308bb59ff5e7b85a0d8c47e5ff9fa27558051cd6903eff7da3'
PIPELINE_SHA256 = 'f17d220e76378b81af4b5fba897a85a3966534fea4924f27ef6220e2361f88b7'
_installed_hook = None
_installed_state = None


def without_range_fold(source: str, namespace: dict):
    """Retain the reviewed pipeline while omitting its single unsafe optimization."""
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError('Unreviewed compiler pipeline definition')
    function = tree.body[0]
    if (function.name != 'make_ttgir'
            or len(function.decorator_list) != 1
            or not isinstance(function.decorator_list[0], ast.Name)
            or function.decorator_list[0].id != 'staticmethod'):
        raise RuntimeError('Unreviewed compiler pipeline decorator')
    expected = ast.parse('amd.passes.ttgpuir.add_fold_true_cmpi(pm)').body[0]
    matches = [node for node in function.body
               if ast.dump(node, include_attributes=False)
               == ast.dump(expected, include_attributes=False)]
    all_calls = [node for node in ast.walk(function)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'add_fold_true_cmpi']
    if len(matches) != 1 or len(all_calls) != 1:
        raise RuntimeError('Unreviewed compiler pipeline range-fold call')
    function.body.remove(matches[0])
    function.decorator_list = []
    scope = dict(namespace)
    exec(compile(tree, '<freetoken-rocm1010-range-safety>', 'exec'), scope)
    return scope['make_ttgir']


def install_range_safety() -> dict:
    """Install before GPU initialization; never write to Triton's installed files."""
    global _installed_hook, _installed_state

    if sys.platform != 'win32' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('Compiler qualification requires Windows / Python 3.12')
    torch = importlib.import_module('torch')
    if (torch.__version__ != '2.14.0+rocm10.1.0'
            or metadata.version('triton-windows') != '3.8.0.post29'
            or metadata.version('rocm-sdk-core') != '10.1.0'):
        raise RuntimeError('Unreviewed Torch/ROCm/Triton stack')
    triton = importlib.import_module('triton')
    if triton.__version__ != '3.8.0' or torch.version.hip != '7.16.26385':
        raise RuntimeError('Unreviewed compiler/runtime version')
    existing = triton.knobs.runtime.add_stages_inspection_hook
    if existing is not None:
        if existing is _installed_hook:
            return dict(_installed_state)
        raise RuntimeError('Refusing to replace an existing Triton compiler hook')
    if torch.cuda.is_initialized():
        raise RuntimeError('Install compiler qualification before GPU initialization')

    compiler = importlib.import_module('triton.backends.amd.compiler')
    backend = compiler.HIPBackend
    compiler_path = Path(inspect.getfile(backend))
    if hashlib.sha256(compiler_path.read_bytes()).hexdigest() != COMPILER_SHA256:
        raise RuntimeError('Unreviewed AMD compiler file')
    source = inspect.getsource(backend.make_ttgir)
    if hashlib.sha256(source.encode()).hexdigest() != PIPELINE_SHA256:
        raise RuntimeError('Unreviewed AMD compiler pipeline')
    pipeline = without_range_fold(source, vars(compiler))
    hook_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cache_key = 'freetoken-rocm1010-range-safety:' + COMPILER_SHA256 + ':' + hook_digest

    def hook(self=None, stages=None, options=None, language=None, capability=None):
        if all(arg is None for arg in (self, stages, options, language, capability)):
            return cache_key, hook_digest
        if not isinstance(self, backend):
            return cache_key, hook_digest
        if self.target.arch != 'gfx1201' or self.target.warp_size != 32:
            raise RuntimeError('Unqualified AMD target for range-fold workaround')
        if language == compiler.Language.TRITON:
            if options.arch != 'gfx1201' or 'ttgir' not in stages:
                raise RuntimeError('Unreviewed AMD compiler stage configuration')
            stages['ttgir'] = lambda mod, meta: pipeline(mod, meta, options)
        return cache_key, hook_digest

    state = {
        'mode': 'experimental-skip-range-fold',
        'torch': torch.__version__, 'hip': torch.version.hip,
        'triton_windows': metadata.version('triton-windows'),
        'target': 'gfx1201', 'disabled_pass': 'tritonamdgpu-fold-true-cmpi',
        'compiler_sha256': COMPILER_SHA256, 'pipeline_sha256': PIPELINE_SHA256,
        'hook_sha256': hook_digest, 'cache_key': cache_key,
        'installed_files_modified': False, 'native_compiler_fixed': False,
    }
    triton.knobs.runtime.add_stages_inspection_hook = hook
    _installed_hook, _installed_state = hook, state
    return dict(state)
