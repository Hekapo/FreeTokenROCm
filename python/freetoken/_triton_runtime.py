"""Optional process-start caching of the effective Triton AMD buffer-ops knob."""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import os
from typing import NamedTuple


class AmdBufferOpsInit(NamedTuple):
    status: str
    value: bool | None = None
    triton_version: str | None = None


def initialize_amd_buffer_ops() -> AmdBufferOpsInit:
    return _initialize_amd_buffer_ops(os.getpid())


@functools.cache
def _initialize_amd_buffer_ops(process_id: int) -> AmdBufferOpsInit:
    flag = os.getenv('FREETOKEN_TRITON_CACHE_AMD_KNOB', '').strip().lower()
    if flag not in {'1', 'true', 'yes', 'on'}:
        return AmdBufferOpsInit('disabled')

    # Torch preloads its matching SDK DLLs before Triton is linked on Windows.
    torch = importlib.import_module('torch')
    if not getattr(torch.version, 'hip', None):
        return AmdBufferOpsInit('not-rocm')
    triton = importlib.import_module('triton')
    version = getattr(triton, '__version__', None)
    if version != '3.8.0':
        return AmdBufferOpsInit('unsupported-version', triton_version=version)

    knobs = getattr(triton, 'knobs', None)
    amd = getattr(knobs, 'amd', None)
    descriptor = inspect.getattr_static(type(amd), 'use_buffer_ops', None)
    if (type(descriptor) is not getattr(knobs, 'env_bool', None)
            or getattr(descriptor, 'key', None) != 'AMDGCN_USE_BUFFER_OPS'
            or getattr(descriptor, 'name', None) != 'use_buffer_ops'
            or not hasattr(amd, '__dict__')):
        return AmdBufferOpsInit('unsupported-knob', triton_version=version)

    override = vars(amd).get('use_buffer_ops')
    if type(override) is bool:
        return AmdBufferOpsInit('already-cached', override, version)
    value = amd.use_buffer_ops
    if type(value) is not bool:
        return AmdBufferOpsInit('invalid-value', triton_version=version)
    amd.use_buffer_ops = value
    if amd.use_buffer_ops is not value:
        raise RuntimeError('Triton AMD buffer-ops value changed during initialization')
    logging.getLogger(__name__).info(
        'Triton AMD buffer ops cached at process start: value=%s version=%s pid=%s',
        value, version, process_id,
    )
    return AmdBufferOpsInit('cached', value, version)
