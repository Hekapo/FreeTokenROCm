"""Install the reviewed workaround before a trial worker imports GPU kernels."""

import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys


class KernelLoader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        try:
            from compiler_safety import install_range_safety
            import torch
            import triton
            from stable_triton_knob import pin_stable_amd_knob

            state = install_range_safety()
            state.update(pid=os.getpid(), parent_pid=os.getppid(),
                         before_gpu_initialization=not torch.cuda.is_initialized(),
                         stable_buffer_ops=pin_stable_amd_knob(triton))
            destination = Path(os.environ['FT_ROCM1010_INIT_OUTPUT'])
            with destination.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(state) + '\n')
        except BaseException as error:
            sys.stderr.write('ROCm 10.1 worker bootstrap failed: ' + repr(error) + '\n')
            sys.stderr.flush()
            # Python normally swallows sitecustomize errors; the trial must fail closed.
            os._exit(86)
        self.original.exec_module(module)


class KernelFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'freetoken.kernel':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise RuntimeError('Trial kernel package is unavailable')
        spec.loader = KernelLoader(spec.loader)
        return spec


if os.environ.get('FT_ROCM1010_RANGE_SAFE') == '1':
    sys.meta_path.insert(0, KernelFinder())
