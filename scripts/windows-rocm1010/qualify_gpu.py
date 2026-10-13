"""Small, explicitly authorized GPU checks; no checkpoint or server is loaded."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def run(output: Path, project_operators: bool, suite: str = 'core') -> dict:
    output.mkdir(parents=True, exist_ok=False)
    for directory in ('triton-cache', 'temp', 'torch-extensions', 'tvm-ffi-cache', 'hip-cache'):
        (output / directory).mkdir()
    os.environ['TRITON_CACHE_DIR'] = str(output / 'triton-cache')
    os.environ['TORCH_EXTENSIONS_DIR'] = str(output / 'torch-extensions')
    os.environ['TVM_FFI_CACHE_DIR'] = str(output / 'tvm-ffi-cache')
    os.environ['HIP_CACHE_DIR'] = str(output / 'hip-cache')
    os.environ['TMP'] = os.environ['TEMP'] = str(output / 'temp')

    import torch
    import triton
    from compiler_safety import install_range_safety
    from qualify_compiler import reproducers

    torch.set_num_threads(1)
    state = install_range_safety()
    properties = torch.cuda.get_device_properties(0)
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'hip' or target.arch != 'gfx1201' or target.warp_size != 32:
        raise RuntimeError('GPU does not match the qualified compilation target')
    print(json.dumps({'phase': 'gpu-initialized', 'name': properties.name,
                      'target': str(target)}), flush=True)
    results = []
    for name, kernel, _, constants in (reproducers() if suite == 'core' else []):
        started = time.monotonic()
        if name.startswith('histogram'):
            m, n = constants['M'], constants['N']
            # Leave empty bins: a correct masked store must retain their sentinel.
            x_cpu = (torch.arange(m, dtype=torch.int32) % (n // 4))
            counts = torch.bincount(x_cpu.long(), minlength=n).to(torch.int32)
            expected = torch.where(counts > 0, counts, -7)
            actual = torch.full((n,), -7, dtype=torch.int32, device='cuda')
            compiled = kernel[(1,)](x_cpu.to('cuda'), actual, **constants,
                                    num_warps=4, num_stages=2)
        else:
            actual = torch.full((8,), -7, dtype=torch.int32, device='cuda')
            expected = (torch.tensor([42] * 4 + [-7] * 4, dtype=torch.int32)
                        if name == 'dynamic-loop' else torch.zeros(8, dtype=torch.int32))
            compiled = kernel[(8,)](actual, num_warps=4, num_stages=2)
        torch.cuda.synchronize()
        observed = actual.cpu()
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        record = {'case': name, 'passed': True, 'shared_bytes': compiled.metadata.shared,
                  'seconds': round(time.monotonic() - started, 3),
                  'output_sha256': hashlib.sha256(observed.numpy().tobytes()).hexdigest()}
        results.append(record)
        print(json.dumps(record), flush=True)

    if suite == 'core':
        torch.manual_seed(619)
        left, right = torch.randn(16, 96), torch.randn(96, 32)
        expected = left @ right
        observed = (left.to('cuda') @ right.to('cuda')).cpu()
        torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)
        results.append({'case': 'torch-fp32-gemm', 'passed': True})
        print(json.dumps(results[-1]), flush=True)

    operator_exit = None
    if project_operators:
        import pytest

        os.environ['FT_TEST_STAGE2_GPU'] = '1'
        selected = [
            'tests/kernels/test_triton_sampling_fallback.py::test_top_p_sampling_returns_a_token_inside_the_nucleus',
            'tests/kernels/test_fla_padding.py::test_gpu_gdn_partial_chunk_and_recurrent_model_geometry',
            'tests/kernels/test_gguf_rocm.py::test_q4_0_dequant_matches_torch_reference',
            'tests/kernels/test_gguf_rocm.py::test_q4_0_matvec_matches_dequantized_torch_reference',
            'tests/kernels/test_gguf_rocm.py::test_q4_0_moe_matvec_routes_experts_and_matches_reference',
        ]
        if suite == 'q4-attention':
            os.environ['FT_TEST_GGUF_GRID_GPU'] = '1'
            selected = [
                'tests/kernels/test_gguf_rocm.py::test_q4_1_moe_grid_boundary_matches_reference_and_split',
                'tests/kernels/test_triton_attention.py::test_qwen35_gguf_attention_prefill_and_decode_matches_reference',
            ]
        operator_exit = int(pytest.main(['-q', '-o', 'addopts=', '-p', 'no:cacheprovider',
                                        '--maxfail=1', '--tb=short',
                                        '--junitxml=' + str(output / 'operators.xml'), *selected]))
        torch.cuda.synchronize()
    result = {
        'torch': torch.__version__, 'hip': torch.version.hip, 'compiler_safety': state,
        'device': properties.name, 'target': str(target), 'checks': results,
        'suite': suite,
        'project_operators_exit': operator_exit,
        'max_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'max_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
        'host_vendor_matrix_qualified': False, 'model_loaded': False,
        'passed': all(record['passed'] for record in results)
                  and operator_exit in (None, 0),
    }
    (output / 'gpu-qualification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'compiler_safety', 'checks'}}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-operators', action='store_true')
    parser.add_argument('--suite', choices=('core', 'q4-attention'), default='core')
    arguments = parser.parse_args()
    raise SystemExit(0 if run(arguments.output.resolve(), arguments.project_operators,
                            arguments.suite)['passed'] else 2)
