"""Compile the range-analysis reproducers without initializing a GPU or model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time


def reproducers():
    import triton
    import triton.language as tl

    @triton.jit
    def histogram_mask(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
        offsets = tl.arange(0, M)
        x = tl.load(x_ptr + offsets)
        hist = tl.histogram(x, N)
        bins = tl.arange(0, N)
        tl.store(out_ptr + bins, hist, mask=hist > 0)

    @triton.jit
    def dynamic_loop_mask(out_ptr):
        pid = tl.program_id(0)
        upper = (pid < 4).to(tl.int32)
        count = 0
        for _ in range(upper):
            count += 1
        tl.store(out_ptr + pid, 42, mask=count == 1)

    @triton.jit
    def nested_loop_value(out_ptr):
        pid = tl.program_id(0)
        upper = 1
        count = 0
        for _ in range(4):
            count = 0
            for _ in range(upper):
                count += 1
            upper += 1
        tl.store(out_ptr + pid, (count != 4).to(tl.int32))

    return [
        ('histogram-1024-64', histogram_mask, {'x_ptr': '*i32', 'out_ptr': '*i32'},
         {'M': 1024, 'N': 64}),
        ('histogram-4096-128', histogram_mask, {'x_ptr': '*i32', 'out_ptr': '*i32'},
         {'M': 4096, 'N': 128}),
        ('dynamic-loop', dynamic_loop_mask, {'out_ptr': '*i32'}, {}),
        ('nested-loop', nested_loop_value, {'out_ptr': '*i32'}, {}),
    ]


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    for variable in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        os.environ[variable] = '-1'
    os.environ['TRITON_CACHE_DIR'] = str(output / 'triton-cache')
    os.environ['TMP'] = os.environ['TEMP'] = str(output / 'temp')
    (output / 'temp').mkdir()

    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from compiler_safety import install_range_safety

    cases = reproducers()
    installed = Path(triton.__file__).parent
    protected = [installed / 'backends/amd/compiler.py', installed / '_C/libtriton.pyd']
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    records = []
    state = None
    for mode in ('original', 'range-safety'):
        if mode == 'range-safety':
            state = install_range_safety()
        for name, function, signature, constants in cases:
            started = time.monotonic()
            source = ASTSource(function, signature=signature, constexprs=constants)
            kernel = triton.compile(source, target=GPUTarget('hip', 'gfx1201', 32),
                                    options={'num_warps': 4, 'num_stages': 2})
            case_output = output / mode / name
            case_output.mkdir(parents=True)
            for kind, content in kernel.asm.items():
                path = case_output / ('kernel.' + kind)
                if isinstance(content, bytes):
                    path.write_bytes(content)
                else:
                    path.write_text(content, encoding='utf-8')
            ttgir = kernel.asm['ttgir']
            stores = [line for line in ttgir.splitlines()
                      if re.search(r'\b(?:tt\.store|amdg\.buffer_store)\b', line)]
            false_masks = re.findall(r'^\s*(%[\w.]+) = arith.constant (?:dense<false>|false)',
                                     ttgir, re.MULTILINE)
            suppressed = [line for line in stores if any(
                re.search(r',\s*' + re.escape(mask) + r'\s*:', line)
                for mask in false_masks)]
            effective = bool(stores) and len(suppressed) < len(stores)
            record = {
                'mode': mode, 'case': name, 'cache_hash': kernel.hash,
                'seconds': round(time.monotonic() - started, 3),
                'shared_bytes': kernel.metadata.shared,
                'lds_within_64k': kernel.metadata.shared <= 65536,
                'output_store_count': len(stores),
                'constant_false_predicate_store_count': len(suppressed),
                'output_stores_effective': effective,
                'loop_comparison_preserved': ('arith.cmpi eq' in ttgir if name == 'dynamic-loop'
                                              else 'arith.cmpi ne' in ttgir if name == 'nested-loop'
                                              else None),
                'ttgir_sha256': hashlib.sha256(ttgir.encode()).hexdigest(),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    by_case = {(row['mode'], row['case']): row for row in records}
    cache_separated = all(by_case[('original', name)]['cache_hash']
                          != by_case[('range-safety', name)]['cache_hash']
                          for name, *_ in cases)
    histograms = ('histogram-1024-64', 'histogram-4096-128')
    histogram_fault_reproduced = all(not by_case[('original', name)]['output_stores_effective']
                                    for name in histograms)
    histogram_predicates_restored = all(by_case[('range-safety', name)]['output_stores_effective']
                                       for name in histograms)
    loops = ('dynamic-loop', 'nested-loop')
    loop_faults_reproduced = all(not by_case[('original', name)]['loop_comparison_preserved']
                                for name in loops)
    loop_comparisons_restored = all(by_case[('range-safety', name)]['loop_comparison_preserved']
                                   for name in loops)
    result = {
        'torch': torch.__version__, 'hip': torch.version.hip, 'compiler_safety': state,
        'cases': records, 'cache_separated': cache_separated,
        'installed_files_unchanged': before == after, 'protected_file_hashes': after,
        'histogram_fault_reproduced': histogram_fault_reproduced,
        'histogram_predicates_restored': histogram_predicates_restored,
        'loop_faults_reproduced': loop_faults_reproduced,
        'loop_comparisons_restored': loop_comparisons_restored,
        'gpu_initialized': torch.cuda.is_initialized(), 'gpu_kernels_launched': False,
        'gpu_numeric_qualified': False, 'model_loaded': False,
    }
    result['passed'] = (cache_separated and before == after
                        and histogram_fault_reproduced and histogram_predicates_restored
                        and loop_faults_reproduced and loop_comparisons_restored
                        and all(row['lds_within_64k'] for row in records)
                        and not result['gpu_initialized'])
    (output / 'qualification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'cases', 'protected_file_hashes', 'compiler_safety'}}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(0 if run(arguments.output.resolve())['passed'] else 2)
