"""Backend CPU placement for the managed Windows 64K launcher."""
from __future__ import annotations

import argparse
import ctypes as C
from datetime import datetime, timezone
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys


MODES = ('windows', 'one_per_core')
_original_scheduler = None
_startup_error = None
_mode = None
_output = None
_gpu_initialized = None


def select_one_per_core(core_masks: list[int], allowed_mask: int) -> int:
    if type(allowed_mask) is not int or not 0 < allowed_mask < 1 << 64 or not core_masks:
        raise ValueError('Invalid inherited CPU affinity')
    covered = selected = 0
    for mask in core_masks:
        if type(mask) is not int or not 0 < mask < 1 << 64 or covered & mask:
            raise ValueError('Invalid or overlapping physical core masks')
        covered |= mask
        eligible = allowed_mask & mask
        selected |= eligible & -eligible
    if allowed_mask & ~covered or not selected:
        raise ValueError('CPU topology does not cover the inherited affinity')
    return selected


class FileTime(C.Structure):
    _fields_ = [('low', C.c_uint32), ('high', C.c_uint32)]


class CacheInfo(C.Structure):
    _fields_ = [('level', C.c_ubyte), ('associativity', C.c_ubyte), ('line_size', C.c_uint16),
                ('size', C.c_uint32), ('type', C.c_int32)]


class ProcessorDetail(C.Union):
    _fields_ = [('core_flags', C.c_ubyte), ('numa_node', C.c_uint32), ('cache', CacheInfo),
                ('reserved', C.c_uint64 * 2)]


class ProcessorInfo(C.Structure):
    _fields_ = [('mask', C.c_size_t), ('relationship', C.c_int32), ('detail', ProcessorDetail)]


def require(value):
    if not value:
        raise C.WinError(C.get_last_error())


class WindowsCpuAffinity:
    def __init__(self):
        if sys.platform != 'win32' or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError('Backend CPU affinity requires 64-bit Windows')
        self.lib = C.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'GetCurrentProcess': ([], C.c_void_p),
            'GetProcessAffinityMask': ([C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)], C.c_int),
            'SetProcessAffinityMask': ([C.c_void_p, C.c_size_t], C.c_int),
            'GetProcessTimes': ([C.c_void_p] + [C.POINTER(FileTime)] * 4, C.c_int),
            'GetActiveProcessorGroupCount': ([], C.c_uint16),
            'GetLogicalProcessorInformation': ([C.c_void_p, C.POINTER(C.c_uint32)], C.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result
        self.handle = self.lib.GetCurrentProcess()

    def masks(self):
        process, system = C.c_size_t(), C.c_size_t()
        require(self.lib.GetProcessAffinityMask(self.handle, C.byref(process), C.byref(system)))
        return process.value, system.value

    def set_mask(self, mask):
        require(self.lib.SetProcessAffinityMask(self.handle, mask))

    def created(self):
        times = [FileTime() for _ in range(4)]
        require(self.lib.GetProcessTimes(self.handle, *(C.byref(value) for value in times)))
        return times[0].low | (times[0].high << 32)

    def topology(self):
        groups = self.lib.GetActiveProcessorGroupCount()
        if groups != 1:
            return {'group_count': groups, 'core_masks': []}
        if C.sizeof(ProcessorInfo) != 32:
            raise RuntimeError('Unexpected Windows processor structure')
        size = C.c_uint32()
        if self.lib.GetLogicalProcessorInformation(None, C.byref(size)) or C.get_last_error() != 122:
            raise RuntimeError('Unexpected CPU topology size response')
        if not 32 <= size.value <= 65536 or size.value % 32:
            raise RuntimeError('Invalid CPU topology buffer size')
        buffer = C.create_string_buffer(size.value)
        require(self.lib.GetLogicalProcessorInformation(buffer, C.byref(size)))
        records = C.cast(buffer, C.POINTER(ProcessorInfo))
        cores = [int(records[i].mask) for i in range(size.value // 32) if records[i].relationship == 0]
        return {'group_count': groups, 'core_masks': cores}


def probe_policy(mode: str, native=None) -> dict:
    if mode not in MODES:
        raise ValueError('Unknown backend CPU affinity mode')
    native = WindowsCpuAffinity() if native is None else native
    original, system = native.masks()
    topology = native.topology()
    if mode == 'one_per_core':
        if topology['group_count'] != 1:
            raise RuntimeError('one_per_core supports one Windows processor group')
        selected = select_one_per_core(topology['core_masks'], original)
        if selected & system != selected:
            raise RuntimeError('Selected CPUs are outside the system affinity')
    else:
        selected = original
    return {'mode': mode, 'original_mask': original, 'applied_mask': selected,
            'system_mask': system, 'group_count': topology['group_count'],
            'core_masks': topology['core_masks'], 'logical_processors': selected.bit_count(),
            'process_affinity_changed': False, 'gpu_runtime_loaded': 'torch' in sys.modules or 'triton' in sys.modules}


def write_receipt(output: Path, record: dict) -> None:
    row = dict(record, at=datetime.now(timezone.utc).isoformat())
    with Path(output).open('a', encoding='utf-8') as sink:
        sink.write(json.dumps(row) + '\n')


def run_with_affinity(worker, args, ack_queue, mode: str, output: Path, *, native=None,
                      gpu_initialized=None):
    native = WindowsCpuAffinity() if native is None else native
    if gpu_initialized is None or gpu_initialized():
        raise RuntimeError('CPU affinity must be applied before GPU initialization')
    record = probe_policy(mode, native)
    record.update(pid=os.getpid(), parent_pid=os.getppid(), creation_filetime=native.created(),
                  role='backend', process_name=mp.current_process().name, event='applied',
                  before_gpu_initialization=True)
    original, selected = record['original_mask'], record['applied_mask']
    attempted = written = False
    try:
        if selected != original:
            attempted = True
            native.set_mask(selected)
        if native.masks()[0] != selected:
            raise RuntimeError('Backend CPU affinity readback differs from the requested mask')
        record['process_affinity_changed'] = selected != original
        write_receipt(output, record)
        written = True
        return worker(args, ack_queue)
    finally:
        if attempted:
            native.set_mask(original)
            if native.masks()[0] != original:
                raise RuntimeError('Backend CPU affinity restoration failed')
            if written:
                write_receipt(output, dict(record, event='restored', applied_mask=original))


def scheduler_worker(args, ack_queue):
    if (_original_scheduler is None or args.tp_info.size != 1 or args.tp_info.rank != 0 or
            mp.current_process().name != 'freetoken-TP0-scheduler'):
        raise RuntimeError('CPU affinity requires the managed single-rank scheduler worker')
    started = False

    def start_worker(worker_args, worker_ack):
        nonlocal started
        started = True
        return _original_scheduler(worker_args, worker_ack)

    try:
        return run_with_affinity(start_worker, args, ack_queue, _mode, _output,
                                 gpu_initialized=_gpu_initialized)
    except BaseException as exc:
        if not started:
            _startup_error(ack_queue, exc)
        raise


def install_scheduler_hook(launch, mode: str, output: Path, gpu_initialized) -> None:
    global _original_scheduler, _startup_error, _mode, _output, _gpu_initialized
    if mode not in MODES or not str(output):
        raise ValueError('Invalid managed backend CPU affinity configuration')
    if launch._run_scheduler is scheduler_worker:
        if (_mode, _output) != (mode, Path(output)):
            raise RuntimeError('Backend CPU affinity hook configuration changed')
        return
    original = launch._run_scheduler
    if (_original_scheduler is not None or original.__module__ != 'freetoken.server.launch' or
            original.__name__ != '_run_scheduler' or
            tuple(inspect.signature(original).parameters) != ('args', 'ack_queue')):
        raise RuntimeError('Unreviewed scheduler bootstrap or existing hook')
    _original_scheduler, _startup_error = original, launch._report_startup_error
    _mode, _output, _gpu_initialized = mode, Path(output), gpu_initialized
    # A module-level target stays importable when Windows spawn reloads the entry point.
    launch._run_scheduler = scheduler_worker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', choices=MODES, required=True)
    print(json.dumps(probe_policy(parser.parse_args().probe)))
