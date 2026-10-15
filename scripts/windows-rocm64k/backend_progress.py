"""Opt-in backend stage records and CPython watchdog stack dumps; no GPU queries."""
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import re
import threading


HOOKS = {
    'normal_loop': ('self',), 'overlap_loop': ('self', 'last_data'),
    'run_when_idle': ('self',),
    '_recv_msg_single_rank': ('self', 'blocking'), '_process_one_msg': ('self', 'msg'),
    '_schedule_next_batch': ('self',), '_restore_linear_states': ('self', 'batch'),
    '_forward': ('self', 'forward_input'), '_process_last_data': ('self', 'last_data'),
    '_reply_tokenizer_rank0': ('self', 'reply'), '_execute_pending_rebuild': ('self',),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    try:
        data = json.dumps(value).encode('utf-8')
        if len(data) > 65536:
            raise ValueError('Backend progress record exceeds its bound')
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ProgressRecorder:
    def __init__(self, directory, identity, *, fault_handler=faulthandler, stack_after_s=10):
        if not 0 < stack_after_s <= 10:
            raise ValueError('Invalid diagnostic stack timeout')
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.latest = self.directory / 'latest.json'
        self.stack_file = self.directory / 'stacks.log'
        self.sink = self.stack_file.open('xb', buffering=0)
        self.identity = dict(identity)
        self.handler, self.timeout = fault_handler, stack_after_s
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.frames = []
        self.history = deque(maxlen=32)
        self.sequence = 0
        self.busy_since = None
        self.last_error = None
        self.armed = False
        self.last_capture = None
        try:
            self._record('bootstrap')
        except BaseException:
            self.sink.close()
            raise

    def _record(self, event):
        at = now()
        phase = self.frames[-1]['phase'] if self.frames else 'idle'
        busy = bool(self.frames) and not self.frames[-1]['waiting']
        if busy and self.busy_since is None:
            self.busy_since = at
            self.handler.dump_traceback_later(self.timeout, repeat=False, file=self.sink, exit=False)
            self.armed = True
        elif not busy:
            if self.armed:
                self.handler.cancel_dump_traceback_later()
                self.armed = False
            self.busy_since = None
        self.sequence += 1
        self.history.append({'event': event, 'phase': phase, 'at': at,
                             'batch': self.frames[-1]['batch'] if self.frames else None})
        atomic_json(self.latest, {'schema': 1, **self.identity, 'sequence': self.sequence,
            'at': at, 'phase': phase, 'busy': busy, 'busy_started_utc': self.busy_since,
            'stack_after_s': self.timeout, 'last_error_type': self.last_error,
            'history': list(self.history)})

    @contextmanager
    def stage(self, phase, *, waiting=False, batch=None):
        with self.lock:
            self.frames.append({'phase': phase, 'waiting': waiting, 'batch': batch})
            try:
                self._record('enter')
            except BaseException:
                self.frames.pop()
                raise
        try:
            yield
        except BaseException as exc:
            with self.lock:
                self.last_error = type(exc).__name__
            raise
        finally:
            with self.lock:
                # Keep the exited stage in history even when its parent resumes.
                self.history.append({'event': 'finished', 'phase': phase, 'at': now(), 'batch': batch})
                self.frames.pop()
                self._record('exit')

    def capture_requested(self):
        path = self.directory / 'capture-request.json'
        try:
            if path.stat().st_size > 4096:
                return
            request = json.loads(path.read_bytes())
        except (OSError, ValueError):
            return
        if (not isinstance(request, dict) or request.get('schema') != 1
                or not isinstance(request.get('request_id'), str)
                or not re.fullmatch('[a-f0-9]{32}', request['request_id'])
                or any(request.get(key) != self.identity[key] for key in ('pid', 'creation_filetime'))):
            return
        with self.lock:
            if request['request_id'] == self.last_capture:
                return
            self.sink.write(f"\nCapture {request['request_id']} at {now()}\n".encode('ascii'))
            self.handler.dump_traceback(file=self.sink, all_threads=True)
            atomic_json(self.directory / 'capture-ack.json', {'schema': 1, **self.identity,
                'request_id': request['request_id'], 'at': now()})
            self.last_capture = request['request_id']

    def _poll(self):
        while not self.stop.wait(0.1):
            try:
                self.capture_requested()
            except OSError:
                # A failed optional capture cannot hold up worker cleanup.
                pass

    def __enter__(self):
        self.thread = threading.Thread(target=self._poll, name='backend-stack-capture', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
        with self.lock:
            if self.armed:
                self.handler.cancel_dump_traceback_later()
                self.armed = False
            self.sink.close()


def batch_summary(batch):
    if batch is None:
        return None
    phase = getattr(batch, 'phase', None)
    summary = {'phase': phase if phase in ('prefill', 'decode') else 'unknown', 'requests': []}
    for req in batch.reqs[:8]:
        summary['requests'].append({key: value for key in ('uid', 'cached_len', 'device_len')
                                   if type(value := getattr(req, key, None)) is int})
    return summary


@contextmanager
def instrument_scheduler(cls, recorder):
    originals = {name: getattr(cls, name) for name in HOOKS}
    if any(tuple(inspect.signature(function).parameters) != HOOKS[name]
           or getattr(function, '_managed_progress', False) for name, function in originals.items()):
        raise RuntimeError('Unreviewed scheduler progress hook contract')
    local_names = set(cls.__dict__)

    def wrap(name, original):
        @wraps(original)
        def run(instance, *args, **kwargs):
            bound = inspect.signature(original).bind(instance, *args, **kwargs)
            values = bound.arguments
            data = values.get('last_data')
            if name == '_process_last_data' and data is None:
                return original(instance, *args, **kwargs)
            batch = values.get('batch')
            if batch is None:
                batch = getattr(values.get('forward_input'), 'batch', None)
            if batch is None and name == '_process_last_data':
                batch = data[0].batch
            waiting = name == '_recv_msg_single_rank' and values.get('blocking', False)
            # Returning from _forward means host submission, not GPU completion.
            phase = {'_forward': 'forward_submit', '_process_last_data': 'gpu_result_wait',
                     'run_when_idle': 'idle_integrity'}.get(name, name)
            with recorder.stage(phase, waiting=waiting, batch=batch_summary(batch)):
                return original(instance, *args, **kwargs)
        run._managed_progress = True
        return run

    try:
        for name, original in originals.items():
            setattr(cls, name, wrap(name, original))
        yield
    finally:
        for name, original in originals.items():
            if name in local_names:
                setattr(cls, name, original)
            else:
                delattr(cls, name)


def run_backend(worker, args, ack_queue):
    directory = os.environ.get('FT_BACKEND_PROGRESS_DIR')
    if not directory:
        return worker(args, ack_queue)
    import multiprocessing as mp
    if (args.tp_info.size != 1 or args.tp_info.rank != 0
            or mp.current_process().name != 'freetoken-TP0-scheduler'):
        raise RuntimeError('Progress diagnostics require the managed backend worker')
    from backend_cpu_affinity import WindowsCpuAffinity
    from freetoken.gpu_select import set_assigned_gpu
    targets = args.gpu_assigned or args.gpu or tuple(str(rank) for rank in range(args.tp_info.size))
    # The original worker repeats this idempotent publication before CUDA init.
    set_assigned_gpu(targets[args.tp_info.rank])
    from freetoken.scheduler import Scheduler
    identity = {'pid': os.getpid(), 'creation_filetime': WindowsCpuAffinity().created()}
    with ProgressRecorder(directory, identity) as recorder, instrument_scheduler(Scheduler, recorder):
        return worker(args, ack_queue)
