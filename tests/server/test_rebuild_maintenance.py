"""Rebuild admission, correlation, and local-send uncertainty without real workers.

A send error or an expired HTTP wait cannot prove that the backend did no work.
Keep the operation active until a matching reply; terminal lifecycle gates stay closed.
Queue tests below inject socket stand-ins, not a live ZMQ transport.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from freetoken.server.accounting import MAINTENANCE_LOCK, AdmissionClosedError
from freetoken.server.api_server import (
    FrontendManager, _mark_backend_failed, _mark_backend_ready, dispatch_rebuild,
)
from freetoken.server.supervisor import (
    BackendHandle,
    LoadProgress,
    run_backend_supervisor,
)


class _FakeState:
    """Stand-in for FrontendManager exposing exactly what dispatch_rebuild / _resolve_rebuild /
    fail_pending_rebuilds read: rebuild_futures, maintenance_state, fatal_error, last_rebuild,
    the event loop (_loop, for cross-thread future resolution), and an async send_one delegating
    to an injected impl (so a test can make the enqueue succeed or raise)."""

    def __init__(
        self, send_impl, *, maintenance_state="serving", fatal_error=None, active_rebuild_id=None
    ):
        self.rebuild_futures: dict = {}
        self._active_rebuild_id = active_rebuild_id
        self.maintenance_state = maintenance_state
        self.fatal_error = fatal_error
        self.last_rebuild = None
        self._loop = None
        self._send_impl = send_impl

    async def send_one(self, msg):
        await self._send_impl(msg)


def _reply(request_id, status, **over):
    base = dict(
        request_id=request_id,
        status=status,
        moe_cache_size=0,
        num_pages=0,
        mamba_slots=0,
        num_swa_pages=0,
        error=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_dispatch_exception_keeps_delivery_unresolved():
    async def boom(_msg):
        raise RuntimeError("zmq push failed")

    state = _FakeState(boom)

    async def _run():
        return await dispatch_rebuild(state, moe_cache_size=8, num_pages=None)

    result = asyncio.run(_run())
    assert result["status"] == "failed"
    assert result["delivery"] == "unknown"
    assert "zmq push failed" in result["error"]
    assert state.maintenance_state == "rebuilding"
    assert state._active_rebuild_id == result["request_id"]
    assert state.rebuild_futures == {}


def test_timeout_stays_rebuilding_then_late_reply_resolves():
    """The HTTP wait timing out deliberately leaves the gate "rebuilding" (the scheduler may
    still be mid-recapture). That is not a dead end: the eventual reply resolves it."""

    async def ok(_msg):
        return None  # enqueue succeeds, but nothing ever resolves the future -> timeout

    state = _FakeState(ok)

    async def _run():
        return await dispatch_rebuild(state, moe_cache_size=8, num_pages=None, timeout=0.01)

    result = asyncio.run(_run())
    assert result["status"] == "timeout"
    assert state.maintenance_state == "rebuilding"  # still gated on purpose
    assert state.rebuild_futures == {}  # cancelled future dropped, not leaked

    # The late reply is what un-wedges it -> definite "serving".
    FrontendManager._resolve_rebuild(state, _reply(result["request_id"], "ok", num_pages=1024))
    assert state.maintenance_state == "serving"
    assert state.last_rebuild["num_pages"] == 1024


def test_resolve_failed_latches_failed():
    state = _FakeState(None, maintenance_state="rebuilding", active_rebuild_id="r1")
    FrontendManager._resolve_rebuild(state, _reply("r1", "failed", error="OOM during recapture"))
    assert state.maintenance_state == "failed"
    assert state.last_rebuild["error"] == "OOM during recapture"


def test_resolve_nonfatal_statuses_keep_serving():
    # ok / busy / rejected / unsupported all leave the prior cache intact -> keep serving.
    for status in ("ok", "busy", "rejected", "unsupported"):
        state = _FakeState(None, maintenance_state="rebuilding", active_rebuild_id="r1")
        FrontendManager._resolve_rebuild(state, _reply("r1", status))
        assert state.maintenance_state == "serving", status


def test_late_reply_cannot_resurrect_a_fatal_latch():
    """A worker crash latched "failed" (the watchdog). A buffered "ok" reply that raced the
    crash must NOT reopen the gate — a dead backend cannot serve — yet it still wakes the
    waiter blocked on that request_id (the reply path) and is recorded for observability."""

    async def _run():
        state = _FakeState(
            None, maintenance_state="failed", fatal_error="scheduler exited", active_rebuild_id="r1"
        )
        # A caller is still parked on this request_id's future (the "late reply" the name
        # promises): _resolve_rebuild must wake it even though the gate stays latched.
        fut = asyncio.get_running_loop().create_future()
        state.rebuild_futures["r1"] = fut

        FrontendManager._resolve_rebuild(state, _reply("r1", "ok", num_pages=2048))

        assert state.maintenance_state == "failed"          # gate stays latched
        assert fut.done() and fut.result()["num_pages"] == 2048  # waiter still woken
        assert state.rebuild_futures == {}                  # future consumed, not leaked
        # The reply is still recorded for observability; the gate stays latched.
        assert state.last_rebuild["num_pages"] == 2048

    asyncio.run(_run())


def test_crash_during_rebuild_latches_failed_via_watchdog():
    """Scheduler-crash path, end to end and cross-thread: a worker dies while a rebuild is in
    flight ("rebuilding"). No reply ever arrives, but the liveness watchdog (on its own thread)
    fires on_failure, which mirrors production _on_failure — it (a) drives the gate to a definite
    "failed" and (b) wakes the in-flight rebuild waiter so dispatch_rebuild returns "failed"
    PROMPTLY, instead of stranding the caller until its full (here 30 s) timeout."""

    class Proc:
        name = "freetoken-TP0-scheduler"

        def __init__(self):
            self._alive = True

        def is_alive(self):
            return self._alive

    async def _run():
        proc = Proc()
        q: "queue.Queue" = queue.Queue()
        q.put("scheduler ready")
        handle = BackendHandle(ack_queue=q, processes=[proc], expected_acks=1)

        async def ok(_msg):
            return None  # enqueue succeeds; the backend then crashes without ever replying

        state = _FakeState(ok, maintenance_state="serving")
        state._loop = asyncio.get_running_loop()
        ready_evt = threading.Event()

        def on_ready():
            state.maintenance_state = "serving"
            ready_evt.set()

        # Production _on_failure: latch failed AND wake any pending rebuild waiter — called here
        # from the supervisor thread, so fail_pending_rebuilds must marshal onto the loop.
        def on_failure(message):
            _mark_backend_failed(state, message)
            FrontendManager.fail_pending_rebuilds(state, message)

        sup = threading.Thread(
            target=run_backend_supervisor,
            args=(handle, LoadProgress(), on_ready),
            kwargs={"on_failure": on_failure, "poll": 0.01},
            daemon=True,
        )
        sup.start()
        # Let the supervisor drain to readiness and enter the post-ready watch loop.
        while not ready_evt.is_set():
            await asyncio.sleep(0.005)
        assert state.maintenance_state == "serving"

        # A rebuild is now in flight: a real pending future, gate latched "rebuilding".
        task = asyncio.create_task(
            dispatch_rebuild(state, moe_cache_size=8, num_pages=None, timeout=30.0)
        )
        await asyncio.sleep(0)
        assert state.rebuild_futures          # a waiter is parked
        assert state.maintenance_state == "rebuilding"

        proc._alive = False  # …and the scheduler crashes mid-rebuild

        # The waiter is woken promptly (well under the 30 s timeout) with a failed result.
        result = await asyncio.wait_for(task, timeout=5.0)
        assert result["status"] == "failed"
        assert "scheduler" in result["error"]
        assert state.maintenance_state == "failed"  # escaped "rebuilding" to a definite state
        assert "scheduler" in state.fatal_error
        assert state.rebuild_futures == {}          # waiter resolved and cleared, not leaked
        sup.join(timeout=1.0)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# The maintenance gate seen from the request side: what a client hits while the server is
# loading or rebuilding, on both the OpenAI generation routes and the rebuild route itself.
# ---------------------------------------------------------------------------


def test_openai_gate_message_is_loading_aware():
    from freetoken.server.openai_api import _maintenance_gate

    assert _maintenance_gate(SimpleNamespace(maintenance_state="serving")) is None
    loading = _maintenance_gate(SimpleNamespace(maintenance_state="loading"))
    assert loading is not None and loading.status_code == 503
    assert b"loading" in loading.body.lower()
    rebuild = _maintenance_gate(SimpleNamespace(maintenance_state="rebuilding"))
    assert rebuild is not None and rebuild.status_code == 503
    assert b"rebuild" in rebuild.body.lower()
    failed = _maintenance_gate(SimpleNamespace(maintenance_state="failed"))
    assert failed is not None and failed.status_code == 503
    # A state object without the attribute defaults to serving (defensive, never blocks).
    assert _maintenance_gate(SimpleNamespace()) is None


def test_cache_rebuild_guarded_during_loading():
    import freetoken.server.api_server as api

    prev = api._GLOBAL_STATE
    api._GLOBAL_STATE = SimpleNamespace(
        maintenance_state="loading",
        rebuild_futures={},
        last_rebuild=None,
    )
    try:
        client = TestClient(api.app)
        r = client.post("/v1/cache/rebuild", json={})
        assert r.status_code == 503
        assert "loading" in r.json().get("error", "").lower()
    finally:
        api._GLOBAL_STATE = prev


def test_cache_rebuild_timeout_keeps_gate_closed():
    # On HTTP timeout the scheduler may still be mid-rebuild, so the endpoint must NOT
    # reopen the maintenance gate -- it stays "rebuilding" until the real reply arrives.
    import asyncio
    from types import SimpleNamespace

    from freetoken.server import api_server
    from freetoken.server.api_server import CacheRebuildRequest, cache_rebuild

    sent = []

    async def send_one(msg):
        sent.append(msg)

    state = SimpleNamespace(
        maintenance_state="serving", rebuild_futures={}, last_rebuild=None, send_one=send_one
    )
    api_server._GLOBAL_STATE = state
    try:
        resp = asyncio.run(
            cache_rebuild(CacheRebuildRequest(moe_cache_size=8, timeout=0.05))
        )  # future never resolves -> times out
    finally:
        api_server._GLOBAL_STATE = None

    assert resp.status_code == 504
    assert state.maintenance_state == "rebuilding"  # gate stays closed
    assert state.rebuild_futures == {}  # cancelled future dropped, no leak
    assert len(sent) == 1  # the rebuild request was still dispatched to the backend


def test_cache_rebuild_send_failure_reports_unknown_delivery():
    import json

    from freetoken.server import api_server
    from freetoken.server.api_server import CacheRebuildRequest, cache_rebuild

    async def boom(msg):
        raise RuntimeError("zmq down")

    state = SimpleNamespace(
        maintenance_state="serving", rebuild_futures={}, last_rebuild=None, send_one=boom
    )
    previous_state = api_server._GLOBAL_STATE
    api_server._GLOBAL_STATE = state
    try:
        resp = asyncio.run(cache_rebuild(CacheRebuildRequest(moe_cache_size=8, timeout=5.0)))
    finally:
        api_server._GLOBAL_STATE = previous_state

    result = json.loads(resp.body)
    assert resp.status_code == 503
    assert result["delivery"] == "unknown"
    assert state.maintenance_state == "rebuilding"
    assert state._active_rebuild_id == result["request_id"]
    assert state.rebuild_futures == {}


def test_cache_rebuild_request_rejects_unknown_mode():
    # The public request model only accepts the implemented mode; "drain" is deferred and
    # must fail fast at the validation layer (422), not reach the scheduler.
    import pytest
    from pydantic import ValidationError

    from freetoken.server.api_server import CacheRebuildRequest

    assert CacheRebuildRequest(mode="if_idle").mode == "if_idle"
    with pytest.raises(ValidationError):
        CacheRebuildRequest(mode="drain")


@pytest.mark.parametrize("terminal", ["failed", "stopping"])
def test_dispatch_error_preserves_terminal_gate(terminal):
    async def run():
        async def send(_msg):
            if terminal == "failed":
                _mark_backend_failed(state, "scheduler exited")
            else:
                # The normal stop endpoint rejects rebuilding; this injects a later gate.
                with MAINTENANCE_LOCK:
                    state.maintenance_state = "stopping"
            raise RuntimeError("send failed")

        state = _FakeState(send)
        result = await dispatch_rebuild(state, moe_cache_size=8, num_pages=None)
        assert result["status"] == "failed"
        assert state.maintenance_state == terminal
        assert state.rebuild_futures == {}
        assert result["delivery"] == "unknown"
        assert state._active_rebuild_id == result["request_id"]

    asyncio.run(run())


@pytest.mark.parametrize("maintenance", ["loading", "rebuilding", "failed", "stopping"])
def test_dispatch_rechecks_admission_before_sending(maintenance):
    async def run():
        sent = []

        async def send(msg):
            sent.append(msg)

        state = _FakeState(send, maintenance_state=maintenance)
        result = await dispatch_rebuild(state, moe_cache_size=8, num_pages=None)
        assert result["status"] == "failed"
        assert state.maintenance_state == maintenance
        assert sent == []
        assert state.rebuild_futures == {}
        assert state._active_rebuild_id is None

    asyncio.run(run())


@pytest.mark.parametrize("status", ["ok", "failed"])
def test_unrelated_reply_preserves_active_operation_and_geometry(status):
    async def run():
        state = _FakeState(None, maintenance_state="rebuilding", active_rebuild_id="r2")
        previous = {"request_id": "r0", "num_pages": 64}
        state.last_rebuild = previous
        fut = asyncio.get_running_loop().create_future()
        state.rebuild_futures["r2"] = fut
        try:
            FrontendManager._resolve_rebuild(state, _reply("r1", status, num_pages=999))
            assert state.maintenance_state == "rebuilding"
            assert state._active_rebuild_id == "r2"
            assert state.last_rebuild is previous
            assert state.rebuild_futures == {"r2": fut}
            assert not fut.done()
        finally:
            fut.cancel()

    asyncio.run(run())


@pytest.mark.parametrize("terminal", ["failed", "stopping"])
@pytest.mark.parametrize("status", ["ok", "busy", "rejected", "unsupported"])
def test_matching_reply_cannot_reopen_terminal_gate(terminal, status):
    async def run():
        state = _FakeState(None, maintenance_state=terminal, active_rebuild_id="r1")
        fut = asyncio.get_running_loop().create_future()
        state.rebuild_futures["r1"] = fut
        FrontendManager._resolve_rebuild(state, _reply("r1", status, num_pages=64))
        assert state.maintenance_state == terminal
        assert state._active_rebuild_id is None
        assert fut.result()["num_pages"] == 64
        assert state.rebuild_futures == {}

    asyncio.run(run())


def test_duplicate_ok_after_failed_reply_does_not_resurrect_cache():
    state = _FakeState(None, maintenance_state="rebuilding", active_rebuild_id="r1")
    FrontendManager._resolve_rebuild(state, _reply("r1", "failed", num_pages=64))
    failed_result = state.last_rebuild
    FrontendManager._resolve_rebuild(state, _reply("r1", "ok", num_pages=999))
    assert state.maintenance_state == "failed"
    assert state.last_rebuild is failed_result
    assert state._active_rebuild_id is None


def test_unknown_matching_status_is_not_permission_to_serve():
    state = _FakeState(None, maintenance_state="rebuilding", active_rebuild_id="r1")
    FrontendManager._resolve_rebuild(state, _reply("r1", "future-status"))
    assert state.maintenance_state == "failed"
    assert state._active_rebuild_id is None


@pytest.mark.parametrize("phase", ["send", "reply"])
def test_cancelled_http_wait_retains_backend_identity(phase):
    async def run():
        sent = []
        started = asyncio.Event()
        hold_send = asyncio.Event()

        async def send(msg):
            sent.append(msg)
            started.set()
            if phase == "send":
                await hold_send.wait()

        state = _FakeState(send)
        task = asyncio.create_task(dispatch_rebuild(state, moe_cache_size=8, num_pages=None))
        await started.wait()
        request_id = sent[0].request_id
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.rebuild_futures == {}
        assert state._active_rebuild_id == request_id
        assert state.maintenance_state == "rebuilding"
        FrontendManager._resolve_rebuild(state, _reply(request_id, "ok", num_pages=64))
        assert state.maintenance_state == "serving"
        assert state._active_rebuild_id is None

    asyncio.run(run())


def test_previous_reply_cannot_complete_a_new_dispatch():
    async def run():
        sent = []
        started = asyncio.Event()

        async def send(msg):
            sent.append(msg)
            started.set()

        state = _FakeState(send)
        first = asyncio.create_task(dispatch_rebuild(state, moe_cache_size=8, num_pages=None))
        await started.wait()
        r1 = sent[-1].request_id
        FrontendManager._resolve_rebuild(state, _reply(r1, "ok", num_pages=64))
        assert (await first)["status"] == "ok"
        started.clear()
        second = asyncio.create_task(dispatch_rebuild(state, moe_cache_size=16, num_pages=None))
        await started.wait()
        r2 = sent[-1].request_id
        assert r1 != r2
        FrontendManager._resolve_rebuild(state, _reply(r1, "ok", num_pages=999))
        assert not second.done()
        assert state.maintenance_state == "rebuilding"
        assert state._active_rebuild_id == r2
        assert state.last_rebuild["num_pages"] == 64
        FrontendManager._resolve_rebuild(state, _reply(r2, "ok", num_pages=128))
        assert (await second)["num_pages"] == 128
        assert state.maintenance_state == "serving"

    asyncio.run(run())


@pytest.mark.parametrize("failure_first", [False, True])
def test_ready_and_failure_order_always_ends_failed(failure_first):
    state = _FakeState(None, maintenance_state="loading")
    if failure_first:
        _mark_backend_failed(state, "scheduler exited")
        assert _mark_backend_ready(state) is False
    else:
        assert _mark_backend_ready(state) is True
        _mark_backend_failed(state, "scheduler exited")
    assert state.maintenance_state == "failed"
    assert state.fatal_error == "scheduler exited"


def test_failure_thread_is_not_blocked_by_awaited_send():
    async def run():
        async def send(_msg):
            await asyncio.wait_for(
                asyncio.to_thread(_mark_backend_failed, state, "scheduler exited"),
                timeout=2.0,
            )
            raise RuntimeError("send failed after worker death")

        state = _FakeState(send)
        result = await dispatch_rebuild(state, moe_cache_size=8, num_pages=None)
        assert "send failed after worker death" in result["error"]
        assert state.fatal_error == "scheduler exited"
        assert state.maintenance_state == "failed"
        assert state.rebuild_futures == {}

    asyncio.run(run())


def test_fatal_evidence_blocks_generation_even_with_stale_serving_label():
    manager = FrontendManager(
        config=SimpleNamespace(served_model_name="model-a"),
        send_tokenizer=None, recv_tokenizer=None,
        maintenance_state="serving", fatal_error="scheduler exited",
    )
    with pytest.raises(AdmissionClosedError):
        manager.new_user()
    assert manager.stats.active == 0
    assert manager.ack_map == {}
    assert manager.event_map == {}


def test_old_send_error_cannot_clear_a_newer_operation():
    async def run():
        sent = []
        second_started = asyncio.Event()
        second = None

        async def send(msg):
            nonlocal second
            sent.append(msg)
            if len(sent) == 1:
                FrontendManager._resolve_rebuild(state, _reply(msg.request_id, "ok"))
                second = asyncio.create_task(
                    dispatch_rebuild(state, moe_cache_size=16, num_pages=None)
                )
                await second_started.wait()
                raise RuntimeError("old send failed after its reply")
            second_started.set()

        state = _FakeState(send)
        result = await dispatch_rebuild(state, moe_cache_size=8, num_pages=None)
        assert result["status"] == "ok"
        r2 = sent[1].request_id
        assert state._active_rebuild_id == r2
        assert state.maintenance_state == "rebuilding"
        assert set(state.rebuild_futures) == {r2}
        assert second is not None and not second.done()
        FrontendManager._resolve_rebuild(state, _reply(r2, "ok", num_pages=128))
        assert (await second)["num_pages"] == 128
        assert state.rebuild_futures == {}

    asyncio.run(run())


@pytest.mark.parametrize("status", ["ok", "busy", "rejected", "unsupported", "failed"])
def test_late_reply_resolves_unknown_send_without_a_retry(status):
    async def run():
        sent = []

        async def send(msg):
            sent.append(msg)
            raise RuntimeError("send outcome unavailable")

        state = _FakeState(send)
        previous = {"request_id": "previous", "num_pages": 64}
        state.last_rebuild = previous
        result = await dispatch_rebuild(state, moe_cache_size=8, num_pages=None)
        request_id = result["request_id"]
        assert result["delivery"] == "unknown"
        assert state.last_rebuild is previous
        assert state._active_rebuild_id == request_id
        assert state.rebuild_futures == {}

        blocked = await dispatch_rebuild(state, moe_cache_size=16, num_pages=None)
        assert blocked["status"] == "failed"
        assert len(sent) == 1
        FrontendManager._resolve_rebuild(state, _reply("unrelated", "ok", num_pages=999))
        assert state._active_rebuild_id == request_id
        assert state.last_rebuild is previous

        FrontendManager._resolve_rebuild(state, _reply(request_id, status, num_pages=128))
        assert state._active_rebuild_id is None
        assert state.last_rebuild["request_id"] == request_id
        assert state.last_rebuild["num_pages"] == 128
        assert state.maintenance_state == ("failed" if status == "failed" else "serving")

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["encoder", "socket_call", "socket_future"])
def test_queue_error_type_does_not_prove_the_send_phase(phase):
    from freetoken.utils.mp import ZmqAsyncPushQueue

    async def run():
        calls = []

        def encode(msg):
            if phase == "encoder":
                raise RuntimeError("injected transport failure")
            return {"request_id": msg.request_id}

        class Socket:
            def send(self, data, **kwargs):
                calls.append(data)
                if phase == "socket_call":
                    raise RuntimeError("injected transport failure")
                fut = asyncio.get_running_loop().create_future()
                fut.set_exception(RuntimeError("injected transport failure"))
                return fut

        # No constructor: it would create a real context and socket.
        queue = ZmqAsyncPushQueue.__new__(ZmqAsyncPushQueue)
        queue.encoder = encode
        queue.socket = Socket()
        state = _FakeState(queue.put)
        result = await dispatch_rebuild(state, moe_cache_size=8, num_pages=None)
        assert len(calls) == (0 if phase == "encoder" else 1)
        assert result["delivery"] == "unknown"
        assert state.maintenance_state == "rebuilding"
        assert state._active_rebuild_id == result["request_id"]
        assert state.rebuild_futures == {}

    asyncio.run(run())


def test_rebuild_deadline_also_bounds_an_awaited_send():
    async def run():
        blocker = asyncio.Event()
        send_exited = asyncio.Event()

        async def send(_msg):
            try:
                await blocker.wait()
            finally:
                send_exited.set()

        state = _FakeState(send)
        # The outer limit is a regression guard, not a second production deadline.
        result = await asyncio.wait_for(
            dispatch_rebuild(state, moe_cache_size=8, num_pages=None, timeout=0.01),
            timeout=2.0,
        )
        assert result["status"] == "timeout"
        assert send_exited.is_set()
        assert state.maintenance_state == "rebuilding"
        assert state._active_rebuild_id == result["request_id"]
        assert state.rebuild_futures == {}

    asyncio.run(run())


@pytest.mark.parametrize("status", ["ok", "rejected", "failed"])
def test_received_result_outranks_a_later_local_send_error(status):
    async def run():
        async def send(msg):
            FrontendManager._resolve_rebuild(
                state, _reply(msg.request_id, status, num_pages=128)
            )
            raise RuntimeError("local error after the backend result")

        state = _FakeState(send)
        result = await dispatch_rebuild(state, moe_cache_size=8, num_pages=None)
        assert result is state.last_rebuild
        assert result["status"] == status
        assert result["num_pages"] == 128
        assert "delivery" not in result
        assert state._active_rebuild_id is None
        assert state.maintenance_state == ("failed" if status == "failed" else "serving")
        assert state.rebuild_futures == {}

    asyncio.run(run())
