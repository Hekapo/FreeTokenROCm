"""A backend worker that dies while serving: waiters get a 503-class error and no worker is left behind."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import freetoken.server.api_server as api_server
from freetoken.server.api_server import FrontendManager
from freetoken.server.generation import ENGINE_UNAVAILABLE
from freetoken.server.openai_api import _generation_error_response
from freetoken.server.responses_api import _error_response


def _manager() -> FrontendManager:
    return FrontendManager(
        config=SimpleNamespace(served_model_name="model-a"),
        send_tokenizer=None,
        recv_tokenizer=None,
        maintenance_state="serving",
    )


def test_inflight_requests_are_failed_from_the_supervisor_thread():
    async def _run():
        manager = _manager()
        manager._loop = asyncio.get_running_loop()
        uids = [manager.new_user(), manager.new_user()]

        async def consume(uid):
            return [ack async for ack in manager.wait_for_ack(uid)]

        tasks = [asyncio.create_task(consume(uid)) for uid in uids]
        await asyncio.sleep(0)
        # the supervisor runs on its own thread
        thread = threading.Thread(
            target=manager.fail_inflight_requests, args=("backend worker freetoken-TP0-scheduler exited",)
        )
        thread.start()
        thread.join()

        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
        for acks in results:
            (ack,) = acks
            assert ack.finished
            assert ack.error_code == ENGINE_UNAVAILABLE
            assert "freetoken-TP0-scheduler exited" in ack.error
        assert manager.ack_map == {} and manager.event_map == {}
        assert manager.stats.active == 0

    asyncio.run(_run())


def test_failing_inflight_requests_before_the_listener_started_is_a_no_op():
    manager = _manager()
    manager.new_user()
    manager.fail_inflight_requests("dead")  # no loop yet: nothing can be waiting on an event


class _Proc:
    def __init__(self, calls, name):
        self.calls, self.name, self.alive = calls, name, True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.calls.append(("terminate", self.name))
        self.alive = False

    def join(self, timeout=None):
        self.calls.append(("join", self.name))

    def kill(self):
        self.calls.append(("kill", self.name))


def test_exit_after_backend_death_tears_down_workers_before_signalling(monkeypatch):
    calls = []
    signalled = threading.Event()

    def fake_kill(pid, sig):
        calls.append(("signal", sig))
        signalled.set()

    monkeypatch.setattr(api_server.os, "kill", fake_kill)
    monkeypatch.setattr(api_server, "_SHUTTING_DOWN", threading.Event())
    scheduler = _Proc(calls, "scheduler")
    scheduler.alive = False  # the one that crashed
    detokenizer = _Proc(calls, "detokenizer")

    api_server._exit_after_backend_death(0.01, [scheduler, detokenizer])
    assert signalled.wait(5.0)

    assert calls.index(("terminate", "detokenizer")) < calls.index(("signal", api_server.signal.SIGTERM))
    assert ("terminate", "scheduler") not in calls  # already dead: nothing to nudge
    assert calls[-1][0] == "signal"


def test_exit_after_backend_death_defers_to_an_orderly_stop(monkeypatch):
    calls = []
    stopping = threading.Event()
    stopping.set()
    monkeypatch.setattr(api_server.os, "kill", lambda pid, sig: calls.append("signal"))
    monkeypatch.setattr(api_server, "_SHUTTING_DOWN", stopping)
    detokenizer = _Proc(calls, "detokenizer")

    api_server._exit_after_backend_death(0.01, [detokenizer]).join()

    assert calls == []


def test_engine_failure_maps_to_503_and_input_errors_stay_400():
    dead = _generation_error_response("engine unavailable: boom", ENGINE_UNAVAILABLE)
    assert dead.status_code == 503
    assert json.loads(dead.body)["error"]["type"] == "server_error"

    bad = _generation_error_response("prompt too long", "context_length_exceeded")
    assert bad.status_code == 400
    assert json.loads(bad.body)["error"]["type"] == "invalid_request_error"

    responses_dead = _error_response(503, "engine unavailable: boom", ENGINE_UNAVAILABLE)
    assert json.loads(responses_dead.body)["error"]["type"] == "server_error"


def test_supervisor_fails_on_a_serving_error_ack_while_the_worker_still_looks_alive():
    import queue

    from freetoken.server.supervisor import BackendHandle, LoadProgress, run_backend_supervisor

    class StuckInDriver:
        """Interpreter gone, but the driver keeps the process object alive (is_alive() True)."""
        name = "freetoken-TP0-scheduler"

        def is_alive(self):
            return True

    acks: "queue.Queue" = queue.Queue()
    acks.put("Scheduler is ready")
    ready, failures = threading.Event(), []
    supervisor = threading.Thread(
        target=run_backend_supervisor,
        args=(BackendHandle(ack_queue=acks, processes=[StuckInDriver()], expected_acks=1), LoadProgress(), ready.set),
        kwargs={"on_failure": failures.append, "poll": 0.01},
        daemon=True,
    )
    supervisor.start()
    assert ready.wait(5.0)

    acks.put(("progress", "late bar", 1, 2))  # stray non-error acks are ignored
    acks.put(("error", "AssertionError: Cooperative launch requested but not supported by device"))
    supervisor.join(5.0)

    assert not supervisor.is_alive()
    assert failures == ["AssertionError: Cooperative launch requested but not supported by device"]


def test_supervisor_stays_quiet_about_an_error_ack_during_an_orderly_stop():
    import queue

    from freetoken.server.supervisor import BackendHandle, LoadProgress, run_backend_supervisor

    class Alive:
        name = "freetoken-TP0-scheduler"

        def is_alive(self):
            return True

    acks: "queue.Queue" = queue.Queue()
    acks.put("Scheduler is ready")
    acks.put(("error", "KeyboardInterrupt: "))
    failures = []
    run_backend_supervisor(
        BackendHandle(ack_queue=acks, processes=[Alive()], expected_acks=1), LoadProgress(), lambda: None,
        on_failure=failures.append, poll=0.01, is_shutting_down=lambda: True,
    )
    assert failures == []
