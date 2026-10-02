"""A limited shell turn must be visible and retry without replaying its partial answer."""

import asyncio

import pytest

from freetoken.shell import tui
from freetoken.shell.client import ContentDelta, ReasoningDelta, TurnDone


@pytest.mark.parametrize("has_initial_stats", [False, True])
def test_shell_context_budget_tracks_resized_kv_within_model_ceiling(monkeypatch, has_initial_stats):
    from freetoken.shell.client import ShellClient

    client = ShellClient.__new__(ShellClient)

    async def metadata(*args):
        return {"data": [{"id": "unit-model", "context_length": 65536,
                          "effective_context_length": 8192}]}

    async def ready(**kwargs):
        return {"status": "ok"}

    async def cache_status():
        return {"geometry": {}}

    async def initial_stats():
        return {"kv": {"used_pages": 0, "total_pages": 8192, "page_size": 1}} if has_initial_stats else {}

    client._request_json = metadata
    client.wait_until_ready = ready
    client.cache_status = cache_status
    client.stats = initial_stats
    captured = []
    stats_type = tui.ShellStats

    def capture_stats(**kwargs):
        stats = stats_type(**kwargs)
        captured.append(stats)
        return stats

    class Renderer:
        _write_stdout = staticmethod(lambda text: None)

        def __init__(self, **kwargs):
            pass

    class StatusLine:
        write_output = staticmethod(lambda text: None)

        def __init__(self, *args, **kwargs):
            pass

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def prompt_async(self):
            stats = captured[0]
            assert stats.context_limit == 8192
            for capacity, expected in [(65536, 65536), (4096, 4096), (131072, 65536)]:
                stats.apply_stats_doc({"kv": {"used_pages": 0, "total_pages": capacity,
                                              "page_size": 1}})
                assert stats.context_limit == expected
                stats.prompt_tokens, stats.completion_tokens = 3072, 1024
                assert f"ctx {expected} remaining {expected - 4096}" in stats.format()
            raise EOFError

    monkeypatch.setattr(tui, "ShellStats", capture_stats)
    monkeypatch.setattr(tui, "ShellConsoleRenderer", Renderer)
    monkeypatch.setattr(tui, "ShellStatusLine", StatusLine)
    monkeypatch.setattr(tui, "PromptSession", Session)
    assert asyncio.run(tui._run_shell(client, "http://127.0.0.1:1919", connect_grace=0)) == 0


@pytest.mark.parametrize("card, effective, configured", [
    ({"context_length": 65536, "effective_context_length": 8192}, 8192, 65536),
    ({"context_length": 262144, "effective_context_length": 65536}, 65536, 262144),
    ({"max_model_len": 65536, "effective_context_length": 8192}, 8192, 65536),
    ({"context_length": 65536}, 65536, 65536),
    ({"max_model_len": 4096}, 4096, 4096),
    ({"effective_context_length": 8192}, 8192, None),
    ({}, None, None),
])
def test_shell_reads_allocated_context_budget(card, effective, configured):
    from freetoken.shell.client import ShellClient

    client = ShellClient.__new__(ShellClient)

    async def metadata(*args):
        return {"data": [{"id": "unit-model", **card}]}

    client._request_json = metadata
    assert asyncio.run(client.model_id()) == "unit-model"
    assert client.context_length == effective
    assert client.model_context_length == configured


@pytest.mark.parametrize("first_answer", ["", "   ", "Partial answer"])
def test_limited_turn_can_be_retried_with_new_limit_and_thinking_off(monkeypatch, first_answer):
    output = []

    class Renderer:
        def __init__(self, write, *, display_width):
            self.write = write
            self.display_width = display_width

        @staticmethod
        def _write_stdout(text):
            output.append(text)

        def begin_turn(self, cmd):
            pass

        def write_reasoning(self, text):
            pass

        def write_content(self, text):
            self.write(text)

        def finish_turn(self):
            pass

    class StatusLine:
        def __init__(self, *args, **kwargs):
            pass

        def write_output(self, text):
            output.append(text)

        def activate(self):
            pass

        def maybe(self):
            pass

        def force(self):
            pass

        def deactivate(self):
            pass

    commands = iter(["first", "/think off", "/retry 3072", "second"])

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def prompt_async(self):
            try:
                return next(commands)
            except StopIteration:
                raise EOFError from None

    class Client:
        def __init__(self):
            self.requests = []

        async def wait_until_ready(self, **kwargs):
            return {"status": "ok"}

        async def model_id(self):
            return "test-model"

        async def cache_status(self):
            return {"geometry": {"reasoning": {
                "gears": ["off", "on"], "default": "on",
                "kwargs": {"off": {"enable_thinking": False}, "on": {"enable_thinking": True}},
            }}}

        async def stats(self):
            return {}

        async def chat(self, messages, *, model, sampling, chat_template_kwargs):
            self.requests.append((messages, sampling, chat_template_kwargs))
            if len(self.requests) == 1:
                yield ReasoningDelta("thinking")
                if first_answer:
                    yield ContentDelta(first_answer)
                yield TurnDone("length", 12, 2048)
            else:
                yield ContentDelta("Complete" if len(self.requests) == 2 else "Next")
                yield TurnDone("stop", 12, 8)

    monkeypatch.setattr(tui, "ShellConsoleRenderer", Renderer)
    monkeypatch.setattr(tui, "ShellStatusLine", StatusLine)
    monkeypatch.setattr(tui, "PromptSession", Session)
    client = Client()
    assert asyncio.run(tui._run_shell(client, "http://127.0.0.1:1919", connect_grace=0)) == 0

    assert len(client.requests) == 3
    assert client.requests[0][0] == [{"role": "user", "content": "first"}]
    assert client.requests[1][0] == [{"role": "user", "content": "first"}]
    assert client.requests[1][1].max_tokens == 3072
    assert client.requests[1][2] == {"enable_thinking": False}
    assert client.requests[2][0] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "Complete"},
        {"role": "user", "content": "second"},
    ]
    rendered = "".join(output)
    assert "Token limit reached: 2048/2048" in rendered
    assert ("answer above may be incomplete" in rendered.lower()) == bool(first_answer.strip())
    if not first_answer.strip():
        assert "No final answer was produced" in rendered
