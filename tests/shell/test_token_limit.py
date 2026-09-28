"""A limited shell turn must be visible and retry without replaying its partial answer."""

import asyncio

import pytest

from freetoken.shell import tui
from freetoken.shell.client import ContentDelta, ReasoningDelta, TurnDone


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
