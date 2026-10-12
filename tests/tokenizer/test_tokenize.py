from __future__ import annotations

import json
from copy import deepcopy
import re

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.tokenize import TokenizeManager, _dsv4_arguments_str
from freetoken.tokenizer.inline_system import InlineSystemError, normalize_inline_system, probe_inline_system


@pytest.mark.parametrize("mode", ["preserve", "fold"])
def test_inline_system_waits_for_parallel_tools_without_mutation(mode):
    messages = [
        {"role": "system", "content": "stable"},
        {"role": "user", "content": "read files"},
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}],
         "reasoning_content": "retain this reasoning"},
        {"role": "system", "content": "reminder"},
        {"role": "tool", "tool_call_id": "a", "content": "first"},
        {"role": "tool", "tool_call_id": "b", "content": "second"},
    ]
    original = deepcopy(messages)
    result = normalize_inline_system(messages, mode)
    assert [m["role"] for m in result[:5]] == ["system", "user", "assistant", "tool", "tool"]
    assert result[2]["reasoning_content"] == "retain this reasoning"
    if mode == "preserve":
        assert result[5] == messages[3]
    else:
        assert result[-1]["content"] == "second\n\nreminder"
    assert messages == original


@pytest.mark.parametrize("mode", ["preserve", "fold"])
@pytest.mark.parametrize("tail", [[], [{"role": "user", "content": "next"}]])
def test_inline_system_rejects_incomplete_tools(mode, tail):
    messages = [{"role": "assistant", "tool_calls": [{"id": "missing"}]},
                {"role": "system", "content": "reminder"}, *tail]
    with pytest.raises(InlineSystemError, match="tool"):
        normalize_inline_system(messages, mode)


def test_inline_fold_preserves_prefix_and_content_blocks():
    messages = [{"role": "system", "content": "stable"},
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
                {"role": "system", "content": "note"},
                {"role": "assistant", "content": "answer"},
                {"role": "system", "content": "next note"},
                {"role": "user", "content": "next question"}]
    original = deepcopy(messages)
    result = normalize_inline_system(messages, "fold")
    assert result[0] == messages[0]
    assert result[1]["content"] == [{"type": "text", "text": "question\n\nnote"}]
    assert result[-1]["content"] == "next note\n\nnext question"
    assert messages == original
    no_system = normalize_inline_system(messages[1:], "preserve")
    assert all(m["role"] != "system" for m in no_system)


class _InlineProbeTokenizer:
    roles = {"<system>": 1, "<user>": 2, "<assistant>": 3, "<tool>": 4, "<end>": 5}
    all_special_ids = list(roles.values())

    def __init__(self, behavior="preserve"):
        self.behavior = behavior
        self.last_messages = None
        self.last_kwargs = None

    def get_added_vocab(self):
        return self.roles

    def encode(self, text, return_tensors=None, **kwargs):
        tokens = [self.roles[x] if x in self.roles else ord(x) + 100
                  for x in re.findall(r"<[^>]+>|.", text, re.S)]
        return torch.tensor([tokens]) if return_tensors else tokens

    def decode(self, ids):
        inverse = {value: key for key, value in self.roles.items()}
        return "".join(inverse[x] if x in inverse else chr(x - 100) for x in ids)

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = deepcopy(messages)
        self.last_kwargs = kwargs
        if self.behavior == "hoist":
            messages = deepcopy(messages)
            notes = [m["content"] for m in messages[1:] if m["role"] == "system"]
            messages[0]["content"] += "".join(notes)
            messages = [m for i, m in enumerate(messages) if i == 0 or m["role"] != "system"]
        out = []
        for i, message in enumerate(messages):
            role = message["role"]
            if role == "system" and i > 0:
                if self.behavior == "reject":
                    raise ValueError("late system unsupported")
                if self.behavior == "drop":
                    continue
                if self.behavior == "user":
                    role = "user"
            out.append(f"<{role}>" + message.get("content", "") +
                       ("" if self.behavior == "reasoning" and any(m["role"] == "system" for m in messages[1:])
                        else message.get("reasoning_content", "")) + "<end>")
        return "".join(out) + "<assistant>"


@pytest.mark.parametrize("behavior,expected", [("preserve", "preserve"), ("reject", "fold"),
                                               ("drop", "fold"), ("user", "fold"),
                                               ("hoist", "fold"), ("reasoning", "fold")])
def test_inline_probe_requires_stable_distinct_system_turns(behavior, expected):
    tokenizer = _InlineProbeTokenizer(behavior)
    mode, _ = probe_inline_system(tokenizer.apply_chat_template, tokenizer)
    assert mode == expected


def test_inline_policy_is_opt_in_and_probe_cache_is_scoped(monkeypatch):
    tokenizer = _InlineProbeTokenizer("reject")
    manager = TokenizeManager(tokenizer)
    messages = [{"role": "system", "content": "stable"}, {"role": "user", "content": "ask"},
                {"role": "system", "content": "note"}]
    msg = TokenizeMsg(1, messages, SamplingParams())
    with pytest.raises(ValueError, match="unsupported"):
        manager.render_prompt(msg)
    calls = []

    def probe(render, tokenizer):
        calls.append(render)
        return "fold", "test"

    monkeypatch.setattr("freetoken.tokenizer.tokenize.probe_inline_system", probe)
    msg.inline_system_policy = "auto"
    first = manager.render_prompt(msg)
    assert manager.render_prompt(msg) == first
    msg.tools = [{"name": "read"}]
    manager.render_prompt(msg)
    msg.chat_template_kwargs = {"enable_thinking": False}
    manager.render_prompt(msg)
    assert len(calls) == 3
    assert "inline_system_policy" not in tokenizer.last_kwargs
    assert messages[-1] == {"role": "system", "content": "note"}
    [ids] = manager.tokenize([msg])
    assert ids.dtype == torch.int32
    assert tokenizer.decode(ids.tolist()).startswith("<system>stable<end><user>ask\n\nnote")


class FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_kwargs = kwargs
        return "rendered prompt"

    def encode(self, prompt, return_tensors=None, add_special_tokens=True):
        assert prompt == "rendered prompt"
        assert return_tensors == "pt"
        # The template rendered every special token already; encode must not
        # add another bos on top (the muse-glimmer/llama double-bos bug).
        assert add_special_tokens is False
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


def test_tokenize_manager_passes_chat_template_kwargs():
    tokenizer = FakeTokenizer()
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"enable_thinking": True},
    )

    [input_ids] = manager.tokenize([msg])

    assert tokenizer.chat_template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    assert input_ids.tolist() == [1, 2, 3]


def test_tokenize_manager_passes_tools_to_chat_template():
    tokenizer = FakeTokenizer()
    manager = TokenizeManager(tokenizer)
    tools = [
        {
            "name": "get_weather",
            "description": "Return weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "weather?"}],
        sampling_params=SamplingParams(),
        tools=tools,
    )

    [input_ids] = manager.tokenize([msg])

    assert tokenizer.chat_template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "tools": tools,
    }
    assert input_ids.tolist() == [1, 2, 3]


class FakeDsv4Tokenizer:
    chat_template = None

    def __init__(self, model_path) -> None:
        self.name_or_path = str(model_path)
        self.prompt = None

    def encode(self, prompt, return_tensors=None, add_special_tokens=True):
        self.prompt = prompt
        assert return_tensors == "pt"
        # dsv4's own encoder path keeps the default special-token behavior.
        assert add_special_tokens is True
        return torch.tensor([[4, 5, 6]], dtype=torch.long)


def test_tokenize_manager_uses_dsv4_encoder_when_chat_template_is_missing(tmp_path):
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        """
def encode_messages(messages, thinking_mode, reasoning_effort=None):
    assert thinking_mode == "thinking"
    assert messages[0]["role"] == "system"
    assert messages[0]["tools"][0]["function"]["name"] == "read"
    assert messages[1]["role"] == "user"
    return "dsv4 prompt"
""".lstrip()
    )
    tokenizer = FakeDsv4Tokenizer(tmp_path)
    manager = TokenizeManager(tokenizer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}},
            },
        }
    ]
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "inspect files"}],
        sampling_params=SamplingParams(),
        tools=tools,
    )

    [input_ids] = manager.tokenize([msg])

    assert tokenizer.prompt == "dsv4 prompt"
    assert input_ids.tolist() == [4, 5, 6]


def test_dsv4_encoder_gets_tool_call_arguments_as_json_string(tmp_path):
    """Regression: render_messages hands the template dict arguments; the dsv4
    encoder contract is a JSON-object STRING -- a dict trips its fallback that
    wraps every replayed call in a parameter literally named "arguments"."""
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        """
import json

def encode_messages(messages, thinking_mode, reasoning_effort=None):
    (tc,) = messages[1]["tool_calls"]
    arguments = tc["function"]["arguments"]
    assert isinstance(arguments, str), f"expected str, got {type(arguments)}"
    assert json.loads(arguments) == {"command": "gog calendar time", "n": 2}
    return "dsv4 prompt"
""".lstrip()
    )
    tokenizer = FakeDsv4Tokenizer(tmp_path)
    manager = TokenizeManager(tokenizer)
    messages = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call0",
                    "type": "function",
                    # dict form, as produced by server render_messages
                    "function": {"name": "exec", "arguments": {"command": "gog calendar time", "n": 2}},
                }
            ],
        },
    ]
    msg = TokenizeMsg(uid=1, text=messages, sampling_params=SamplingParams())

    [input_ids] = manager.tokenize([msg])

    assert tokenizer.prompt == "dsv4 prompt"
    assert input_ids.tolist() == [4, 5, 6]
    # caller's messages must not be mutated (message copies are shallow)
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {
        "command": "gog calendar time",
        "n": 2,
    }


def test_dsv4_arguments_str_normalization():
    assert _dsv4_arguments_str({"a": 1, "b": "x"}) == '{"a": 1, "b": "x"}'
    assert _dsv4_arguments_str({"t": "héllo 世界"}) == '{"t": "héllo 世界"}'  # ensure_ascii=False
    assert _dsv4_arguments_str('{"a": 1}') == '{"a": 1}'  # object string passes through verbatim
    assert _dsv4_arguments_str(None) == "{}"
    assert _dsv4_arguments_str("") == "{}"
    assert _dsv4_arguments_str("  ") == "{}"
    for bad in ("[1,2]", "5", "true", '"x"', "not json", [1, 2], 5):
        with pytest.raises(ValueError):
            _dsv4_arguments_str(bad)


class Qwen38LikeTokenizer:
    """Fake whose template grades effort like Qwen3.8: validates the vocabulary
    whenever thinking is not explicitly off, distinct preamble per gear."""

    def __init__(self) -> None:
        self.chat_template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_kwargs = kwargs
        if kwargs.get("enable_thinking") is not False:
            effort = kwargs.get("reasoning_effort", "xhigh")
            if effort not in ("xhigh", "medium", "low"):
                raise ValueError(f"Unexpected reasoning effort {effort}")
            return f"prompt effort={effort}"
        return "prompt effort=off"

    def encode(self, prompt, return_tensors=None, add_special_tokens=True):
        return torch.tensor([[7, 8]], dtype=torch.long)


def test_tokenize_quantizes_foreign_effort_onto_the_template_vocabulary():
    """DeepSeek-dialect "high" must reach a Qwen3.8-style template as its
    nearest supported gear, not raw (raw would raise_exception)."""
    tokenizer = Qwen38LikeTokenizer()
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "high"},
    )

    manager.tokenize([msg])

    assert tokenizer.chat_template_kwargs["reasoning_effort"] == "xhigh"
    assert tokenizer.chat_template_kwargs["enable_thinking"] is True
    # the caller's kwargs stay untouched
    assert msg.chat_template_kwargs["reasoning_effort"] == "high"


def test_tokenize_drops_effort_for_templates_that_ignore_it():
    tokenizer = FakeTokenizer()  # renders the same prompt regardless of kwargs
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"reasoning_effort": "high"},
    )

    manager.tokenize([msg])

    assert "reasoning_effort" not in tokenizer.chat_template_kwargs


def test_tokenize_drops_far_effort_for_the_dsv4_encoder(tmp_path):
    """An OpenAI-dialect "medium" has no nearby dsv4 gear, so nothing is sent
    and the encoder default ("low") applies -- never a silent escalation to
    the absolute-maximum "high" prompt."""
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        """
SEEN = []

def encode_messages(messages, thinking_mode, reasoning_effort=None):
    effort = reasoning_effort or "low"
    assert effort in ("low", "high", "max"), f"Invalid reasoning effort: {effort}"
    SEEN.append(reasoning_effort)
    return f"dsv4 prompt effort={effort}"
""".lstrip()
    )
    tokenizer = FakeDsv4Tokenizer(tmp_path)
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"reasoning_effort": "medium"},
    )

    manager.tokenize([msg])

    assert tokenizer.prompt == "dsv4 prompt effort=low"


def test_tokenize_survives_an_unhashable_effort():
    tokenizer = Qwen38LikeTokenizer()
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "hello"}],
        sampling_params=SamplingParams(),
        chat_template_kwargs={"reasoning_effort": ["high"]},  # legal JSON on the wire
    )

    manager.tokenize([msg])

    assert "reasoning_effort" not in tokenizer.chat_template_kwargs
