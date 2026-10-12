import builtins
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models

import freetoken.models.gguf.tokenizer as gguf_tokenizer


TOKENS = ["a", "<tool>", "b", "<|im_end|>", "<|endoftext|>", "[UNK]"]
TOKEN_TYPES = [1, 4, 1, 3, 3, 2]


@pytest.mark.parametrize("modern", [False, True])
def test_generic_converter_supports_both_metadata_interfaces(monkeypatch, modern):
    original_import = builtins.__import__
    calls = []
    backend = object()

    def convert(arch, data):
        calls.append((arch, data))
        return backend, {}

    def converter_import(name, *args, **kwargs):
        if name == "transformers.integrations.gguf":
            if not modern:
                raise ModuleNotFoundError(name=name)
            return SimpleNamespace(convert_gguf_tokenizer=convert)
        if name == "transformers.integrations.ggml":
            return SimpleNamespace(convert_gguf_tokenizer=convert)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", converter_import)
    data = {"model": "llama", "pre": "default", "tokens": ["a"]}
    assert gguf_tokenizer._convert_gguf_tokenizer("gemma4_text", data) == (backend, {})
    expected = dict(data)
    if modern:
        expected.update(tokenizer_type="llama", pre_tokenizer_type="default")
    assert calls == [("gemma4_text", expected)]
    assert set(data) == {"model", "pre", "tokens"}


def test_generic_converter_does_not_hide_missing_transitive_dependency(monkeypatch):
    original_import = builtins.__import__

    def converter_import(name, *args, **kwargs):
        if name == "transformers.integrations.gguf":
            raise ModuleNotFoundError(name="sentencepiece")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", converter_import)
    with pytest.raises(ModuleNotFoundError) as raised:
        gguf_tokenizer._convert_gguf_tokenizer("gemma4_text", {})
    assert raised.value.name == "sentencepiece"


def _backend_tokenizer():
    vocab = {token: index for index, token in enumerate(TOKENS)}
    return Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))


def _metadata(token_types=TOKEN_TYPES):
    return {
        "tokenizer.ggml.tokens": TOKENS,
        "tokenizer.ggml.token_type": token_types,
        "tokenizer.ggml.eos_token_id": 4,
        "tokenizer.ggml.unknown_token_id": 5,
    }


@pytest.mark.parametrize("arch", ("qwen35", "qwen35moe"))
def test_qwen_user_defined_tokens_remain_atomic_at_existing_ids(monkeypatch, arch):
    backend = _backend_tokenizer()
    original_ids = {token: backend.token_to_id(token) for token in TOKENS}
    calls = []

    def convert_gguf_tokenizer(converter_arch, tokenizer_dict):
        calls.append((converter_arch, tokenizer_dict))
        return backend, {}

    monkeypatch.setattr(gguf_tokenizer, "load_gguf_metadata", lambda _path: _metadata())
    monkeypatch.setattr(gguf_tokenizer, "gguf_architecture", lambda _path: arch)
    monkeypatch.setattr(gguf_tokenizer, "_convert_gguf_tokenizer", convert_gguf_tokenizer)

    tokenizer = gguf_tokenizer.load_gguf_tokenizer("unused.gguf")

    assert calls[0][0] == "qwen2"
    assert tokenizer.convert_tokens_to_ids(TOKENS) == list(original_ids.values())
    assert tokenizer.encode("a<tool>b", add_special_tokens=False) == [0, 1, 2]
    assert "<tool>" not in tokenizer.all_special_tokens
    assert len(tokenizer) == len(TOKENS)
    restored = backend.get_added_tokens_decoder()[original_ids["<tool>"]]
    assert restored.content == "<tool>"
    assert restored.normalized is False
    assert restored.special is False


def test_qwen_token_type_length_mismatch_fails(monkeypatch):
    monkeypatch.setattr(
        gguf_tokenizer,
        "load_gguf_metadata",
        lambda _path: _metadata(token_types=TOKEN_TYPES[:-1]),
    )
    monkeypatch.setattr(gguf_tokenizer, "gguf_architecture", lambda _path: "qwen35")
    monkeypatch.setattr(
        gguf_tokenizer,
        "_convert_gguf_tokenizer",
        lambda _arch, _tokenizer_dict: (_backend_tokenizer(), {}),
    )

    with pytest.raises(ValueError, match="token_type length"):
        gguf_tokenizer.load_gguf_tokenizer("unused.gguf")
