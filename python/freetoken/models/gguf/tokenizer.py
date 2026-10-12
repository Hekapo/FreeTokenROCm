"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from .reader import gguf_architecture, load_gguf_metadata

# Keep the validated Qwen2 BPE behavior for the existing Qwen GGUF profile.
# Transformers 5.19's generic GGUF converter changes normalization and AddedTokens.
_TOKENIZER_ARCH = {"gemma4": "gemma4_text", "qwen35moe": "qwen2", "qwen35": "qwen2"}

# Per-arch chat/stop tokens, in preference order: the first one present in the vocab
# becomes eos (so chat generation halts on the turn end rather than the formal document
# eos), and every one present is a stop id. Keyed by GGUF arch because these names are
# vocab-specific -- gemma4 ends a turn with <turn|>, Qwen with <|im_end|>. An arch absent
# here falls back to tokenizer.ggml.eos_token_id alone.
_STOP_TOKENS: dict[str, tuple[str, ...]] = {
    "gemma4": ("<turn|>", "<eos>"),
    "qwen35moe": ("<|im_end|>", "<|endoftext|>"),
    # Dense sibling: same vocab and same chat markers as the MoE variant.
    "qwen35": ("<|im_end|>", "<|endoftext|>"),
}


def _convert_gguf_tokenizer(architecture: str, tokenizer_dict: dict):
    if architecture == "qwen2":
        from transformers import AddedToken, Qwen2Tokenizer

        vocab = {token: index for index, token in enumerate(tokenizer_dict["tokens"])}
        merges = [tuple(merge.split(" ")) for merge in tokenizer_dict["merges"]]
        backend = Qwen2Tokenizer(vocab=vocab, merges=merges).backend_tokenizer
        backend.add_special_tokens([
            AddedToken(token, normalized=False, special=True)
            for token in ("<|endoftext|>", "<|im_start|>", "<|im_end|>") if token in vocab
        ])
        return backend, {}
    try:
        from transformers.integrations.gguf import convert_gguf_tokenizer
    except ModuleNotFoundError as exc:
        if exc.name != "transformers.integrations.gguf":
            raise
        from transformers.integrations.ggml import convert_gguf_tokenizer
    else:
        tokenizer_dict = dict(tokenizer_dict)
        tokenizer_dict["tokenizer_type"] = tokenizer_dict.get("model")
        tokenizer_dict["pre_tokenizer_type"] = tokenizer_dict.get("pre")
    return convert_gguf_tokenizer(architecture, tokenizer_dict)


def load_gguf_tokenizer(model_path: str):
    from transformers import AddedToken, PreTrainedTokenizerFast

    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    conv_arch = _TOKENIZER_ARCH.get(arch, arch)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    fast, _extra = _convert_gguf_tokenizer(conv_arch, tok_dict)

    tokens = tok_dict["tokens"]
    if arch in ("qwen35", "qwen35moe"):
        # Transformers' Qwen GGUF converter hardcodes only endoftext/im_start/im_end
        # and ignores GGUF USER_DEFINED (type 4) entries. Qwen3.5 stores its tool and
        # thinking delimiters as type 4: keep them non-special (parsers must see them)
        # while restoring atomic AddedToken matching at their existing vocab ids.
        token_types = tok_dict.get("token_type", ())
        if len(token_types) != len(tokens):
            raise ValueError("GGUF tokenizer token_type length does not match tokens")
        fast.add_tokens(
            [
                AddedToken(tokens[index], normalized=False, special=False)
                for index, token_type in enumerate(token_types)
                if int(token_type) == 4
            ]
        )

    def tok_for(id_key: str, default: str | None) -> str | None:
        """The token named by ``tokenizer.ggml.<id_key>``, else ``default`` if it is in the
        vocab. A default that is not in this vocab is dropped: handing
        PreTrainedTokenizerFast an unknown special token would append it to the vocab and
        shift nothing but confuse decoding (Qwen has no <unk> at all)."""
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        if tid is not None and int(tid) < len(tokens):
            return tokens[int(tid)]
        return default if default is not None and default in tokens else None

    # Prefer the chat turn end as eos so chat generation halts there; the formal document
    # eos stays a stop id (see gguf_eos_token_ids).
    turn_end = next((t for t in _STOP_TOKENS.get(arch, ()) if t in tokens), None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        unk_token=tok_for("unknown_token_id", "<unk>"),
        pad_token=tok_for("padding_token_id", "<pad>"),
    )
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal <eos> plus the chat turn end <turn|>."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    eid = meta.get("tokenizer.ggml.eos_token_id")
    if eid is not None:
        ids.add(int(eid))
    # Look the stop tokens up in the vocab directly (convert_tokens_to_ids would map an
    # absent name to <unk>, wrongly adding it as a stop id). Names are per-arch: gemma4's
    # <eos>/<turn|> do not exist in a Qwen vocab and vice versa.
    for name in _STOP_TOKENS.get(gguf_architecture(model_path), ()):
        try:
            ids.add(tokens.index(name))
        except ValueError:
            pass
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
