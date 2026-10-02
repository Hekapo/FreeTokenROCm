from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("lengths", [(3, 5), (512, 37), (17,)])
def test_gdn_prefill_uses_host_query_lengths(monkeypatch, lengths):
    """The production call must not recover the launch grid from device metadata."""
    from freetoken.models.qwen3_5_moe import gdn
    from freetoken.attention.linear import FLAMetadata
    from freetoken.kernel import backend
    from freetoken.kernel.triton import causal_conv1d_triton

    total = sum(lengths)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    fla = FLAMetadata(cu_seqlens=torch.tensor(offsets),
                      cache_indices=torch.arange(len(lengths), dtype=torch.int32),
                      has_initial_state=torch.ones(len(lengths), dtype=torch.bool))
    batch = SimpleNamespace(is_decode=False, fla_metadata=fla,
                            padded_reqs=[SimpleNamespace(extend_len=n, cached_len=8192)
                                         for n in lengths])
    pool = SimpleNamespace(local_index=lambda _: 0,
                           conv_states=torch.zeros(1, len(lengths), 3, 3),
                           recurrent_states=torch.zeros(1, len(lengths), 1, 1, 1))
    monkeypatch.setattr(gdn, "get_global_ctx", lambda: SimpleNamespace(batch=batch, linear_state_pool=pool))
    monkeypatch.setattr(backend, "is_sgl_kernel_installed", lambda: False)
    calls = []

    def convolution(x, weight, states, cu_seqlens, indices, initial, **kwargs):
        assert kwargs["max_seq_len"] == max(lengths)
        calls.append(kwargs["max_seq_len"])
        return x

    monkeypatch.setattr(causal_conv1d_triton, "causal_conv1d_varlen", convolution)
    monkeypatch.setattr(gdn, "gdn_prefill_chunk_fla", lambda *args, **kwargs: torch.zeros(total, 1, 1))
    op = gdn.Qwen3_5GatedDeltaNet.__new__(gdn.Qwen3_5GatedDeltaNet)
    op.layer_id, op._split_in_proj = 0, False
    op.conv_dim, op.key_dim, op.value_dim = 3, 1, 1
    op.num_k_heads = op.num_v_heads = op.head_k_dim = op.head_v_dim = 1
    op._in_proj_split = [3, 1, 1, 1]
    op.in_proj = SimpleNamespace(forward=lambda x: torch.zeros(total, 6))
    op.conv1d = SimpleNamespace(weight=torch.ones(3, 1, 4))
    op.A_log, op.dt_bias = torch.zeros(1), torch.zeros(1)
    op.norm = SimpleNamespace(forward=lambda x, z: x)
    op.out_proj = SimpleNamespace(forward=lambda x: x)
    assert op.forward(torch.zeros(total, 1)).shape == (total, 1)
    assert calls == [max(lengths)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm")
@pytest.mark.parametrize("continuation", [False, True])
def test_ragged_convolution_without_device_scalar_read(monkeypatch, continuation):
    from freetoken.kernel import backend
    from freetoken.kernel.causal_conv1d import causal_conv1d_varlen

    monkeypatch.setattr(backend, "is_sgl_kernel_installed", lambda: False)
    torch.manual_seed(94)
    lengths, channels = [37, 512], 16
    x = torch.randn(channels, sum(lengths), device="cuda")
    weight = torch.randn(channels, 4, device="cuda")
    states = torch.randn(4, channels, 3, device="cuda")
    before = states.clone()
    cu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32, device="cuda")
    indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    initial = torch.full((2,), continuation, dtype=torch.bool, device="cuda")
    original_item = torch.Tensor.item

    def reject_device_item(tensor, *args, **kwargs):
        if tensor.is_cuda:
            raise AssertionError("device scalar read in causal convolution launch")
        return original_item(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", reject_device_item)
        out = causal_conv1d_varlen(x, weight, states, cu, indices, initial, max_seq_len=max(lengths))
    offset = 0
    for slot, length in zip([1, 3], lengths):
        prefix = before[slot] if continuation else torch.zeros_like(before[slot])
        segment = x[:, offset:offset + length]
        window = torch.cat([prefix, segment], dim=1)
        expected = F.silu(F.conv1d(window.unsqueeze(0), weight.unsqueeze(1), groups=channels))[0]
        torch.testing.assert_close(out[:, offset:offset + length], expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(states[slot], window[:, -3:])
        offset += length
    torch.testing.assert_close(states[[0, 2]], before[[0, 2]])
