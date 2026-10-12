from types import MethodType

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache


def _cache(**kwargs):
    return OffloadMoeCache(num_layers=2, num_experts=16, cache_size=24,
        device=torch.device('cpu'), quant_format='gguf', gguf_expert_types=(3, 3), **kwargs)


def test_prefill_demand_preserves_default():
    cache = _cache()
    assert cache.prefill_demand_tokens == 0
    assert not cache.loads_prefill_on_demand(2, 8, layer_id=0)


@pytest.mark.parametrize('value', [-1, 33, 1.5])
def test_prefill_demand_rejects_unqualified_threshold(value):
    with pytest.raises(ValueError, match='0.*32'):
        _cache(prefill_demand_tokens=value)


@pytest.mark.parametrize('change,tokens,routes,expected', [
    ({}, 3, 12, True),
    ({}, 0, 0, False),
    ({}, 33, 12, False),
    ({}, 3, 16, False),
    ({'cache_size': 8}, 3, 12, False),
    ({'num_experts': 512, 'cache_size': 1024}, 32, 264, False),
    ({'prefill_overlap': True}, 3, 12, False),
    ({'decode_target': 'hybrid'}, 3, 12, False),
    ({'decode_target': 'cpu'}, 3, 12, False),
    ({'cpu_layer_ids': frozenset({0})}, 3, 12, False),
    ({'quant_format': 'bf16'}, 3, 12, False),
    ({'gguf_expert_types': (2, 3)}, 3, 12, False),
    ({'device': torch.device('cpu')}, 3, 12, False),
])
def test_prefill_demand_gate(monkeypatch, change, tokens, routes, expected):
    cache = _cache(prefill_demand_tokens=32)
    cache.device = torch.device('cuda')
    monkeypatch.setattr(torch.version, 'hip', '7.15.0')
    for name, value in change.items():
        setattr(cache, name, value)
    assert cache.loads_prefill_on_demand(tokens, routes, layer_id=0) is expected
    monkeypatch.setattr(torch.version, 'hip', None)
    assert not cache.loads_prefill_on_demand(tokens, routes, layer_id=0)


def test_short_prefill_uses_remapped_slot_banks_and_prefill_dispatch(monkeypatch):
    from freetoken.layers.moe import OffloadMoELayer

    cache = _cache(prefill_demand_tokens=32)
    cache.device = torch.device('cuda')
    monkeypatch.setattr(torch.version, 'hip', '7.15.0')
    layer = object.__new__(OffloadMoELayer)
    layer.offload_cache, layer.layer_id, layer.num_experts = cache, 0, 16
    layer.quant_method = None
    ids = torch.tensor([[2, 7], [7, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
    x = torch.randn(2, 8)
    bank = torch.randn(24, 8)
    calls = []
    def admit(layer_id, got_ids, *, counts_as_decode=True):
        assert layer_id == 0 and not counts_as_decode
        assert got_ids is ids
        got_ids.copy_(torch.tensor([[18, 21], [21, 17]], dtype=torch.int32))
        calls.append('admit')
    monkeypatch.setattr(cache, 'ensure_experts', admit)
    monkeypatch.setattr(cache, 'copy_missing', lambda: calls.append('copy'))
    monkeypatch.setattr(cache, 'bank_views', lambda n=None: (bank,))
    monkeypatch.setattr(cache, 'alphas_for_slots', lambda layer_id: None)
    monkeypatch.setattr(cache, 'materialize_layer', lambda layer_id: pytest.fail('whole layer copied'))
    def compute(self, got_cache, hidden, got_weights, got_ids, **kwargs):
        assert got_cache is cache and hidden is x and got_weights is weights
        assert kwargs == {'views': (bank,), 'n': None, 'alphas': None, 'is_prefill': True}
        calls.append('compute')
        return (bank[got_ids.long()] * got_weights[..., None]).sum(1)
    layer._expert_gemm = MethodType(compute, layer)
    out = layer._prefill_routed(x, weights, ids)
    assert calls == ['admit', 'copy', 'compute']
    torch.testing.assert_close(out, (bank[ids.long()] * weights[..., None]).sum(1))
    assert ids[0, 1] == ids[1, 0]


def test_prefill_admission_keeps_decode_counters_and_frequency_unchanged(monkeypatch):
    import freetoken.moe.offload_kernels as kernels

    cache = _cache()
    cache.collect_stats = cache.collect_decode_freq = True
    cache.decode_freq = torch.zeros(2, 16, dtype=torch.int64)
    captured = []
    monkeypatch.setattr(kernels, 'lru_ensure', lambda *args, **kwargs: captured.append(kwargs['stats']))
    ids = torch.tensor([[2, 7], [7, 3]], dtype=torch.int32)
    cache.ensure_experts(0, ids, counts_as_decode=False)
    assert captured[-1] is None and not cache.decode_freq.any()
    assert not cache.lru_stats.any() and not cache._pending_whole_layer
    cache.ensure_experts(0, ids)
    assert captured[-1] is not None
    assert cache.decode_freq[0, 7] == 2 and cache.decode_freq.sum() == 4


def test_cli_exposes_bounded_default_off_threshold():
    from freetoken.server.args import parse_args

    args = ['--model', 'unused', '--dtype', 'bfloat16', '--tool-call-parser', 'llama3', '--reasoning-parser', 'off']
    assert parse_args(args)[0].moe_prefill_demand_tokens == 0
    assert parse_args(args + ['--moe-prefill-demand-tokens', '32'])[0].moe_prefill_demand_tokens == 32
    for value in ('-1', '33'):
        with pytest.raises(SystemExit):
            parse_args(args + ['--moe-prefill-demand-tokens', value])


def _q4_1_banks(experts, rows, columns, phase):
    packed = torch.empty(experts, rows, columns // 32, 20, dtype=torch.uint8)
    e = torch.arange(experts)[:, None, None]
    r = torch.arange(rows)[None, :, None]
    b = torch.arange(columns // 32)[None, None, :]
    scale = (0.0625 * (1 + (e + r + b + phase) % 7)).to(torch.float16)
    minimum = (-0.25 * (1 + (e + b + phase) % 5)).expand_as(scale).to(torch.float16).contiguous()
    packed[..., :2] = scale.contiguous().view(torch.uint8).reshape(experts, rows, -1, 2)
    packed[..., 2:4] = minimum.view(torch.uint8).reshape(experts, rows, -1, 2)
    low = (torch.arange(16)[None, None, None, :] + r[..., None] + e[..., None] + phase) % 16
    high = (15 - torch.arange(16)[None, None, None, :] + 2 * e[..., None] + r[..., None] + phase) % 16
    packed[..., 4:] = (low | (high << 4)).to(torch.uint8)
    dense = torch.cat((low, high), -1).float() * scale[..., None].float() + minimum[..., None].float()
    return packed.reshape(experts, rows, -1).pin_memory(), dense.reshape(experts, rows, columns)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason='ROCm operator gate')
@pytest.mark.parametrize('experts,slots,tokens,routes', [(16, 24, 3, 4), (256, 2048, 31, 8)])
def test_native_q4_1_prefill_matches_full_layer_after_hits_and_eviction(experts, slots, tokens, routes):
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    hidden, intermediate = 64, 32
    sources, dense = {}, {}
    for name, rows, columns in [('gate_up', 2 * intermediate, hidden), ('down', hidden, intermediate)]:
        pairs = [_q4_1_banks(experts, rows, columns, phase) for phase in (0, 9)]
        sources[name], dense[name] = [p[0] for p in pairs], [p[1] for p in pairs]
    cache = OffloadMoeCache(num_layers=2, num_experts=experts, cache_size=slots, device=torch.device('cuda'),
        quant_format='gguf', gguf_expert_types=(3, 3), prefill_demand_tokens=32)
    cache.set_bank_sources(sources)
    cache.collect_stats = cache.collect_decode_freq = True
    layer = object.__new__(OffloadMoELayer)
    layer.offload_cache, layer.layer_id, layer.num_experts = cache, 0, experts
    layer.quant_method, layer.activation = None, 'silu'
    raw = ((torch.arange(tokens)[:, None] * 3 + torch.arange(routes)[None, :]) % experts).to(torch.int32)
    weights = (torch.arange(1, routes + 1).float() / (routes * (routes + 1) / 2)).expand(tokens, routes).contiguous()
    x = (((torch.arange(tokens * hidden).reshape(tokens, hidden) % 7) - 3) * 0.05).to(torch.bfloat16)
    x_gpu, w_gpu = x.cuda(), weights.cuda()
    whole = fused_experts_gguf(x_gpu, sources['gate_up'][0].cuda(), sources['down'][0].cuda(),
        w_gpu, raw.cuda(), 'silu', 3, 3)
    # Occupy slots with another layer before admitting the current query.
    cache.materialize_layer(1)
    cache.copy_missing()
    def demand():
        ids = raw.cuda()
        actual = layer._prefill_routed(x_gpu, w_gpu, ids)
        torch.cuda.synchronize()
        cache.check_copy_status()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, whole, rtol=0, atol=0)
        assert torch.equal(cache.id_of_slot[ids.long()].cpu(), raw)
        assert not cache.lru_stats.any() and not cache.decode_freq.any()
        return actual
    actual = demand()
    assert int(cache.num_indices.item()) == torch.unique(raw).numel()
    if experts == 16:
        assert (cache.slot_for_id[1] < 0).any(), 'other-layer eviction not exercised'
    demand()
    assert int(cache.num_indices.item()) == 0, 'cache-hit path still fetched experts'
    cache.materialize_layer(1)
    cache.copy_missing()
    demand()
    if experts == 16:
        expected = []
        for token in range(tokens):
            values = []
            for route in range(routes):
                expert = int(raw[token, route])
                up = (dense['gate_up'][0][expert] @ x[token].float()).to(torch.bfloat16)
                inter = (torch.nn.functional.silu(up[:intermediate].float()) * up[intermediate:].float()).to(torch.bfloat16)
                down = (dense['down'][0][expert] @ inter.float()).to(torch.bfloat16)
                values.append((down * weights[token, route].to(torch.bfloat16)).to(torch.bfloat16))
            expected.append(torch.stack(values).sum(0))
        torch.testing.assert_close(actual.cpu(), torch.stack(expected), rtol=2e-2, atol=0.1)
