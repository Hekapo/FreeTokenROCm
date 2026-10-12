import json
from types import SimpleNamespace

import pytest
import torch

from freetoken.diagnostics import FiniteDiagnostics, FirstNonFiniteError, _from_env


def test_disabled_monitor_does_not_inspect_tensors(monkeypatch, tmp_path):
    monkeypatch.delenv("FT_FINITE_DIAGNOSTICS_DIR", raising=False)
    assert _from_env() is None
    monitor = FiniteDiagnostics(tmp_path, start_forward=3, end_forward=4)
    monkeypatch.setattr(torch, "isfinite", lambda _: pytest.fail("inactive diagnostic touched tensor"))
    for phase in ["prefill", "prefill", "decode"]:
        monitor.begin_forward(phase, [])
        monitor.check("unused", {"bad": torch.tensor(float("nan"))})
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_first_nonfinite_is_bounded_and_cannot_resume(tmp_path):
    monitor = FiniteDiagnostics(tmp_path, snapshot_bytes=256 * 1024)
    monitor.begin_forward("prefill", [{"cached_len": 4096, "extend_len": 512}])
    monitor.set_layer(7)
    pool = torch.ones(1000, 1024)
    view = pool[300:301, ::2]
    bad = torch.ones(256, 1024)
    bad[-1, -1] = float("inf")
    with pytest.raises(FirstNonFiniteError, match="gdn.test/out"):
        monitor.check("gdn.test", {"out": bad}, inputs={"live_state": view}, metadata={"BT": 64})
    target, = tmp_path.iterdir()
    report = json.loads((target / "record.json").read_text())
    tensors = torch.load(target / "tensors.pt", weights_only=True)
    assert report["layer"] == 7 and report["forward"] == 1
    assert report["first_bad_index"] == [255, 1023]
    assert torch.isinf(tensors["first_bad_value"])
    assert report["saved"]["output/out"]["truncated"]
    assert sum(p.stat().st_size for p in target.iterdir()) <= monitor.snapshot_bytes
    for tensor in tensors.values():
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    with pytest.raises(FirstNonFiniteError):
        monitor.begin_forward("prefill", [])
    assert len(list(tmp_path.iterdir())) == 1


def test_indexed_mutating_state_capture_keeps_preoperation_values(tmp_path):
    monitor = FiniteDiagnostics(tmp_path)
    monitor.begin_forward("prefill", [])
    source = torch.tensor([[float("nan"), float("nan")], [2., 3.], [4., 5.]])
    indices = torch.tensor([2, 1, 2], dtype=torch.int32)
    selected = monitor.selected_states(source, indices)
    monitor.check("state.input", selected)  # Unselected NaNs are irrelevant.
    saved = monitor.preserve(selected)
    source[2, 1] = float("nan")
    with pytest.raises(FirstNonFiniteError):
        monitor.check("state.update", monitor.selected_states(source, indices), inputs=saved)
    target, = tmp_path.iterdir()
    tensors = torch.load(target / "tensors.pt", weights_only=True)
    torch.testing.assert_close(tensors["input/state_slot_2"], torch.tensor([4., 5.]))


def test_moe_diagnostic_stops_before_routing_when_router_is_bad(monkeypatch, tmp_path):
    from freetoken.models.qwen3_5_moe import moe

    monitor = FiniteDiagnostics(tmp_path)
    monitor.begin_forward("prefill", [])
    monitor.set_layer(9)
    monkeypatch.setattr(moe, "FINITE_DIAGNOSTICS", monitor)
    op = moe.Qwen3_5MoE.__new__(moe.Qwen3_5MoE)
    op.gate = SimpleNamespace(forward=lambda x: torch.full((x.shape[0], 4), float("nan")))
    op.shared_expert = SimpleNamespace(forward=lambda x: pytest.fail("continued past bad router"))
    with pytest.raises(FirstNonFiniteError, match="moe.router"):
        op.forward(torch.ones(2, 3))


def test_snapshot_write_error_does_not_hide_the_nonfinite(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    monitor = FiniteDiagnostics(blocked)
    monitor.begin_forward("prefill", [])
    with pytest.raises(FirstNonFiniteError, match="first nonfinite"):
        monitor.check("operation", {"out": torch.tensor(float("nan"))})


def test_chunk_boundary_keeps_indexed_state_before_inplace_update(monkeypatch, tmp_path):
    from freetoken.kernel.fla import chunk

    monitor = FiniteDiagnostics(tmp_path)
    monitor.begin_forward("prefill", [{"cached_len": 4096, "extend_len": 17}])
    monitor.set_layer(4)
    monkeypatch.setattr(chunk, "FINITE_DIAGNOSTICS", monitor)
    monkeypatch.setattr(chunk, "chunk_local_cumsum", lambda g, **kw: g.cumsum(1))
    q = torch.ones(1, 17, 1, 2)
    v = torch.ones_like(q)
    state = torch.tensor([[[[float("nan"), float("nan")], [float("nan"), float("nan")]]],
                          [[[2., 3.], [4., 5.]]]])
    monkeypatch.setattr(chunk, "chunk_gated_delta_rule_fwd_intra", lambda **kw: (q, v, torch.ones(1)))

    def update(**kwargs):
        kwargs["initial_state"][1, 0, 0, 0] = float("nan")
        return torch.ones(1, 1, 1, 2, 2), v

    monkeypatch.setattr(chunk, "chunk_gated_delta_rule_fwd_h", update)
    monkeypatch.setattr(chunk, "chunk_fwd_o", lambda **kw: pytest.fail("continued past bad state"))
    with pytest.raises(FirstNonFiniteError, match="gdn.state_update/state_slot_1"):
        chunk.chunk_gated_delta_rule_fwd(q, q, v, torch.zeros(1, 17, 1), torch.ones(1, 17, 1),
                                        1., state, torch.tensor([1], dtype=torch.int32))
    target, = tmp_path.iterdir()
    tensors = torch.load(target / "tensors.pt", weights_only=True)
    torch.testing.assert_close(tensors["input/state_slot_1"], torch.tensor([[[2., 3.], [4., 5.]]]))


def test_model_failure_ends_scope_and_does_not_run_another_forward(monkeypatch, tmp_path):
    from freetoken.models.qwen3_5_moe import model

    monitor = FiniteDiagnostics(tmp_path)
    monkeypatch.setattr(model, "FINITE_DIAGNOSTICS", monitor)
    batch = SimpleNamespace(phase="prefill", input_ids=torch.tensor([1, 2]),
                            padded_reqs=[SimpleNamespace(cached_len=4096, extend_len=17)])
    monkeypatch.setattr(model, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
    op = model.Qwen3_5MoEForCausalLM.__new__(model.Qwen3_5MoEForCausalLM)
    op.model = SimpleNamespace(forward=lambda ids: torch.ones(2, 3))
    op.lm_head = SimpleNamespace(forward=lambda hidden: torch.full((2, 8), float("inf")))
    with pytest.raises(FirstNonFiniteError, match="model.lm_head"):
        op.forward()
    assert not monitor.active
    with pytest.raises(FirstNonFiniteError):
        op.forward()
    assert monitor.forward == 1


def test_ragged_location_uses_query_offsets_and_cached_prefix(tmp_path):
    monitor = FiniteDiagnostics(tmp_path)
    monitor.begin_forward("prefill", [{"cached_len": 4096, "extend_len": 17},
                                      {"cached_len": 8192, "extend_len": 65}])
    monitor.metadata["gdn"] = {"cu_seqlens": [0, 17, 82], "indices": [2, 5]}
    out = torch.zeros(1, 82, 1, 2)
    out[0, 69, 0, 1] = float("nan")
    with pytest.raises(FirstNonFiniteError):
        monitor.check("gdn.chunk_output", {"o": out}, metadata={"BT": 64})
    target, = tmp_path.iterdir()
    location = json.loads((target / "record.json").read_text())["location"]
    assert location == {"request": 1, "token_in_forward_sequence": 52,
                        "chunk_in_forward_sequence": 0, "absolute_token": 8244,
                        "state_at_chunk_start": False}


@pytest.mark.parametrize("kwargs", [{"start_forward": 0}, {"start_forward": 4, "end_forward": 3},
                                   {"snapshot_bytes": 65 * 1024 * 1024}])
def test_invalid_configuration_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        FiniteDiagnostics(tmp_path, **kwargs)
