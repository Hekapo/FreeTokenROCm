"""The exact Triton samplers retry with one CTA per row when a cooperative launch is unavailable."""

import pytest
import torch

import freetoken.kernel.triton.sampling as sampling

AMD_REJECTION = AssertionError("Cooperative launch requested but not supported by device")


@pytest.fixture
def fake_launch(monkeypatch):
    calls = []

    def launch(probs, kernel, tk, tp, draw, seed, offset, force_single=False):
        calls.append(force_single)
        if not force_single:
            raise AMD_REJECTION
        return "single"

    monkeypatch.setattr(sampling, "_fused_launch", launch)
    monkeypatch.setattr(sampling, "_fused_plan", lambda B, V, device, force_single=False: (1 if force_single else 4, V))
    monkeypatch.setattr(sampling, "_COOPERATIVE_DISABLED", set())
    return calls


def test_amd_cooperative_rejection_retries_with_one_cta(fake_launch):
    probs = torch.full((2, 8), 1 / 8)

    assert sampling._exact_launch(probs, sampling._topp_fused, None, None, None, None, None) == "single"
    assert fake_launch == [False, True]
    # later calls go straight to the single-CTA launch
    assert sampling._exact_launch(probs, sampling._topp_fused, None, None, None, None, None) == "single"
    assert fake_launch == [False, True, True]


def test_unrelated_assertion_is_not_swallowed(monkeypatch, fake_launch):
    def launch(*args, force_single=False, **kwargs):
        raise AssertionError("shape mismatch")

    monkeypatch.setattr(sampling, "_fused_launch", launch)
    with pytest.raises(AssertionError, match="shape mismatch"):
        sampling._exact_launch(torch.full((2, 8), 1 / 8), sampling._topp_fused, None, None, None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("vocab", [1000, 151936])
def test_top_p_sampling_returns_a_token_inside_the_nucleus(vocab):
    torch.manual_seed(0)
    logits = torch.randn(3, vocab, device="cuda") * 4
    probs = torch.softmax(logits, dim=-1)
    top_p = torch.tensor([0.5, 0.9, 1.0], device="cuda")

    tokens = sampling.top_p_sampling_from_probs(probs, top_p).long()

    sorted_probs, order = probs.sort(dim=-1, descending=True)
    for row in range(probs.size(0)):
        rank = (order[row] == tokens[row]).nonzero().item()
        mass_before = sorted_probs[row, :rank].sum().item()
        assert mass_before < top_p[row].item() + 1e-4
