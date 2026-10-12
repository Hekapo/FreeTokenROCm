"""Generated-token penalties use request histories, independent of batch row and host acks."""

import gc
import os
import weakref
from collections import Counter
from contextlib import nullcontext

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine import sample
from freetoken.scheduler.prefill import ChunkedReq


@pytest.fixture
def cpu_sampler(monkeypatch):
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(
        sample, "make_device_tensor", lambda data, dtype, device: torch.tensor(data, dtype=dtype, device=device)
    )
    return sample.Sampler(torch.device("cpu"), vocab_size=4)


def _req(params, *, cls=Req, output_len=8):
    return cls(
        input_ids=torch.tensor([0, 0, 0], dtype=torch.int32), table_idx=1, cached_len=0,
        output_len=output_len, uid=7, sampling_params=params, cache_handle=None,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("presence, frequency", [(0.75, 0.5), (-0.75, -0.5), (0.75, -0.5)])
def test_penalties_follow_generated_history_without_host_ack(cpu_sampler, dtype, presence, frequency):
    req = _req(SamplingParams(presence_penalty=presence, frequency_penalty=frequency))
    logits = torch.tensor([[4, 3, 2, 1]], dtype=dtype)
    original = logits.clone()
    history = Counter()
    for _ in range(6):
        expected_logits = [
            value - frequency * history[token] - (presence if history[token] else 0)
            for token, value in enumerate([4, 3, 2, 1])
        ]
        expected = max(range(4), key=expected_logits.__getitem__)
        args = cpu_sampler.prepare(Batch([req], phase="decode"))
        token = cpu_sampler.sample(logits, args).item()
        assert token == expected
        history[token] += 1
        assert req.output_token_counts.tolist() == [history[i] for i in range(4)]
        assert req.input_ids.tolist() == [0, 0, 0]
        assert torch.equal(logits, original)


@pytest.mark.parametrize("greedy", [True, False])
def test_zero_penalties_preserve_sampling_and_allocate_no_counts(cpu_sampler, monkeypatch, greedy):
    seen = []

    def probability_sampler(logits, *args):
        seen.append(logits.clone())
        return torch.tensor([2], dtype=torch.int32)

    monkeypatch.setattr(sample, "sample_impl", probability_sampler)
    req = _req(SamplingParams(temperature=0.0 if greedy else 0.7))
    args = cpu_sampler.prepare(Batch([req], phase="decode"))
    logits = torch.tensor([[4, 3, 2, 1]], dtype=torch.bfloat16)
    assert cpu_sampler.sample(logits, args).item() == (0 if greedy else 2)
    assert req.output_token_counts is None
    assert args.penalties == []
    if greedy:
        assert seen == []
    else:
        assert torch.equal(seen[0], logits.float())


def test_mixed_greedy_penalties_survive_batch_reordering(cpu_sampler, monkeypatch):
    seen = []

    def probability_sampler(logits, *args):
        seen.append(logits.clone())
        return torch.full((logits.shape[0],), 3, dtype=torch.int32)

    monkeypatch.setattr(sample, "sample_impl", probability_sampler)
    greedy = _req(SamplingParams(temperature=0.7, top_k=1, presence_penalty=2, frequency_penalty=0.25))
    random = _req(SamplingParams(temperature=0.7, top_k=2, top_p=0.9,
                               presence_penalty=-0.5, frequency_penalty=-0.25))
    logits = torch.tensor([[4, 3, 2, 1], [4, 3, 2, 1]], dtype=torch.bfloat16)
    args = cpu_sampler.prepare(Batch([greedy, random], phase="decode"))
    assert cpu_sampler.sample(logits, args).tolist() == [0, 3]

    args = cpu_sampler.prepare(Batch([random, greedy], phase="decode"))
    assert cpu_sampler.sample(logits, args).tolist() == [3, 1]
    torch.testing.assert_close(seen[1], torch.tensor([[4, 3, 2, 1.75], [1.75, 3, 2, 1]]))
    assert greedy.output_token_counts.tolist() == [1, 1, 0, 0]
    assert random.output_token_counts.tolist() == [0, 0, 0, 2]

    args = cpu_sampler.prepare(Batch([random], phase="decode"))
    assert cpu_sampler.sample(logits[:1], args).tolist() == [3]
    assert random.output_token_counts.tolist() == [0, 0, 0, 3]


def test_intermediate_prefill_and_exhausted_requests_have_no_counters(cpu_sampler):
    params = SamplingParams(presence_penalty=1, frequency_penalty=1)
    for req in [_req(params, cls=ChunkedReq), _req(params, output_len=0)]:
        args = cpu_sampler.prepare(Batch([req], phase="prefill"))
        assert args.penalties == []
        assert req.output_token_counts is None


def test_final_token_is_counted_after_forward_consumes_budget(cpu_sampler):
    req = _req(SamplingParams(presence_penalty=1), output_len=1)
    args = cpu_sampler.prepare(Batch([req], phase="prefill"))
    req.complete_one()
    assert not req.can_decode
    assert cpu_sampler.sample(torch.tensor([[4., 3., 2., 1.]]), args).item() == 0
    assert req.output_token_counts.tolist() == [1, 0, 0, 0]


def test_counters_end_with_request_and_do_not_follow_reused_uid_or_table(cpu_sampler):
    req = _req(SamplingParams(presence_penalty=1))
    args = cpu_sampler.prepare(Batch([req], phase="decode"))
    cpu_sampler.sample(torch.tensor([[4., 3., 2., 1.]]), args)
    counts = weakref.ref(req.output_token_counts)
    del req
    gc.collect()
    assert counts() is not None
    del args
    gc.collect()
    assert counts() is None

    req = _req(SamplingParams(presence_penalty=1))
    cpu_sampler.prepare(Batch([req], phase="prefill"))
    assert req.output_token_counts.tolist() == [0, 0, 0, 0]


@pytest.mark.skipif(os.environ.get("FT_TEST_PENALTY_GPU") != "1", reason="GPU gate deferred to hardware window")
def test_gpu_greedy_penalties_and_sampling_stream_order():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    sampler = sample.Sampler(torch.device("cuda:0"), vocab_size=4)
    req = _req(SamplingParams(presence_penalty=2, frequency_penalty=0.25))
    batch = Batch([req], phase="decode")
    metadata_stream, sampling_stream = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(metadata_stream):
        first_args = sampler.prepare(batch)
    with torch.cuda.stream(sampling_stream):
        sampling_stream.wait_stream(metadata_stream)
        logits = torch.tensor([[4, 3, 2, 1]], device="cuda", dtype=torch.bfloat16)
        first = sampler.sample(logits, first_args)
    with torch.cuda.stream(metadata_stream):
        next_args = sampler.prepare(batch)
    with torch.cuda.stream(sampling_stream):
        sampling_stream.wait_stream(metadata_stream)
        second = sampler.sample(logits, next_args)
    sampling_stream.synchronize()
    assert first.item() == 0 and second.item() == 1
    assert req.output_token_counts.cpu().tolist() == [1, 1, 0, 0]
    assert req.input_ids.tolist() == [0, 0, 0]
    assert logits.cpu().tolist() == [[4, 3, 2, 1]]
