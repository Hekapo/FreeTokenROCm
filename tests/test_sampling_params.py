import pytest

from freetoken.core import SamplingParams


@pytest.mark.parametrize(
    ("params", "expected"),
    (
        (SamplingParams(temperature=0.0, top_k=-1, top_p=0.95), True),
        (SamplingParams(temperature=-1.0, top_k=50, top_p=0.1), True),
        (SamplingParams(temperature=0.7, top_k=1, top_p=1.0), True),
        (SamplingParams(temperature=0.7, top_k=1, top_p=0.95), True),
        (SamplingParams(temperature=0.7, top_k=-1, top_p=1.0), False),
    ),
)
def test_is_greedy_contract(params, expected):
    assert params.is_greedy is expected


@pytest.mark.parametrize("name", ["presence_penalty", "frequency_penalty"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -2.01, 2.01])
def test_penalties_reject_nonfinite_or_out_of_range_values(name, value):
    with pytest.raises(ValueError, match=name):
        SamplingParams(**{name: value})


@pytest.mark.parametrize("value", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_penalty_bounds_are_inclusive(value):
    params = SamplingParams(presence_penalty=value, frequency_penalty=value)
    assert params.presence_penalty == params.frequency_penalty == value
