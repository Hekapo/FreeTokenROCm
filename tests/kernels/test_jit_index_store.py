import pytest
import torch

from freetoken.kernel import indexing, store_cache
from freetoken.kernel.index import num_splits_for


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA or ROCm GPU is required"
)


def test_indexing_jit_matches_torch_on_cold_and_warm_loads():
    weights = torch.arange(8 * 64, dtype=torch.float32, device="cuda").reshape(8, 64)

    for values in ([7, 2, 0], [1, 6, 3]):
        indices = torch.tensor(values, dtype=torch.int32, device="cuda")
        actual = indexing(weights, indices)
        torch.testing.assert_close(actual, weights[indices.long()])


@pytest.mark.parametrize(("width", "expected_splits"), [(256, 2), (512, 4)])
def test_indexing_jit_copies_rows_split_across_multiple_warps(width, expected_splits):
    weights = torch.arange(8 * width, dtype=torch.float32, device="cuda").reshape(8, width)
    assert num_splits_for(weights.shape[1] * weights.element_size()) == expected_splits

    for values in ([7, 2, 0], [1, 6, 3]):
        indices = torch.tensor(values, dtype=torch.int64, device="cuda")
        actual = indexing(weights, indices)
        torch.testing.assert_close(actual, weights[indices])


def test_masked_indexing_zeros_indices_outside_vocab_range():
    width = 256
    weights = torch.arange(6 * width, dtype=torch.float32, device="cuda").reshape(6, width)
    indices = torch.tensor([9, 10, 15, 16], dtype=torch.int32, device="cuda")

    actual = indexing(weights, indices, vocab_range=(10, 6))
    expected = torch.stack(
        (torch.zeros_like(weights[0]), weights[0], weights[5], torch.zeros_like(weights[0]))
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_jit_matches_torch_on_cold_and_warm_loads(index_dtype):
    k_cache = torch.zeros((8, 64), dtype=torch.float32, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    indices = torch.tensor([5, 0, 3], dtype=index_dtype, device="cuda")

    for offset in (0.0, 1000.0):
        k = torch.arange(3 * 64, dtype=torch.float32, device="cuda").reshape(3, 64)
        k = k + offset
        v = k + 500.0
        store_cache(k_cache, v_cache, indices, k, v)
        torch.testing.assert_close(k_cache[indices], k)
        torch.testing.assert_close(v_cache[indices], v)

    untouched = torch.tensor([1, 2, 4, 6, 7], device="cuda")
    torch.testing.assert_close(k_cache[untouched], torch.zeros((5, 64), device="cuda"))
    torch.testing.assert_close(v_cache[untouched], torch.zeros((5, 64), device="cuda"))


# The empty and overlap cases below are rejected or skipped on the host before any launch.


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("vocab_range", [None, (10, 6)])
def test_empty_indexing_returns_empty_rows(index_dtype, vocab_range):
    weights = torch.arange(8 * 64, dtype=torch.float32, device="cuda").reshape(8, 64)
    indices = torch.empty((0,), dtype=index_dtype, device="cuda")

    actual = indexing(weights, indices, vocab_range=vocab_range)
    torch.cuda.synchronize()

    assert actual.shape == (0, 64)
    assert actual.dtype == weights.dtype


def test_empty_indexing_still_checks_row_width():
    weights = torch.zeros((8, 64), dtype=torch.float32, device="cuda")
    indices = torch.empty((0,), dtype=torch.int32, device="cuda")
    output = torch.empty((0, 32), dtype=torch.float32, device="cuda")

    with pytest.raises(Exception, match="Tensor match failed"):
        indexing(weights, indices, output=output)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_empty_store_leaves_cache_untouched(index_dtype):
    k_cache = torch.full((8, 64), 7.0, dtype=torch.float32, device="cuda")
    v_cache = torch.full_like(k_cache, 9.0)
    indices = torch.empty((0,), dtype=index_dtype, device="cuda")
    k = torch.empty((0, 64), dtype=torch.float32, device="cuda")

    store_cache(k_cache, v_cache, indices, k, k.clone())
    torch.cuda.synchronize()

    torch.testing.assert_close(k_cache, torch.full_like(k_cache, 7.0))
    torch.testing.assert_close(v_cache, torch.full_like(v_cache, 9.0))


def test_empty_store_still_checks_input_width():
    k_cache = torch.zeros((8, 64), dtype=torch.float32, device="cuda")
    indices = torch.empty((0,), dtype=torch.int32, device="cuda")
    k = torch.empty((0, 32), dtype=torch.float32, device="cuda")

    with pytest.raises(Exception, match="Tensor match failed"):
        store_cache(k_cache, torch.zeros_like(k_cache), indices, k, k.clone())


@pytest.mark.parametrize("row_stride", [0, 32])  # 0 needs D6 to bind a zero stride
def test_store_rejects_aliased_cache_rows(row_stride):
    # overlapping rows: distinct indices would write the same memory
    def aliased():
        storage = torch.zeros(7 * row_stride + 64, dtype=torch.float32, device="cuda")
        return storage.as_strided((8, 64), (row_stride, 1))

    k_cache, v_cache = aliased(), aliased()
    indices = torch.tensor([5, 0, 3], dtype=torch.int32, device="cuda")
    k = torch.ones((3, 64), dtype=torch.float32, device="cuda")

    with pytest.raises(Exception, match="kv cache rows overlap"):
        store_cache(k_cache, v_cache, indices, k, k.clone())
