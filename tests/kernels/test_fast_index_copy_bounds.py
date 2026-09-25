"""Bounds guard of the expert copy kernels.

Safe by construction: every GPU case passes the new ``status`` argument, so a module built
from sources without the guard rejects the call on argument count before any launch.
"""

from types import SimpleNamespace

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA or ROCm GPU is required")

BAD_COUNT, BAD_INDEX = 1, 2


def _bank(rows, width=32, fill=0.0):
    return torch.full((rows, width), fill, dtype=torch.float32, device="cuda")


def _idx(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _count(value):
    return torch.tensor([value], dtype=torch.int64, device="cuda")


def _status():
    return torch.zeros((1,), dtype=torch.int32, device="cuda")


def _per_bank(dst, dst_idx, src, src_idx, count, status):
    from freetoken.kernel import fast_index_copy_jit

    fast_index_copy_jit(dst, dst_idx, src, src_idx, count, status=status)
    torch.cuda.synchronize()


@CUDA
def test_valid_copy_reports_nothing():
    src = torch.arange(6 * 32, dtype=torch.float32, device="cuda").reshape(6, 32)
    dst, status = _bank(4), _status()
    _per_bank(dst, _idx([0, 3]), src, _idx([5, 1]), _count(2), status)
    assert int(status.item()) == 0
    torch.testing.assert_close(dst[[0, 3]], src[[5, 1]])


@CUDA
@pytest.mark.parametrize("count", [3, 1 << 40, -1])
def test_count_outside_indices_copies_nothing(count):
    src, dst, status = _bank(6, fill=1.0), _bank(4), _status()
    _per_bank(dst, _idx([0, 1]), src, _idx([2, 3]), _count(count), status)
    assert int(status.item()) & BAD_COUNT
    torch.testing.assert_close(dst, torch.zeros_like(dst))


@CUDA
@pytest.mark.parametrize(("dst_idx", "src_idx"), [([0, 1], [2, 6]), ([0, -1], [2, 3]), ([0, 4], [2, 3])])
def test_out_of_range_rows_are_skipped_and_the_rest_copied(dst_idx, src_idx):
    src = torch.arange(6 * 32, dtype=torch.float32, device="cuda").reshape(6, 32)
    dst, status = _bank(4, fill=-1.0), _status()
    _per_bank(dst, _idx(dst_idx), src, _idx(src_idx), _count(2), status)
    assert int(status.item()) & BAD_INDEX
    torch.testing.assert_close(dst[0], src[2])  # the valid first pair still lands
    untouched = [r for r in range(4) if r != 0]
    torch.testing.assert_close(dst[untouched], torch.full((3, 32), -1.0, device="cuda"))


@CUDA
def test_guard_does_not_need_a_status_word():
    src, dst = _bank(6, fill=1.0), _bank(4)
    _per_bank(dst, _idx([0, 1]), src, _idx([2, 3]), _count(1 << 40), None)
    torch.testing.assert_close(dst, torch.zeros_like(dst))


@CUDA
def test_status_bits_are_sticky_across_launches():
    src, dst, status = _bank(6, fill=1.0), _bank(4), _status()
    _per_bank(dst, _idx([0, 1]), src, _idx([2, 3]), _count(-5), status)
    _per_bank(dst, _idx([0, 1]), src, _idx([9, 3]), _count(2), status)
    assert int(status.item()) == BAD_COUNT | BAD_INDEX


@CUDA
def test_fused_copy_bounds_rows_and_count():
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

    feat = 32 * 4  # bytes per row, a multiple of 16
    src = torch.arange(6 * 32, dtype=torch.float32, device="cuda").reshape(6, 32)
    dst = _bank(4, fill=-1.0)
    ptr = lambda t: torch.tensor([t.data_ptr()], dtype=torch.int64, device="cuda")
    feats = torch.tensor([feat], dtype=torch.int64, device="cuda")
    status = _status()

    fast_index_copy_multi_jit(ptr(dst), ptr(src), feats, _idx([0, 1]), _idx([2, 6]), _count(2),
                              dst_rows=4, src_rows=6, status=status)
    fast_index_copy_multi_jit(ptr(dst), ptr(src), feats, _idx([0, 1]), _idx([2, 3]), _count(7),
                              dst_rows=4, src_rows=6, status=status)
    torch.cuda.synchronize()

    assert int(status.item()) == BAD_COUNT | BAD_INDEX
    torch.testing.assert_close(dst[0], src[2])
    torch.testing.assert_close(dst[1:], torch.full((3, 32), -1.0, device="cuda"))


@pytest.mark.parametrize(("bits", "fragment"), [(1, "count"), (2, "index"), (3, "count")])
def test_check_copy_status_decodes_and_resets(bits, fragment):
    from freetoken.moe.offload_cache import OffloadMoeCache

    fake = SimpleNamespace(copy_status=torch.tensor([bits], dtype=torch.int32))
    with pytest.raises(RuntimeError, match=fragment):
        OffloadMoeCache.check_copy_status(fake)
    assert int(fake.copy_status.item()) == 0
    OffloadMoeCache.check_copy_status(fake)  # clean word: no error
