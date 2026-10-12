"""CPU interpreter regression for undefined padding in the production FLA kernels.

Run in a child process: TRITON_INTERPRET must precede importing JIT functions.
This checks kernel arithmetic, not GPU compilation, LDS usage or performance.
"""

import os
from pathlib import Path
import subprocess
import sys


def test_gdn_partial_tiles_with_poisoned_undefined_loads():
    env = dict(os.environ, TRITON_INTERPRET="1", OMP_NUM_THREADS="1",
               HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1", CUDA_VISIBLE_DEVICES="-1")
    result = subprocess.run([sys.executable, "-B", str(Path(__file__))], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "padding/reference PASS" in result.stdout


def test_gpu_gdn_partial_chunk_and_recurrent_model_geometry():
    import pytest
    if os.environ.get("FT_TEST_STAGE2_GPU") != "1":
        pytest.skip("opt-in model-geometry operator gate")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from freetoken.kernel.fla.chunk import chunk_gated_delta_rule
    from freetoken.kernel.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update

    torch.manual_seed(813)
    H, Hg, K, V, T = 32, 16, 128, 128, 65
    q = (torch.randn(1, T, Hg, K) * 0.1).bfloat16()
    k = (torch.randn_like(q.float()) * 0.1).bfloat16()
    v = (torch.randn(1, T, H, V) * 0.1).bfloat16()
    g = -torch.rand(1, T, H) * 0.04
    beta = torch.rand(1, T, H) * 0.5
    state = torch.full((4, H, V, K), float("nan"))
    state[2] = torch.randn(H, V, K) * 0.02
    reference = state[2].double()
    expected = torch.empty(1, T, H, V, dtype=torch.float64)
    qn = (q.float() / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16().double()
    kn = (k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16().double()
    for t in range(T):
        qt, kt = qn[0, t].repeat_interleave(H // Hg, dim=0), kn[0, t].repeat_interleave(H // Hg, dim=0)
        reference *= g[0, t].double().exp()[:, None, None]
        delta = (v[0, t].double() - (reference * kt[:, None, :]).sum(-1)) * beta[0, t].double()[:, None]
        reference += delta[:, :, None] * kt[:, None, :]
        expected[0, t] = (reference * qt[:, None, :]).sum(-1) * K ** -0.5
    pool = state.to("cuda")
    slots = torch.tensor([2], device="cuda", dtype=torch.int32)
    actual, _, _ = chunk_gated_delta_rule(
        q.to("cuda"), k.to("cuda"), v.to("cuda"), g.to("cuda"), beta.to("cuda"),
        initial_state=pool, initial_state_indices=slots,
        cu_seqlens=torch.tensor([0, T], device="cuda", dtype=torch.int64), use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=0.025, atol=0.001)
    torch.testing.assert_close(pool[2].cpu().double(), reference, rtol=0.025, atol=0.001)
    assert torch.isnan(pool[[0, 1, 3]]).all()

    qd, kd, vd = q[:, :1].to("cuda"), k[:, :1].to("cuda"), v[:, :1].to("cuda")
    qt = q[0, 0].double()
    kt = k[0, 0].double()
    qt = (qt / (qt.square().sum(-1, keepdim=True) + 1e-6).sqrt()).repeat_interleave(H // Hg, dim=0)
    kt = (kt / (kt.square().sum(-1, keepdim=True) + 1e-6).sqrt()).repeat_interleave(H // Hg, dim=0)
    reference *= torch.nn.functional.softplus(torch.tensor(-4., dtype=torch.float64)).neg().exp()
    delta = (v[0, 0].double() - (reference * kt[:, None, :]).sum(-1)) * 0.5
    reference += delta[:, :, None] * kt[:, None, :]
    expected_decode = (reference * qt[:, None, :]).sum(-1) * K ** -0.5
    actual_decode = fused_sigmoid_gating_delta_rule_update(
        torch.zeros(H, device="cuda"), torch.full((1, 1, H), -4., device="cuda"),
        torch.zeros(H, device="cuda"), 1., 20., qd, kd, vd, torch.zeros(1, 1, H, device="cuda"),
        pool, slots, use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(actual_decode[0, 0].cpu().double(), expected_decode, rtol=0.025, atol=0.001)
    torch.testing.assert_close(pool[2].cpu().double(), reference, rtol=0.025, atol=0.001)
    assert torch.isnan(pool[[0, 1, 3]]).all()


def _probe():
    import numpy as np
    import torch
    import triton.language as tl
    import triton.runtime.interpreter as interpreter
    from freetoken.kernel.fla import chunk_delta_h, chunk_fwd, chunk_o, cumsum, l2norm, op, wy_fast

    torch.set_num_threads(1)
    # Interpreter patches module attributes, but not saved aliases to tl.exp.
    # Resolve the same operation at call time; leave the production JIT unchanged.
    assert os.environ.get("FLA_USE_FAST_OPS", "0") == "0"
    op.exp = chunk_delta_h.exp = chunk_o.exp = lambda x: tl.exp(x)
    original = interpreter.InterpreterBuilder.create_masked_load
    poisoned = [0]

    def poison(self, ptrs, mask, other, *args):
        # Undefined float loads may return NaN. The interpreter ordinarily
        # substitutes zero and would hide this bug. Explicit zero is preserved.
        dtype = interpreter._get_np_dtype(ptrs.get_element_ty())
        if other is None and np.issubdtype(dtype, np.floating):
            poisoned[0] += int(np.count_nonzero(~mask.data))
            other = interpreter.TensorHandle(np.full(ptrs.data.shape, np.nan, dtype=dtype),
                                             ptrs.get_element_ty())
        return original(self, ptrs, mask, other, *args)

    interpreter.InterpreterBuilder.create_masked_load = poison

    def bare(kernel):
        while hasattr(kernel, "fn") and not isinstance(kernel, interpreter.InterpretedFunction):
            kernel = kernel.fn
        return kernel

    # Compact edge geometry and the model's H/Hg/K/V geometry. Unequal
    # sequences cross 16/64 boundaries and resume nonzero, reordered slots.
    for lengths, H, Hg, K, V in [([1, 15, 16, 17, 63, 64, 65], 2, 1, 17, 19),
                                  ([17, 65], 32, 16, 128, 128)]:
        torch.manual_seed(813)
        T, BT = sum(lengths), 64
        q = torch.randn(1, T, Hg, K, dtype=torch.float16) * 0.1
        k = torch.randn_like(q) * 0.1
        q, k = l2norm.l2norm_fwd(q), l2norm.l2norm_fwd(k)
        assert torch.isfinite(q).all() and torch.isfinite(k).all(), "l2norm padding"
        v = torch.randn(1, T, H, V, dtype=torch.float16) * 0.1
        beta = torch.rand(1, T, H) * 0.5
        g = -torch.rand(1, T, H) * 0.04
        cu = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int64)
        chunks = torch.tensor([(n, i) for n, length in enumerate(lengths)
                               for i in range((length + 63) // 64)], dtype=torch.int32)
        offsets = torch.tensor([0, *np.cumsum([(n + 63) // 64 for n in lengths])], dtype=torch.int32)
        slots = torch.arange(len(lengths) - 1, -1, -1, dtype=torch.int32)
        state = torch.randn(len(lengths) + 1, H, V, K) * 0.02
        state[-1].fill_(float("nan"))  # Unselected pool slot must never be read/written.
        before = state.clone()
        gc = torch.empty_like(g)
        cumsum.chunk_local_cumsum_scalar_kernel[(len(chunks), H)](
            g, gc, None, cu, chunks, T, 1, H, BT, False, False, True, False)
        A = torch.zeros(1, T, H, BT, dtype=q.dtype)
        bare(chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel)[(len(chunks), H)](
            k, gc, beta, A, cu, chunks, T, H, Hg, K, BT, 16, 32, True, True)
        assert torch.isfinite(A).all(), "KKT/solve padding"
        w, u = torch.empty(1, T, H, K, dtype=k.dtype), torch.empty_like(v)
        wy_fast.recompute_w_u_fwd_kernel[(len(chunks), H)](
            k, v, beta, w, u, A, gc, cu, chunks, T, H, Hg, K, V, BT, 64, 64, True)
        assert torch.isfinite(w).all() and torch.isfinite(u).all(), "WY padding"
        h = torch.empty(1, len(chunks), H, V, K, dtype=k.dtype)
        vn = torch.empty_like(v)
        bare(chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)[((V + 31) // 32, len(lengths) * H)](
            k, u, w, vn, gc, None, h, state, slots, cu, offsets,
            T, H, Hg, K, V, BT, 32, True, False, True, True, True, True, 0)
        assert torch.isfinite(h).all() and torch.isfinite(vn).all(), "state padding"
        out = torch.empty_like(v)
        chunk_o.chunk_fwd_kernel_o[((V + 63) // 64, len(chunks), H)](
            q, k, vn, h, gc, out, cu, chunks, K ** -0.5,
            T, H, Hg, K, V, BT, 128, 64, True, True)
        expected = torch.empty_like(out, dtype=torch.float64)
        for n, length in enumerate(lengths):
            ref = before[slots[n]].double().clone()
            for t in range(int(cu[n]), int(cu[n]) + length):
                kt = k[0, t].double().repeat_interleave(H // Hg, dim=0)
                qt = q[0, t].double().repeat_interleave(H // Hg, dim=0)
                ref *= g[0, t].double().exp()[:, None, None]
                delta = (v[0, t].double() - (ref * kt[:, None, :]).sum(-1)) * beta[0, t, :, None].double()
                ref += delta[:, :, None] * kt[:, None, :]
                expected[0, t] = (ref * qt[:, None, :]).sum(-1) * K ** -0.5
            torch.testing.assert_close(state[slots[n]].double(), ref, rtol=0.008, atol=0.003)
        torch.testing.assert_close(out.double(), expected, rtol=0.008, atol=0.003)
        assert torch.isnan(state[-1]).all(), "unselected state was changed"
    assert poisoned[0] == 0, f"{poisoned[0]} undefined float elements consumed"

    # Inspect addresses generated by the real state kernel without allocating
    # multi-GiB pools or dereferencing those synthetic addresses.
    loads, stores = set(), set()

    def inspect_load(self, ptrs, mask, other, *args):
        dtype = interpreter._get_np_dtype(ptrs.get_element_ty())
        if np.issubdtype(dtype, np.floating):
            loads.update(int(p) for p in ptrs.data[mask.data].flat)
            return interpreter.TensorHandle(np.zeros(ptrs.data.shape, dtype=dtype), ptrs.get_element_ty())
        return original(self, ptrs, mask, other, *args)

    def inspect_store(self, ptrs, value, mask, *args):
        stores.update(int(p) for p in ptrs.data[mask.data].flat)

    interpreter.InterpreterBuilder.create_masked_load = inspect_load
    interpreter.InterpreterBuilder.create_masked_store = inspect_store
    H, Hg, K, V, bos, slot, boh = 32, 16, 128, 128, 2 ** 20, 4096, 4096
    k = torch.zeros(1, 1, Hg, K, dtype=torch.float16)
    u = torch.zeros(1, 1, H, V, dtype=torch.float16)
    w = torch.zeros(1, 1, H, K, dtype=torch.float16)
    vn, h = torch.empty_like(u), torch.empty(1, 1, H, V, K, dtype=torch.float16)
    state = torch.zeros(1, H, V, K)
    bare(chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)[(1, 1)](
        k, u, w, vn, None, None, h, state, torch.tensor([slot], dtype=torch.int32),
        torch.tensor([bos, bos + 1], dtype=torch.int64), torch.tensor([boh], dtype=torch.int32),
        1, H, Hg, K, V, 64, 32, False, False, True, True, True, True, 0)
    assert state.data_ptr() + slot * H * V * K * 4 in loads
    assert k.data_ptr() + bos * Hg * K * 2 in loads
    assert vn.data_ptr() + bos * H * V * 2 in stores
    assert h.data_ptr() + boh * H * V * K * 2 in stores

    # Decode and optional intermediate caches use the same state[V,K] layout.
    # Inspect their real pointer arithmetic above 2^31 without dereferencing it.
    from freetoken.kernel.fla import fused_sigmoid_gating_recurrent as recurrent

    loads.clear()
    stores.clear()
    q = torch.zeros_like(k)
    v = torch.zeros(1, 3, H, V, dtype=torch.float16)
    a, b = torch.zeros(1, 3, H), torch.zeros(1, 3, H)
    out = torch.empty_like(v)
    cached = torch.empty(1, 1, H, V, K)
    recurrent.fused_sigmoid_gating_delta_rule_update_kernel[(1, 1, 1)](
        A_log=torch.zeros(H), a=a, dt_bias=torch.zeros(H), softplus_beta=1.0,
        softplus_threshold=20.0, q=q, k=k, v=v, b=b, o=out,
        h0_source=state, h0_indices=torch.tensor([slot], dtype=torch.int32),
        cu_seqlens=torch.tensor([bos, bos + 3], dtype=torch.int64),
        intermediate_states_buffer=cached, intermediate_state_indices=torch.tensor([slot], dtype=torch.int32),
        cache_steps=65, retrieve_parent_token_ptr=torch.tensor([[0, 0, 1]], dtype=torch.int32),
        stride_retrieve_parent_token_seq=3, stride_retrieve_parent_token_token=1,
        scale=K ** -0.5, T=3, stride_a=H, stride_q=Hg*K, stride_k=Hg*K,
        stride_v=H*V, stride_b=H, NP2_T=4, B=1, H=Hg, HV=H, K=K, V=V, BK=K, BV=32,
        USE_INITIAL_STATE=True, USE_QK_L2NORM_IN_KERNEL=False, IS_VARLEN=True, IS_KDA=False,
        CACHE_INTERMEDIATE_STATES=True, HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=True,
    )
    assert state.data_ptr() + slot * H * V * K * 4 in loads, "recurrent state read offset overflow"
    assert state.data_ptr() + slot * H * V * K * 4 in stores, "recurrent state write offset overflow"
    cache_base = cached.data_ptr() + slot * 65 * H * V * K * 4
    assert cache_base in loads, "recurrent parent cache read offset overflow"
    assert cache_base + H * V * K * 4 in loads, "recurrent parent-step read offset overflow"
    assert cache_base + 2 * H * V * K * 4 in stores, "recurrent step cache write offset overflow"
    print("padding/reference PASS (CPU interpreter; no GPU/model)")


if __name__ == "__main__":
    _probe()
