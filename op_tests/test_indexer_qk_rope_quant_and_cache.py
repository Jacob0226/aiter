# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf harness for indexer_qk_rope_quant_and_cache.

The kernel fuses the DSA indexer q/k preparation: k LayerNorm + RoPE + fp8 quant
+ paged store, q RoPE + fp8 quant, and the head-gate scale. It is used by SGLang
at both decode (num_tokens = batch) and prefill (num_tokens = chunk) widths, and
those two are very different regimes for the launch geometry.
"""

import argparse

import torch

import aiter
from aiter import dtypes
from aiter.ops.cache import indexer_qk_rope_quant_and_cache

torch.set_default_device("cuda")

HEAD_DIM = 128
ROPE_DIM = 64
TILE = 16  # MFMA 16x16 tile used by the preshuffle cache layout
CACHE_STRIDE = HEAD_DIM + HEAD_DIM // HEAD_DIM * 4  # quant_block_size == head_dim


def _fp8_dtype():
    return dtypes.fp8


def _fp8_max():
    return torch.finfo(_fp8_dtype()).max


def _rope_interleaved(x, cos, sin):
    """Non-neox RoPE over the leading ROPE_DIM columns, pairing (2i, 2i+1)."""
    rot = x[..., :ROPE_DIM].float()
    even = rot[..., 0::2]
    odd = rot[..., 1::2]
    c = cos.float()
    s = sin.float()
    out = torch.empty_like(rot)
    out[..., 0::2] = even * c - odd * s
    out[..., 1::2] = odd * c + even * s
    y = x.float().clone()
    # The kernel materializes the rotated half back in the input dtype before
    # quantizing, so the reference has to round the same way.
    y[..., :ROPE_DIM] = out.to(x.dtype).float()
    return y


def _quantize(vals, amax_floor, use_ue8m0, reciprocal_fp8_max):
    # Follows the kernel bit for bit: q divides by fp8_max through a reciprocal
    # multiply while k divides directly, and both apply the scale as a
    # reciprocal multiply. Doing either as a plain divide here shifts ~0.5% of
    # the outputs by one fp8 ulp, which near the top of e4m3 range is 32.
    amax = vals.abs().amax(dim=-1, keepdim=True)
    floored = torch.clamp(amax, min=amax_floor)
    scale = floored * (1.0 / _fp8_max()) if reciprocal_fp8_max else floored / _fp8_max()
    if use_ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (vals * (1.0 / scale)).to(_fp8_dtype())
    return q, scale


def reference(
    q, weights, k, slot_mapping, norm_weight, norm_bias, positions, cos, sin,
    epsilon, weights_scale, use_ue8m0, cache_block_size, num_blocks, preshuffle,
):
    num_tokens, n_heads, _ = q.shape
    cos_t = cos[positions]
    sin_t = sin[positions]

    q_rot = _rope_interleaved(q, cos_t.unsqueeze(1), sin_t.unsqueeze(1))
    q_fp8, q_scale = _quantize(q_rot, 1e-10, use_ue8m0, True)
    weights_out = weights.float() * q_scale.squeeze(-1) * weights_scale

    kf = k.float()
    mean = kf.mean(dim=-1, keepdim=True)
    centered = kf - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    normed = (centered * torch.rsqrt(var + epsilon)) * norm_weight + norm_bias
    normed = normed.to(k.dtype)
    k_rot = _rope_interleaved(normed, cos_t, sin_t)
    k_fp8, k_scale = _quantize(k_rot, 1e-4, use_ue8m0, False)

    cache = torch.zeros(
        (num_blocks, cache_block_size, CACHE_STRIDE),
        dtype=torch.uint8,
        device=q.device,
    )
    flat = cache.view(num_blocks, cache_block_size * CACHE_STRIDE)
    k_bytes = k_fp8.view(torch.uint8)
    block_idx = (slot_mapping // cache_block_size).tolist()
    block_off = (slot_mapping % cache_block_size).tolist()
    for t in range(num_tokens):
        b, o = block_idx[t], block_off[t]
        if preshuffle:
            tile_id, in_tile = o // TILE, o % TILE
            for col_tile in range(HEAD_DIM // TILE):
                base = (
                    tile_id * TILE * HEAD_DIM
                    + col_tile * TILE * TILE
                    + in_tile * TILE
                )
                flat[b, base : base + TILE] = k_bytes[
                    t, col_tile * TILE : (col_tile + 1) * TILE
                ]
        else:
            flat[b, o * HEAD_DIM : (o + 1) * HEAD_DIM] = k_bytes[t]
    scale_region = flat[:, cache_block_size * HEAD_DIM :].view(torch.float32)
    for t in range(num_tokens):
        scale_region[block_idx[t], block_off[t]] = k_scale[t, 0]
    return q_fp8, weights_out, cache.view(_fp8_dtype())


def make_inputs(num_tokens, n_heads, cache_block_size, preshuffle, seed=0):
    torch.manual_seed(seed)
    dt = dtypes.bf16
    max_pos = 65536
    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dt) * 0.5
    weights = torch.randn(num_tokens, n_heads, dtype=dt)
    k = torch.randn(num_tokens, HEAD_DIM, dtype=dt) * 0.5
    norm_weight = torch.randn(HEAD_DIM, dtype=torch.float32) * 0.1 + 1.0
    norm_bias = torch.randn(HEAD_DIM, dtype=torch.float32) * 0.1
    positions = torch.randint(0, 8192, (num_tokens,), dtype=torch.int64)
    cos = torch.randn(max_pos, ROPE_DIM // 2, dtype=dt)
    sin = torch.randn(max_pos, ROPE_DIM // 2, dtype=dt)

    # Distinct slots, shuffled so the paged store is not artificially coalesced.
    num_blocks = (num_tokens + cache_block_size - 1) // cache_block_size + 4
    slots = torch.randperm(num_blocks * cache_block_size)[:num_tokens]
    slot_mapping = slots.to(torch.int64)
    return dict(
        q=q, weights=weights, k=k, norm_weight=norm_weight, norm_bias=norm_bias,
        positions=positions, cos=cos, sin=sin, slot_mapping=slot_mapping,
        num_blocks=num_blocks, cache_block_size=cache_block_size,
        preshuffle=preshuffle,
    )


def run_kernel(inp, epsilon, quant_block_size, scale_fmt, weights_scale, out=None):
    num_tokens, n_heads, _ = inp["q"].shape
    if out is None:
        q_out = torch.empty(
            num_tokens, n_heads, HEAD_DIM, dtype=_fp8_dtype(), device="cuda"
        )
        weights_out = torch.empty(num_tokens, n_heads, dtype=torch.float32)
        cache = torch.zeros(
            (inp["num_blocks"], inp["cache_block_size"], CACHE_STRIDE),
            dtype=torch.uint8,
        ).view(_fp8_dtype())
        out = (q_out, weights_out, cache)
    q_out, weights_out, cache = out
    indexer_qk_rope_quant_and_cache(
        inp["q"], q_out, inp["weights"], weights_out, inp["k"], cache,
        inp["slot_mapping"], inp["norm_weight"], inp["norm_bias"], inp["positions"],
        inp["cos"], inp["sin"], epsilon, quant_block_size, scale_fmt, weights_scale,
        preshuffle=inp["preshuffle"], is_neox=False,
    )
    return out


def _fp8_ulp_diff(a, b):
    """Distance in representable fp8 steps. e4m3 is monotonic in its low 7 bits,
    so sign-magnitude byte arithmetic gives the ulp gap directly."""
    ai = a.view(torch.uint8).to(torch.int32)
    bi = b.view(torch.uint8).to(torch.int32)
    am, bm = ai & 0x7F, bi & 0x7F
    sign_flip = ((ai >> 7) != (bi >> 7)) & (am != 0) & (bm != 0)
    return torch.where(sign_flip, torch.full_like(am, 99), (am - bm).abs())


def cmp_tensors(label, got, ref, max_ulp, max_ulp_frac):
    """fp8 outputs are compared in ulps: the K path runs through the hardware
    rsqrtf, which no torch reference reproduces bit for bit, and a single bf16
    ulp before quantization becomes one fp8 ulp (32.0 near the top of e4m3)."""
    if got.dtype == _fp8_dtype():
        d = _fp8_ulp_diff(got, ref)
        off = (d > 0).sum().item()
        frac = off / d.numel()
        ok = d.max().item() <= max_ulp and frac <= max_ulp_frac
        print(
            f"    {label:<10} max={d.max().item():>3} ulp  "
            f"differing={off}/{d.numel()} ({frac * 100:.3f}%)  "
            f"{'OK' if ok else 'FAIL'}"
        )
        return ok
    d = (got.float() - ref.float()).abs()
    rel = d / ref.float().abs().clamp(min=1e-30)
    ok = rel.max().item() <= 1e-6
    print(
        f"    {label:<10} max|rel|={rel.max().item():.2e}  {'OK' if ok else 'FAIL'}"
    )
    return ok


def check(num_tokens, n_heads, cache_block_size, preshuffle, scale_fmt):
    inp = make_inputs(num_tokens, n_heads, cache_block_size, preshuffle)
    eps, wscale = 1e-6, 0.088388
    got_q, got_w, got_cache = run_kernel(inp, eps, HEAD_DIM, scale_fmt, wscale)
    ref_q, ref_w, ref_cache = reference(
        inp["q"], inp["weights"], inp["k"], inp["slot_mapping"], inp["norm_weight"],
        inp["norm_bias"], inp["positions"], inp["cos"], inp["sin"], eps, wscale,
        scale_fmt == "ue8m0", cache_block_size, inp["num_blocks"], preshuffle,
    )
    torch.cuda.synchronize()
    print(
        f"  T={num_tokens} H={n_heads} page={cache_block_size} "
        f"preshuffle={preshuffle} fmt={scale_fmt or 'none'}"
    )
    ok = cmp_tensors("q_out", got_q, ref_q, 0, 0.0)
    ok &= cmp_tensors("weights", got_w, ref_w, 0, 0.0)
    ok &= cmp_tensors("cache", got_cache, ref_cache, 1, 0.02)
    return ok


GOLDEN_CASES = [
    (1, 64, True), (4, 64, True), (63, 64, True), (128, 64, True),
    (1024, 64, True), (4096, 64, True), (7679, 64, True),
    (1, 1, False), (128, 1, False), (4096, 1, False),
]


def golden(path, n_heads, scale_fmt, save):
    """A/B one kernel build against another. The torch reference cannot pin the
    K path to the bit, but two kernel builds that do the same math should agree
    exactly, so this is the gate that actually catches a bad rewrite."""
    store = {} if save else torch.load(path)
    all_ok = True
    for T, page, pre in GOLDEN_CASES:
        inp = make_inputs(T, n_heads, page, pre)
        q, w, cache = run_kernel(inp, 1e-6, HEAD_DIM, scale_fmt, 0.088388)
        torch.cuda.synchronize()
        key = f"{T}_{page}_{pre}"
        if save:
            store[key] = (q.cpu(), w.cpu(), cache.cpu())
            continue
        rq, rw, rc = (t.cuda() for t in store[key])
        print(f"  T={T} page={page} preshuffle={pre}")
        ok = cmp_tensors("q_out", q, rq, 0, 0.0)
        ok &= cmp_tensors("weights", w, rw, 0, 0.0)
        ok &= cmp_tensors("cache", cache, rc, 0, 0.0)
        all_ok &= ok
    if save:
        torch.save(store, path)
        print(f"  saved {len(store)} cases to {path}")
        return True
    return all_ok


def _set_bytes(num_tokens, n_heads):
    return (
        num_tokens * n_heads * HEAD_DIM * 3  # q read bf16 + q_out write fp8
        + num_tokens * n_heads * 6  # weights bf16 in, fp32 out
        + num_tokens * HEAD_DIM * 3  # k read bf16 + cache write fp8
        + num_tokens * 4
    )


def bench(
    num_tokens, n_heads, cache_block_size, preshuffle, scale_fmt,
    per_graph=None, reps=20, min_footprint=768 << 20,
):
    """Times the kernel inside a CUDA graph over a rotating set of buffers.

    Two things would otherwise measure the harness instead of the kernel.
    Launching from Python costs ~37 us per call, which is more than the kernel
    at every decode width, so the work is captured in a graph -- which is also
    what SGLang does at decode. And MI355X has a 256 MB last level cache that a
    single prefill-width input set fits inside, so the buffers are rotated until
    their combined footprint is past it and the reads come from HBM.
    """
    traffic = _set_bytes(num_tokens, n_heads)
    rotate = max(2, min(16, -(-min_footprint // max(traffic, 1))))
    if per_graph is None:
        per_graph = max(rotate, 20 - 20 % rotate)
    per_graph -= per_graph % rotate

    sets = [
        make_inputs(num_tokens, n_heads, cache_block_size, preshuffle, seed=i)
        for i in range(rotate)
    ]
    calls = [(s, 1e-6, HEAD_DIM, scale_fmt, 0.088388) for s in sets]
    outs = [run_kernel(*c) for c in calls]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(rotate):
            run_kernel(*calls[i], out=outs[i])
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(per_graph):
            run_kernel(*calls[i % rotate], out=outs[i % rotate])
    graph.replay()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000 / (reps * per_graph)

    print(
        f"  T={num_tokens:>6} H={n_heads} page={cache_block_size:>2} "
        f"rot={rotate:>2} {us:>9.2f} us  {traffic / us / 1e3:>8.1f} GB/s"
    )
    return us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--save-golden")
    p.add_argument("--cmp-golden")
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--scale-fmt", default="")
    args = p.parse_args()
    if args.save_golden or args.cmp_golden:
        print(f"device: {torch.cuda.get_device_name()}")
        path = args.save_golden or args.cmp_golden
        ok = golden(path, args.heads, args.scale_fmt, save=bool(args.save_golden))
        print("GOLDEN MATCH" if ok else "GOLDEN MISMATCH")
        return
    if not (args.check or args.bench):
        args.check = args.bench = True

    print(f"device: {torch.cuda.get_device_name()}  fp8: {_fp8_dtype()}")
    if args.check:
        print("== correctness ==")
        ok = True
        for T in (1, 4, 63, 128, 1024, 4096):
            for page, pre in ((64, True), (1, False)):
                ok &= check(T, args.heads, page, pre, args.scale_fmt)
        print("ALL OK" if ok else "FAILURES PRESENT")
    if args.bench:
        widths = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7679, 16384)
        for page, pre in ((64, True), (1, False)):
            print(f"== perf (page={page}, preshuffle={pre}, cuda graph) ==")
            for T in widths:
                bench(T, args.heads, page, pre, args.scale_fmt)


if __name__ == "__main__":
    main()
