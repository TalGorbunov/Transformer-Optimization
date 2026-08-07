#!/usr/bin/env python3
"""CPU tests for gnnformer/gating.py (seconds, no model load).

Pins, in order of how badly each one bites if it breaks:
  1. bit-for-bit identity at init for every variant  (a gate that is not exactly 1.0 at
     init makes every gated arm incomparable to the LoRA control);
  2. gate shapes under GQA 28q/4kv, head-specific vs head-shared;
  3. the gradient reaches W and is NOT tiny at b0=+2 — and IS tiny at b0=+6, which is
     the whole reason the campaign brief forbids b0=+6;
  4. state-dict round-trip, and remove() restoring the original forward exactly.

Run: python tests/test_gating.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gnnformer.gating import (  # noqa: E402
    GATE_VARIANTS,
    attach_gate,
    gate_out_features,
    gate_scores,
)

# Qwen2.5-VL-7B geometry, shrunk 8x in hidden so the test stays in milliseconds but keeps
# the 7:1 GQA ratio that makes g2_literal head-SHARED (the reason P4 exists).
HID, NH, NKV, HD = 448, 28, 4, 16
assert NH * HD == HID and NH % NKV == 0


class FakeAttn(nn.Module):
    """Stand-in with the same module names the real hooks bind to."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HID, NH * HD, bias=False)
        self.k_proj = nn.Linear(HID, NKV * HD, bias=False)
        self.v_proj = nn.Linear(HID, NKV * HD, bias=False)
        self.o_proj = nn.Linear(NH * HD, HID, bias=False)

    def forward(self, hidden_states: torch.Tensor, **_kw: object) -> torch.Tensor:
        b, t, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, t, NH, HD).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, NKV, HD).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, NKV, HD).transpose(1, 2)
        rep = NH // NKV
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, NH * HD))


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(HID)
        self.self_attn = FakeAttn()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.self_attn(hidden_states=self.input_layernorm(h))


def build(n: int = 3):
    torch.manual_seed(0)
    return nn.ModuleList([FakeLayer() for _ in range(n)])


def run(layers, x: torch.Tensor) -> torch.Tensor:
    h = x
    for ly in layers:
        h = ly(h)
    return h


DIMS = dict(hidden=HID, n_heads=NH, n_kv=NKV, head_dim=HD)


def test_identity_at_init() -> None:
    layers = build()
    x = torch.randn(2, 7, HID)
    ref = run(layers, x)
    for var in GATE_VARIANTS:
        g = attach_gate(layers, [1, 2], var, **DIMS)
        got = run(layers, x)
        assert torch.equal(ref, got), f"{var}: not bit-for-bit identity at init"
        ms = g.mean_scores()
        assert set(ms) == {1, 2}, f"{var}: stats missing layers, got {sorted(ms)}"
        for li, v in ms.items():
            assert abs(v - 1.0) < 1e-6, f"{var} L{li}: init gate score {v} != 1.0"
        g.remove()
        assert torch.equal(ref, run(layers, x)), f"{var}: remove() did not restore forward"
    print("ok  identity at init (bit-for-bit) + remove() restores, all variants")


def test_shapes() -> None:
    expect = {"g1_headwise": NH, "g1_headshared": 1,
              "g1_elementwise": NH * HD, "g2_literal": NKV * HD}
    layers = build()
    for var, n_out in expect.items():
        assert gate_out_features(var, **DIMS) == n_out, var
        g = attach_gate(layers, [0], var, **DIMS)
        W = g.params[0]
        assert tuple(W.shape) == (n_out, HID), f"{var}: W {tuple(W.shape)} != {(n_out, HID)}"
        assert g.num_parameters() == n_out * HID, var
        g.remove()
    # g2_literal gates the KV width, so under 7:1 GQA one gate score is forcibly shared by
    # 7 query heads — head-SHARED, the paper's weak condition. This is why P4 exists.
    assert expect["g2_literal"] // HD == NKV < NH
    assert expect["g1_headwise"] == NH  # head-specific
    print(f"ok  shapes under GQA {NH}q/{NKV}kv (g2_literal shares 1 gate across "
          f"{NH // NKV} query heads)")


def test_gradient_reaches_w() -> None:
    x = torch.randn(2, 7, HID)
    norms = {}
    for b0 in (2.0, 6.0):
        layers = build()
        for p in layers.parameters():
            p.requires_grad_(False)
        g = attach_gate(layers, [1], "g1_headwise", b0=b0, **DIMS)
        run(layers, x).pow(2).mean().backward()
        gr = g.params[1].grad
        assert gr is not None, f"b0={b0}: no gradient on W"
        norms[b0] = float(gr.norm())
        g.remove()
    assert norms[2.0] > 0, "b0=2: zero gradient on W"
    ratio = norms[2.0] / max(norms[6.0], 1e-30)
    # sigma'(2)/sigma(2) = 0.1192 vs sigma'(6)/sigma(6) = 0.00248 -> ~48x
    assert ratio > 10, f"b0=2 should give a far larger gate gradient than b0=6, got {ratio:.1f}x"
    print(f"ok  gradient reaches W; b0=2 is {ratio:.0f}x b0=6 (the reason b0=+6 is banned)")


def test_state_roundtrip() -> None:
    layers = build()
    x = torch.randn(2, 7, HID)
    for var in GATE_VARIANTS:
        g = attach_gate(layers, [1, 2], var, b0=2.0, **DIMS)
        with torch.no_grad():
            for li in g.layer_ids:
                g.params[li].normal_(0, 0.02)
        trained = run(layers, x)
        st = g.state()
        g.remove()
        g2 = attach_gate(layers, [1, 2], var, state=st, **DIMS)
        assert torch.equal(trained, run(layers, x)), f"{var}: state round-trip changed the forward"
        assert g2.b0 == 2.0 and g2.variant == var
        g2.remove()
    print("ok  state() -> attach_gate(state=...) round-trip is exact, all variants")


def test_gate_is_attenuating_only() -> None:
    """sigmoid(.)/sigmoid(b0) is bounded by 1/sigmoid(b0) — with b0=2 the gate can amplify
    by at most 1.135x and attenuates toward 0. Pins the campaign's core claim that gating
    cannot add bandwidth to the aggregation channel."""
    x = torch.randn(64, HID) * 5
    W = torch.randn(8, HID) * 0.1
    b = torch.full((8,), 2.0)
    g = gate_scores(x, W, b)
    assert float(g.min()) >= 0.0
    assert float(g.max()) <= 1.0 / float(torch.sigmoid(torch.tensor(2.0))) + 1e-6
    print(f"ok  gate range [{float(g.min()):.3f}, {float(g.max()):.3f}] "
          f"<= 1/sigma(2) = {1/float(torch.sigmoid(torch.tensor(2.0))):.3f} (attenuation-dominant)")


def main() -> int:
    test_identity_at_init()
    test_shapes()
    test_gradient_reaches_w()
    test_state_roundtrip()
    test_gate_is_attenuating_only()
    print("\nALL GATING TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
