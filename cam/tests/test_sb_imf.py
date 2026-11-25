"""Stage 1D of docs/20261125/plan.md, acceptance 1: on small instances, exact (tabular) discrete-time IMF converges to
the Sinkhorn SB coupling, total variation < 1e-3 (utils/sb_imf.py; CSBM arXiv 2502.01416 §4.1 does the same check in
one dimension). Exact enumeration of all 2^n states, float64.

    /usr0/home/shengyuf/yunyi/envs/sb/bin/python -m pytest cam/tests/test_sb_imf.py
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import sb_imf as imf                           # noqa: E402
from utils import sb_coupling as sb                       # noqa: E402

TOL_TV = 1e-3


def instance(seed, n, M, omega, T, weighted=False):
    g = torch.Generator().manual_seed(seed)
    while True:
        Y = torch.unique((torch.rand(M, n, generator=g) < 0.4).to(torch.uint8), dim=0)
        if len(Y) >= 3:
            break
    pi1 = torch.softmax(Y.double().sum(1), 0) if weighted else torch.full((len(Y),), 1.0 / len(Y), dtype=torch.float64)
    a_T = torch.tanh(torch.tensor(omega / 2)).item()
    # a_k = a_T + (1 - a_T) * cos-like decay, a_0 = 1 (the shape of sb_coupling.sb_signal_schedule)
    s = torch.cos(torch.linspace(0, 1, T + 1, dtype=torch.float64) * torch.pi / 2) ** 2
    a = a_T + (1 - a_T) * s
    a[0] = 1.0
    return imf.TabularIMF(Y, pi1, a)


CASES = [(0, 6, 8, 0.5, 4, False), (1, 7, 10, 1.0, 6, False), (2, 6, 6, 2.0, 3, True), (3, 8, 12, 0.35, 5, False),
         (4, 5, 5, 3.0, 8, True), (5, 7, 9, 1.0, 2, False)]


@pytest.mark.parametrize('case', CASES)
def test_imf_converges_to_sinkhorn(case):
    P = instance(*case)
    star = P.sb_coupling()
    assert (star.sum(1) - P.pi0).abs().max() < 1e-12 and (star.sum(0) - P.pi1).abs().max() < 1e-12
    qs = P.run(2000)
    tvs = [imf.tv(q, star) for q in qs]
    kls = [imf.kl(q, star) for q in qs]
    first = next((l + 1 for l, v in enumerate(tvs) if v < TOL_TV), None)
    print(f'case {case}: TV after 1/10/60 projections {tvs[0]:.3g}/{tvs[9]:.3g}/{tvs[59]:.3g}, below {TOL_TV} after '
          f'{first} projections, final {tvs[-1]:.2e}')
    assert first is not None and tvs[-1] < TOL_TV, (case, tvs[:: 100], tvs[-1])
    # KL to the SB never increases along the alternation (DDSBM Thm 3.3 / CSBM Cor. 3.2), up to round-off
    assert all(b <= a + 1e-12 for a, b in zip(kls, kls[1:])), kls[:8]
    # gen keeps pi0 exactly, fwd keeps pi1 exactly; the other marginal is also kept (exact projections)
    for l, q in enumerate(qs[:6]):
        assert (q.sum(1) - P.pi0).abs().max() < 1e-12 and (q.sum(0) - P.pi1).abs().max() < 1e-12
    # starts closer than independent: the mean Hamming cost decreases to the SB's
    h_ind = imf.mean_hamming(P.independent(), P.X, P.Y)
    h_star = imf.mean_hamming(star, P.X, P.Y)
    assert h_star < h_ind and abs(imf.mean_hamming(qs[-1], P.X, P.Y) - h_star) < 1e-2


def test_sb_is_fixed_point():
    for case in CASES[:3]:
        P = instance(*case)
        star = P.sb_coupling()
        assert imf.tv(P.gen(star), star) < 1e-10 and imf.tv(P.fwd(star), star) < 1e-10


def test_one_direction_also_converges_when_exact():
    """With exact projections the gen-only iteration converges too (the alternation matters only for approximate,
    learned projections: CSBM [C] L427-437, DSBM §4)."""
    P = instance(*CASES[1])
    star = P.sb_coupling()
    assert imf.tv(P.run(60, order='gen')[-1], star) < TOL_TV


@pytest.mark.parametrize('case', CASES[:3])
def test_wrong_bridge_does_not_reach_sb(case):
    """Negative control: lifting the coupling with the forward marginals instead of the bridge (z ignored) keeps the
    coupling far from the SB."""
    P = instance(*case)
    star = P.sb_coupling()
    P.bridge = False
    q = P.run(40)[-1]
    assert imf.tv(q, star) > 0.02, imf.tv(q, star)


def test_wrong_cost_strength_is_detected():
    """Negative control: the Sinkhorn coupling for a different endpoint signal differs by more than the tolerance."""
    P = instance(*CASES[0])
    q = P.run(60)[-1]
    other = instance(CASES[0][0], CASES[0][1], CASES[0][2], CASES[0][3] * 1.5, CASES[0][4], CASES[0][5]).sb_coupling()
    assert imf.tv(q, other) > 10 * TOL_TV
