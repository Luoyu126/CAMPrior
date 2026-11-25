"""Stage 2A of docs/20261125/plan.md: the ported single-graph DASBS parts (utils/dasbs.py, scripts/sb/dasbs_single.py).
The first group is the DASBS worktree's own tests (tests_sb/test_core.py, test_dasbs.py) on the port; the rest
check the port against utils/dasbs_exact.py (the M2 tools) and the additions (corrector time 0, quadratic grid)."""
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from utils import dasbs as D            # noqa: E402
from utils import dasbs_exact as de     # noqa: E402

spec = importlib.util.spec_from_file_location('dasbs_single', os.path.join(HERE, '..', 'scripts', 'sb', 'dasbs_single.py'))
single = importlib.util.module_from_spec(spec)
spec.loader.exec_module(single)

torch.set_num_threads(1)
P3 = D.MISGraph(torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
REF = D.UniformFlipReference()


# ------------------------------------------------------------------ worktree tests on the port

def test_energy_flip_and_optimum():
    x = D.states(3)
    delta = P3.flip_delta(x)
    for i in range(3):
        flipped = x.clone()
        flipped[:, i] ^= 1
        torch.testing.assert_close(delta[:, i], P3.energy(flipped) - P3.energy(x))
    assert x[P3.energy(x).argmin()].tolist() == [1, 0, 1]


def test_reference_against_generator_and_bridge():
    r = D.UniformFlipReference('constant', 1.7)
    x = D.states(2)
    Q = np.zeros((4, 4))
    for j in range(4):
        Q[j, j ^ 1] = Q[j, j ^ 2] = 1.7 / 2
        Q[j, j] = -1.7
    np.testing.assert_allclose(r.log_kernel(x[:, None], x[None, :]).exp().numpy(), expm(Q), atol=1e-12)
    for a in x:
        for b in x:
            p = r.bridge_prob_one(a, b, .37)
            direct = (r.log_kernel(a, x, 0, .37) + r.log_kernel(x, b, .37, 1)).exp()
            direct /= direct.sum()
            torch.testing.assert_close(p, (direct[:, None] * x).sum(0))
            torch.testing.assert_close(r.bridge_prob_one(a, b, 0), a.double())
            torch.testing.assert_close(r.bridge_prob_one(a, b, 1), b.double())


def test_corrector_reference_ratios():
    x = D.states(3)
    for start in x:
        for y in x:
            ratio = REF.endpoint_flip_log_ratio(start, y)
            for i in range(3):
                flip = y.clone()
                flip[i] ^= 1
                torch.testing.assert_close(ratio[i], REF.log_kernel(start, flip) - REF.log_kernel(start, y))


def test_exact_sb_marginals_and_matching():
    exact = D.solve_bridge(P3, REF, temperature=.8)
    np.testing.assert_allclose(exact.joint.sum(0), exact.target, atol=1e-10)
    np.testing.assert_allclose(exact.joint.sum(1), exact.source, atol=1e-10)
    x = exact.states
    ids = np.arange(len(x))[:, None] ^ (1 << np.arange(3))
    t = .41
    K = REF.log_kernel(x[:, None], x[None, :], t, 1).exp().numpy()
    b = np.exp(exact.log_b)
    conditional = K * b[None, :] / (K @ b)[:, None]
    np.testing.assert_allclose(conditional @ (b[ids] / b[:, None]), np.exp(exact.controller_log_ratio(REF, t)),
                               atol=1e-10)
    K01 = REF.log_kernel(x[:, None], x[None, :]).exp().numpy()
    for i in range(3):
        expectation = (exact.joint * (K01[:, ids[:, i]] / K01)).sum(0) / exact.target
        np.testing.assert_allclose(expectation, np.exp(exact.corrector[ids[:, i]] - exact.corrector), atol=1e-9)


def test_matching_gradient_is_ratio_not_log_mse():
    pred = torch.tensor([.3, -.2], requires_grad=True)
    label = torch.tensor([1., -2.], requires_grad=True)
    D.ratio_bregman(pred, label).backward()
    torch.testing.assert_close(pred.grad, (pred.detach().exp() - label.detach().exp()) / 2)
    assert label.grad is None


def test_rollout_source_and_feasibility():
    class ReferenceModel(torch.nn.Module):
        def forward(self, x, t, graph):
            return torch.zeros_like(x, dtype=torch.float32)
    gen = torch.Generator().manual_seed(17)
    r = D.UniformFlipReference('constant', 2)
    result = D.sample_paths(ReferenceModel(), P3, r, 20000, steps=10, source_probability=0, generator=gen)
    assert (result.x0 == 0).all()
    assert abs(float(result.x1.float().mean()) - float(r.flip_probability(0., 1.))) < .01
    metrics = D.evaluate_samples(P3, result.x1)
    assert metrics['decoded_feasible'] and metrics['decoded_size_best'] <= 2


def test_paper_architecture_and_non_square_nodes():
    torch.manual_seed(1)
    ctrl, corr = single.DASBSModel(7), single.DASBSModel(7, timed=False)
    assert len(ctrl.blocks) == 6 and ctrl.embedding.shape == (2, 32) and ctrl.blocks[0].attn.heads == 4
    assert not hasattr(corr, 'time_embed')
    x = torch.randint(2, (3, 7))
    y = ctrl(x, torch.rand(3))
    assert y.shape == (3, 7)
    y.sum().backward()
    assert float(ctrl.blocks[0].attn.freqs.grad.abs().sum()) > 0
    assert float(ctrl.time_embed[0].weight.grad.abs().sum()) > 0
    torch.testing.assert_close(corr(x, None), corr(x, torch.ones(3)))


@pytest.mark.parametrize('time', ['uniform', 'zero'])
def test_corrector_freeze_and_both_losses(time):
    torch.manual_seed(1)
    graph = D.MISGraph.erdos_renyi(3, .5, 7)
    ctrl = single.DASBSModel(3, width=16, layers=2, heads=2)
    corr = single.DASBSModel(3, width=16, layers=2, heads=2, timed=False)
    rollout = D.sample_paths(ctrl, graph, REF, 8, steps=4, keep_path=False)
    loss = D.controller_loss(ctrl, corr, rollout, graph, REF, 1., 2)
    loss.backward()
    assert all(p.grad is None for p in corr.parameters()) and torch.isfinite(loss)
    ctrl.zero_grad(set_to_none=True)
    loss2 = D.corrector_loss(corr, rollout, graph, REF, 2, time=time)
    loss2.backward()
    assert all(p.grad is None for p in ctrl.parameters()) and torch.isfinite(loss2)
    assert any(p.grad is not None for p in corr.parameters())


# ------------------------------------------------------------------ port vs the M2 tools

def to_product_order(n):
    """index permutation: LSB-first state k of utils/dasbs.py -> its row in dasbs_exact.states (MSB first)."""
    X = D.states(n).numpy()
    return (X * (2 ** np.arange(n - 1, -1, -1))).sum(1)


@pytest.mark.parametrize('sched', [('log_linear', 1.0, 0.5), ('constant', 1.5, 1.0)])
def test_exact_bridge_equals_m2_exact_sb(sched):
    name, gam, alpha = sched
    r = D.UniformFlipReference(name, gam, alpha)
    r2 = de.Reference('constant' if name == 'constant' else 'loglinear', gam, alpha)
    rng = np.random.default_rng(0)
    for n, tau in [(5, 1.0), (6, 0.5), (7, 0.7)]:
        A = np.triu(rng.random((n, n)) < .35, 1); A = (A | A.T).astype(np.int64)
        g = D.MISGraph(torch.from_numpy(A))
        ex = D.solve_bridge(g, r, tau)
        X2 = de.states(n)
        nu = de.boltzmann(X2, A, 2.0, tau)
        sb = de.ExactSB(np.full(2 ** n, 2.0 ** -n), nu, r2, n)
        perm = to_product_order(n)
        assert np.abs(ex.target - nu[perm]).max() < 1e-12
        assert np.abs(ex.joint - sb.joint[np.ix_(perm, perm)]).max() < 1e-10
        for t in (0.0, 0.3, 0.8):
            Phi2 = sb.controller(t)                 # (S, n) in product order, bit i = column i
            np.testing.assert_allclose(np.exp(ex.controller_log_ratio(r, t)), Phi2[perm], rtol=1e-8)
        ids = np.arange(2 ** n)[:, None] ^ (1 << np.arange(n))
        np.testing.assert_allclose(np.exp(ex.corrector[ids] - ex.corrector[:, None]), sb.corrector()[perm], rtol=1e-8)


def test_flip_delta_matches_m2_energy():
    rng = np.random.default_rng(1)
    n = 7
    A = np.triu(rng.random((n, n)) < .4, 1); A = (A | A.T).astype(np.int64)
    g = D.MISGraph(torch.from_numpy(A))
    X = D.states(n)
    np.testing.assert_allclose(g.energy(X.double()).numpy(), de.mis_energy(X.numpy(), A, 2.0), atol=1e-12)
    np.testing.assert_allclose(g.flip_delta(X.double()).numpy(), de.mis_flip_delta(X.numpy(), A, 2.0), atol=1e-12)


@pytest.mark.parametrize('time', ['uniform', 'zero'])
def test_corrector_targets_average_to_exact_corrector(time):
    """Population check of corr-DM: with (X0, X1) from the exact SB, E[exp(target) | X1 = y] = Phihat(y) for both
    time choices (math_route2.md, prop. 5.1). The identity holds at every t, so the average over a grid of t (X_t
    summed out with the closed-form bridge) is exact too."""
    n = 5
    rng = np.random.default_rng(2)
    A = np.triu(rng.random((n, n)) < .4, 1); A = (A | A.T).astype(np.int64)
    g = D.MISGraph(torch.from_numpy(A))
    ex = D.solve_bridge(g, REF, 0.8)
    X = ex.states
    S = len(X)
    ids = np.arange(S)[:, None] ^ (1 << np.arange(n))
    want = np.exp(ex.corrector[ids] - ex.corrector[:, None])            # (S, n)
    J = torch.from_numpy(ex.joint)
    x0 = X[:, None, :].expand(S, S, n).reshape(-1, n)
    x1 = X[None, :, :].expand(S, S, n).reshape(-1, n)
    w = J.reshape(-1)
    ts = [0.0] if time == 'zero' else list((np.arange(400) + 0.5) / 400 * 0.99)
    got = torch.zeros(S, n, dtype=torch.float64)
    for t in ts:
        tt = torch.full((len(x0),), t, dtype=torch.float64)
        p1 = REF.bridge_prob_one(x0, x1, tt)                             # P(X_t,i = 1 | x0, x1); bits independent
        # E[ratio_i] = sum over X_t,i in {0, 1}; ratio_i depends on bit i of X_t only
        r_eq = REF.endpoint_flip_log_ratio(x1, x1, tt, torch.ones_like(tt)).exp()        # X_t,i == y_i
        r_ne = (-REF.endpoint_flip_log_ratio(x1, x1, tt, torch.ones_like(tt))).exp()     # X_t,i != y_i
        p_eq = torch.where(x1 == 1, p1, 1 - p1)
        val = p_eq * r_eq + (1 - p_eq) * r_ne
        got += torch.zeros(S, n, dtype=torch.float64).index_add_(0, D.state_indices(x1), w[:, None] * val) / len(ts)
    got = got.numpy() / ex.target[:, None]
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_quadratic_grid():
    t = D.time_grid(50, 'quadratic')
    assert t[0] == 0 and t[-1] == 1 and (t.diff() > 0).all()
    assert t.diff()[-1] < t.diff()[0] / 40
    torch.testing.assert_close(D.time_grid(7), torch.linspace(0, 1, 8, dtype=torch.float64))


def test_load_graph_specs():
    assert single.load_graph('cycle8').adjacency.sum() == 16
    assert single.load_graph('path6').adjacency.sum() == 10
    g = single.load_graph('er8_p0.3_s7')
    torch.testing.assert_close(g.adjacency, D.MISGraph.erdos_renyi(8, .3, 7).adjacency)
