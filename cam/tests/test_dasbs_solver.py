"""Tests for solvers/dasbs.py (cross-graph DASBS, stage 2B of docs/20261125/plan.md). CPU only.

    source cam/scripts/env.sh
    $CAM_PYTHON cam/tests/test_dasbs_solver.py

The training targets of the solver are checked at the population level against the exact SB of utils/dasbs_exact.py
(the M2 tools): for every state x_t, the mean of exp(target) under the exact SB coupling and the reference bridge must
equal the exact controller (ctrl-AM) or corrector (corr-DM). Acceptance 1 of 2B is the memoryless case: with the
memoryless reference the corrector is identically 1 and the ctrl-AM target is the plain Boltzmann ratio, whose
conditional mean is the M2 controller (math_route2.md corollary 7.2).
(Own runner like test_block_labels.py; the test_* functions are also pytest-compatible.)
"""
import os
import sys
import time
import traceback
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('WANDB_MODE', 'disabled')
CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from envs.mis import Env  # noqa: E402
from solvers.dasbs import (DASBSNet, controller_target, corrector_log_bound, corrector_pairs,  # noqa: E402
                           flip_delta, tau_leap)
from utils import dasbs as D  # noqa: E402
from utils import dasbs_exact as de  # noqa: E402

torch.set_num_threads(2)
BETA = 2.0


def graph_of(A):
    n = len(A)
    ei = torch.tensor(np.array(np.nonzero(A)), dtype=torch.long)
    g = Data(edge_index=ei, num_nodes=n)
    g.edge_weight = torch.ones(ei.shape[1]) / 2                    # as datasets/mis.py
    g.b = -torch.ones(n, 1)
    g.batch = torch.zeros(n, dtype=torch.long)
    g.A = torch.sparse_coo_tensor(ei, g.edge_weight, (n, n)).to_sparse_csr()
    return g


def lsb_to_msb(n):
    X = D.states(n).numpy()
    return (X * (2 ** np.arange(n - 1, -1, -1))).sum(1)              # row of each LSB state in dasbs_exact order


class TabularCorrector(object):
    """Stands in for the network: log Phihat(x1)_i looked up in a table over states (LSB order)."""

    def __init__(self, table):
        self.table = table                                           # (S, n)

    def corr(self, graph, y):                                        # y (n, K)
        return self.table[D.state_indices(y.T.long())].T


def population_controller(ref, joint, target_fn, X, t):
    """E[exp(target(x1)) | x_t] when (x0, x1) ~ joint (S, S, LSB order) and x_t ~ reference bridge; (S, n)."""
    S, n = X.shape
    x0 = X[:, None, :].expand(S, S, n).reshape(-1, n)
    x1 = X[None, :, :].expand(S, S, n).reshape(-1, n)
    p1 = ref.bridge_prob_one(x0, x1, torch.full((len(x0),), t, dtype=torch.float64))      # (S*S, n)
    law = torch.prod(torch.where(X[None, :, :] == 1, p1[:, None, :], 1 - p1[:, None, :]), -1)   # (pairs, x_t)
    w = torch.as_tensor(joint).reshape(-1, 1) * law                  # P(x0, x1, x_t)
    lab = target_fn(x1).exp()                                       # (pairs, n)
    num = w.T @ lab                                                  # (x_t, n)
    return (num / w.sum(0)[:, None]).numpy()


def setup(n, seed, tau, ref_args):
    rng = np.random.default_rng(seed)
    A = de.random_graph(n, 0.5, rng).astype(np.int64)
    X = D.states(n)
    nu_lsb = torch.softmax(-D.MISGraph(torch.from_numpy(A).float(), BETA).energy(X.double()) / tau, 0).numpy()
    perm = lsb_to_msb(n)
    nu_msb = np.zeros_like(nu_lsb); nu_msb[perm] = nu_lsb
    return A, X, nu_lsb, nu_msb, perm


def test_flip_delta_matches_exact_energy():
    env = Env(SimpleNamespace(beta=BETA, mis_decoding='continuous'))
    for seed in range(5):
        n = 5 + seed
        A, X, *_ = setup(n, seed, 1.0, None)
        g = graph_of(A)
        dH = flip_delta(env, g, X.T.float()).T.double().numpy()
        Xm = X.numpy()
        assert np.abs(dH - de.mis_flip_delta(Xm, A.astype(float), BETA)).max() < 1e-5


def test_memoryless_no_corrector_is_m2_special_case():
    """2B acceptance 1. Memoryless log-linear reference (alpha = 0): corrector bound 0 (network output 0), corr-DM
    targets 0, and the ctrl-AM target -dH/tau has conditional mean = the exact M2 controller under mu x nu."""
    env = Env(SimpleNamespace(beta=BETA, mis_decoding='continuous'))
    ref = D.UniformFlipReference('log_linear', 1.0, 0.0)
    ref2 = de.Reference('loglinear', 1.0, 0.0)
    assert corrector_log_bound(ref) == 0.0
    for seed, n, tau in ((0, 4, 1.0), (1, 5, 0.5), (2, 5, 2.0)):
        A, X, nu, nu_msb, perm = setup(n, seed, tau, None)
        g = graph_of(A)
        net = DASBSNet(torch.nn.Identity(), _Zero(), corrector_log_bound(ref))
        assert (net.corr(g, X.T.float()) == 0).all()
        x0 = X.T.float(); x1 = x0[:, torch.randperm(len(X))]
        _, _, tgt = corrector_pairs(ref, x0, x1, 'zero')
        assert (tgt == 0).all()
        mu = np.full(2 ** n, 2.0 ** -n)
        joint = mu[:, None] * nu[None, :]                            # memoryless SB coupling = mu x nu
        sb = de.ExactSB(np.full(2 ** n, 2.0 ** -n), nu_msb, ref2, n)
        assert np.abs(sb.joint[np.ix_(perm, perm)] - joint).max() < 1e-12
        fn = lambda x1: controller_target(env, net, g, x1.T.float(), tau).T
        for t in (0.1, 0.5, 0.9):
            got = population_controller(ref, joint, fn, X, t)
            want = sb.controller(t)[perm]
            assert np.abs(got / want - 1).max() < 1e-6, (n, t)


class _Zero(torch.nn.Module):
    """Corrector network stand-in with a zero head (DASBSNet zero-initialises .head)."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 1)

    def forward(self, graph, y, time=None):
        return self.head(torch.zeros_like(y))


def test_targets_match_exact_sb_with_exact_corrector():
    """Non-memoryless reference (log-linear gamma 1, alpha 0.5): with the exact SB corrector in the ctrl-AM target,
    the conditional mean is the exact controller; the corr-DM target at t = 0 averages to the exact corrector."""
    env = Env(SimpleNamespace(beta=BETA, mis_decoding='continuous'))
    ref = D.UniformFlipReference('log_linear', 1.0, 0.5)
    for seed, n, tau in ((3, 4, 1.0), (4, 5, 0.7)):
        A, X, nu, nu_msb, perm = setup(n, seed, tau, None)
        g = graph_of(A)
        ex = D.solve_bridge(D.MISGraph(torch.from_numpy(A).float(), BETA), ref, tau)
        S = 2 ** n
        ids = np.arange(S)[:, None] ^ (1 << np.arange(n))
        table = torch.from_numpy(ex.corrector[ids] - ex.corrector[:, None])                 # log Phihat, (S, n)
        bound = corrector_log_bound(ref)
        assert table.abs().max() <= bound + 1e-12                    # math_route2.md §5 range
        fn = lambda x1: controller_target(env, TabularCorrector(table), g, x1.T.float(), tau).T
        for t in (0.0, 0.4, 0.8):
            got = population_controller(ref, ex.joint, fn, X, t)
            want = np.exp(ex.controller_log_ratio(ref, t))
            assert np.abs(got / want - 1).max() < 1e-6, (n, t)
        # negative control: dropping the corrector (all-one) with this non-memoryless reference is off
        fn0 = lambda x1: controller_target(env, TabularCorrector(torch.zeros_like(table)), g, x1.T.float(), tau).T
        got0 = population_controller(ref, ex.joint, fn0, X, 0.4)
        assert np.abs(got0 / np.exp(ex.controller_log_ratio(ref, 0.4)) - 1).max() > 1e-2
        x0 = X[:, None, :].expand(S, S, n).reshape(-1, n).T.float()
        x1 = X[None, :, :].expand(S, S, n).reshape(-1, n).T.float()
        _, _, tgt = corrector_pairs(ref, x0, x1, 'zero')
        w = torch.from_numpy(ex.joint).reshape(-1)
        num = torch.zeros(S, n, dtype=torch.float64).index_add_(0, D.state_indices(x1.T.long()), w[:, None] * tgt.T.exp())
        got = (num / torch.from_numpy(ex.target)[:, None]).numpy()
        assert np.abs(got / table.exp().numpy() - 1).max() < 1e-9


def test_corrector_output_range_and_zero_init():
    import models
    ref = D.UniformFlipReference('log_linear', 1.0, 0.5)
    corr = models.EncodeProcessDecode(n_layers=2, hidden_dim=16, out_channels=1)
    net = DASBSNet(torch.nn.Identity(), corr, corrector_log_bound(ref))
    A, X, *_ = setup(6, 0, 1.0, None)
    g = graph_of(A)
    y = X.T.float()
    assert net.corr(g, y).abs().max() == 0                         # ratio 1 before training
    with torch.no_grad():
        corr.head.bias.fill_(50.0)
    assert abs(float(net.corr(g, y).max()) - corrector_log_bound(ref)) < 1e-6
    assert abs(corrector_log_bound(ref) - np.log(2.0)) < 1e-12      # a = 1/3 -> log 2


def test_tau_leap_reference_and_memoryless():
    """Controller 0 (rates = the reference): the terminal per-bit flip frequency matches the reference kernel to first
    order; with the memoryless reference the first step resamples uniformly, so x1 is independent of x0."""
    class ZeroCtrl(object):
        def ctrl(self, graph, x, t):
            return torch.zeros_like(x)
    A, X, *_ = setup(6, 0, 1.0, None)
    g = graph_of(A)
    torch.manual_seed(0)
    x0 = torch.zeros(6, 20000)
    ref = D.UniformFlipReference('constant', 1.0)
    x1 = tau_leap(ZeroCtrl(), g, x0.clone(), ref, D.time_grid(200, 'uniform'))
    assert abs(float(x1.mean()) - float(ref.flip_probability(0.0, 1.0))) < 0.01
    ref0 = D.UniformFlipReference('log_linear', 1.0, 0.0)
    x1 = tau_leap(ZeroCtrl(), g, x0.clone(), ref0, D.time_grid(50, 'quadratic'))
    assert abs(float(x1.mean()) - 0.5) < 0.01
    assert set(x1.unique().tolist()) <= {0.0, 1.0}


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f'PASS {name} ({time.time()-t0:.1f}s)', flush=True)
        except Exception:
            failed += 1
            print(f'FAIL {name}\n{traceback.format_exc()}', flush=True)
    print(f'{len(tests)-failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
