"""Exact-enumeration checks of the route-2 (discrete ASBS / DASBS) formulas for binary MIS targets
(utils/dasbs_exact.py, docs/20261125/math_route2.md). Acceptance criteria of stage M2 (docs/20261125/plan.md §4):
  1. on >= 20 random small graphs (n <= 10), tau in [0.3, 2], constant and log-linear schedules, errors < 1e-8 for
     the optimal rate (Kolmogorov forward equation), the controller ratio identity, the corrector ratio identity,
     and the reference kernel / bridge (against the matrix exponential);
  2. fixed point: alternating the controller and corrector updates (tabular conditional expectations) converges
     to the Sinkhorn SB (terminal TV < 1e-6);
  3. wrong versions (log-ratio regression, CAM sign labels, no corrector with a non-memoryless reference) do not;
  4. one bit: symbolic verification (sympy).
Identities are checked entry by entry with a relative tolerance RTOL = 1e-10 (all compared quantities are positive),
except the forward equation, whose residual is compared with the size of its terms.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import dasbs_exact as de   # noqa: E402

TOL = 1e-8
RTOL = 1e-10


def rel(a, b):
    """max entrywise relative error of a against b (b > 0 everywhere)."""
    b = np.asarray(b)
    assert (b > 0).all()
    return float(np.max(np.abs(np.asarray(a) - b) / b))
SCHEDULES = (('constant', 1.5, None), ('loglinear', 1.0, 0.5))


def make_ref(sched):
    name, g, a = sched
    return de.Reference(name, g) if name == 'constant' else de.Reference(name, g, a)


def cases(count=20):
    """count random graphs: n in 4..8, edge prob in [0.2, 0.6], tau in [0.3, 2], penalty 2."""
    out = []
    for seed in range(count):
        rng = np.random.default_rng(seed)
        n = 4 + seed % 5
        A = de.random_graph(n, rng.uniform(0.2, 0.6), rng)
        tau = float(rng.uniform(0.3, 2.0))
        out.append((n, A, tau))
    return out


def setup(n, A, tau, sched, lam=2.0):
    X = de.states(n)
    mu = np.full(2 ** n, 1.0 / 2 ** n)
    nu = de.boltzmann(X, A, lam, tau)
    ref = make_ref(sched)
    return X, mu, nu, ref, de.ExactSB(mu, nu, ref, n)


# ------------------------------------------------------------------ reference process

@pytest.mark.parametrize('sched', SCHEDULES)
def test_reference_kernel_is_matrix_exponential(sched):
    from scipy.linalg import expm
    ref = make_ref(sched)
    for n in (1, 2, 4):
        X = de.states(n); L0 = de.unit_generator(n)
        for s, t in ((0.0, 1.0), (0.2, 0.7), (0.5, 0.95)):
            assert np.abs(ref.kernel(X, s, t) - expm(ref.Gamma(s, t) * L0)).max() < 1e-12
        # Chapman-Kolmogorov
        assert np.abs(ref.kernel(X, 0.1, 0.4) @ ref.kernel(X, 0.4, 0.9) - ref.kernel(X, 0.1, 0.9)).max() < 1e-12


def test_mis_flip_delta():
    for n, A, tau in cases(10):
        X = de.states(n); nb = de.neighbours(n)
        H = de.mis_energy(X, A, 2.0)
        assert np.abs(de.mis_flip_delta(X, A, 2.0) - (H[nb] - H[:, None])).max() < 1e-12


# ------------------------------------------------------------------ SB characterisation (DASBS Thm 3.1, eq. 11-13)

@pytest.mark.parametrize('sched', SCHEDULES)
def test_optimal_rate_solves_forward_equation(sched):
    """d/dt P*_t = U_t^T P*_t with P*_t = phi_t phihat_t and U_t the generator with rates r_t phi_t(y)/phi_t(x)."""
    for n, A, tau in cases():
        X, mu, nu, ref, sb = setup(n, A, tau, sched)
        assert sb.marginal_error < 1e-12
        L0 = de.unit_generator(n)
        assert np.abs(sb.marginal(0.0) - mu).max() < TOL and np.abs(sb.marginal(1.0) - nu).max() < TOL
        for t in (0.05, 0.4, 0.8, 0.97):
            g = ref.rate(t)
            ph, phh = sb.phi(t), sb.phihat(t)
            dphi = -g * L0 @ ph                 # backward equation of phi_t = K_{t,1} phi_1
            dphihat = g * L0.T @ phh            # forward equation of phihat_t = K_{0,t}^T phihat_0
            dP = dphi * phh + ph * dphihat
            U = sb.controlled_generator(t)
            P = ph * phh
            scale = np.abs(U.T) @ P + np.abs(dphi * phh) + np.abs(ph * dphihat)   # size of the terms, entrywise
            assert (np.abs(U.T @ P - dP) <= RTOL * scale).all(), (n, t)


@pytest.mark.parametrize('sched', SCHEDULES)
def test_controller_ratio_identity(sched):
    """DASBS eq. 19-21: phi_t(x^i)/phi_t(x) = E*[ exp(-dH_i(x1)/tau) / Phihat(x1)_i | x_t = x ]."""
    for n, A, tau in cases():
        X, mu, nu, ref, sb = setup(n, A, tau, sched)
        Phihat = sb.corrector()
        target = np.exp(-de.mis_flip_delta(X, A, 2.0) / tau) / Phihat          # = phi_1(x1^i)/phi_1(x1)
        phi1 = np.exp(sb.log_phi1)
        assert rel(target, phi1[sb.nb] / phi1[:, None]) < RTOL
        for t in (0.0, 0.3, 0.7, 0.95):
            Kt1 = ref.kernel(X, t, 1.0)
            cond = Kt1 * phi1[None, :] / sb.phi(t)[:, None]                     # P*(x1 | x_t = x)
            assert np.abs(cond.sum(1) - 1).max() < 1e-12
            assert rel(cond @ target, sb.controller(t)) < RTOL


@pytest.mark.parametrize('sched', SCHEDULES)
def test_corrector_ratio_identity(sched):
    """DASBS eq. 25: phihat_1(y^i)/phihat_1(y) = E*[ K_{t,1}(x_t, y^i) / K_{t,1}(x_t, y) | x_1 = y ], t in [0, 1)."""
    for n, A, tau in cases():
        X, mu, nu, ref, sb = setup(n, A, tau, sched)
        phi1 = np.exp(sb.log_phi1)
        for t in (0.0, 0.3, 0.7):
            Kt1 = ref.kernel(X, t, 1.0)
            cond = sb.phihat(t)[:, None] * Kt1 * phi1[None, :] / nu[None, :]      # P*(x_t | x1 = y), columns y
            assert np.abs(cond.sum(0) - 1).max() < 1e-12
            est = np.stack([(cond * Kt1[:, sb.nb[:, i]] / Kt1).sum(0) for i in range(n)], 1)
            assert rel(est, sb.corrector()) < RTOL


@pytest.mark.parametrize('sched', SCHEDULES)
def test_bridge_resampling_reproduces_sb_joint(sched):
    """DASBS eq. 26: (x0, x1) ~ pi* and x_t ~ reference bridge give P*(x_t, x1) = phihat_t(x) K_{t,1}(x, x1) phi_1(x1)."""
    for n, A, tau in cases():
        X, mu, nu, ref, sb = setup(n, A, tau, sched)
        for t in (0.2, 0.6, 0.9):
            J = de.joint_xt_x1(sb.joint, ref, X, t)
            exact = sb.phihat(t)[:, None] * ref.kernel(X, t, 1.0) * np.exp(sb.log_phi1)[None, :]
            assert rel(J, exact) < RTOL


@pytest.mark.parametrize('sched', SCHEDULES)
def test_per_bit_bridge_against_matrix_exponential(sched):
    """DASBS Prop. B.6: given both ends the reference bridge is a product over bits, with the closed form
    de.bridge_prob1; checked against K_{0,t} K_{t,1} / K_{0,1} built from the matrix exponential of the full
    generator, on 20 random (n, x0, x1, t). Also the 2A sampler (utils/dasbs.py) uses the same numbers."""
    import torch
    from scipy.linalg import expm
    from utils import dasbs as D
    ref = make_ref(sched)
    name, gam, alpha = sched
    ref2 = D.UniformFlipReference('constant' if name == 'constant' else 'log_linear', gam, 0.5 if alpha is None else alpha)
    rng = np.random.default_rng(7)
    for k in range(20):
        n = 2 + k % 4
        X = de.states(n); L0 = de.unit_generator(n)
        t = float(rng.uniform(0.05, 0.95))
        a, b = rng.integers(2 ** n), rng.integers(2 ** n)
        K0t, Kt1, K01 = (expm(ref.Gamma(*iv) * L0) for iv in ((0.0, t), (t, 1.0), (0.0, 1.0)))
        law = K0t[a] * Kt1[:, b] / K01[a, b]                                  # P(x_t = x | x0, x1), all x
        assert abs(law.sum() - 1) < 1e-12
        p1 = de.bridge_prob1(ref, X[a], X[b], t)
        prod = np.prod(np.where(X == 1, p1[None, :], 1 - p1[None, :]), axis=1)
        assert rel(prod, law) < 1e-9, k
        p2 = ref2.bridge_prob_one(torch.from_numpy(X[a]), torch.from_numpy(X[b]), t).numpy()
        assert np.abs(p2 - p1).max() < 1e-12


# ------------------------------------------------------------------ fixed points and the alternation (Thm 5.2)

@pytest.mark.parametrize('sched', SCHEDULES)
def test_sb_is_fixed_point_of_both_maps(sched):
    for n, A, tau in cases():
        X, mu, nu, ref, sb = setup(n, A, tau, sched)
        phi1 = np.exp(sb.log_phi1)
        R = phi1[sb.nb] / phi1[:, None]
        for t in (0.1, 0.5, 0.9):
            assert rel(de.ctrl_am_map(sb.joint, R, ref, X, t), sb.controller(t)) < RTOL
        for t in (0.0, 0.5):
            assert rel(de.corr_dm_map(sb.joint, ref, X, sb.nb, t), sb.corrector()) < RTOL
        P01, _ = de.forward_half_bridge(mu, nu, np.log(sb.phihat(1.0)), ref, X)
        assert rel(P01, sb.joint) < RTOL


@pytest.mark.parametrize('sched', SCHEDULES)
def test_alternation_converges_to_sb(sched):
    """Controller = fixed point of (ctrl-AM) for the frozen corrector (closed form, checked above and in
    test_am_iteration_reaches_fixed_point); corrector = (corr-DM). From the all-one corrector, terminal TV < 1e-6."""
    worst = 0.0
    for n, A, tau in cases():
        X, mu, nu, ref, sb = setup(n, A, tau, sched)
        hist = de.ipf_alternation(mu, nu, ref, n, stages=400)
        worst = max(worst, de.tv(hist[-1], nu))
        assert de.tv(hist[-1], nu) < 1e-6, (n, tau, de.tv(hist[-1], nu))
        # the corr-DM labels integrate to a potential (no inconsistency): the ratio field is a gradient
        P01, _ = de.forward_half_bridge(mu, nu, np.zeros(2 ** n), ref, X)
        _, incons = de.ratios_to_log_potential(de.corr_dm_map(P01, ref, X, sb.nb), sb.nb)
        assert incons < 1e-9


def _am_fixed_point(mu, R_fn, ref, X, nb, J, iters=400, eta=0.5, log_version=False, Phis=None, changes=None):
    """Damped tabular (ctrl-AM) iteration on a piecewise-constant controller, grid refined near t = 1.
    Phis: optional start (default all one); changes: optional list that receives the max change per iteration."""
    s = np.linspace(0, 1, J + 1); grid = 1 - (1 - s) ** 2; mids = (grid[:-1] + grid[1:]) / 2
    Phis = [np.ones(nb.shape) for _ in mids] if Phis is None else Phis
    for _ in range(iters):
        P01 = de.evolve_piecewise(mu, Phis, grid, ref, nb)
        R = R_fn(P01)
        new = [(de.ctrl_am_map_log(P01, np.log(R), ref, X, t) if log_version else de.ctrl_am_map(P01, R, ref, X, t))
               for t in mids]
        ch = max(np.abs(a - b).max() for a, b in zip(new, Phis))
        if changes is not None:
            changes.append(ch)
        Phis = [(1 - eta) * a + eta * b for a, b in zip(Phis, new)]
        if ch < 1e-11:
            break
    return de.evolve_piecewise(mu, Phis, grid, ref, nb), Phis, mids


def test_am_iteration_reaches_fixed_point():
    """The damped (ctrl-AM) iteration from Phi = 1 converges to the closed-form half bridge; the remaining error is
    the piecewise-constant time discretisation and falls as 1/J^2."""
    rng = np.random.default_rng(0)
    n = 3; A = de.random_graph(n, 0.5, rng)
    for sched in SCHEDULES:
        X, mu, nu, ref, sb = setup(n, A, 0.7, sched)
        phi1 = np.exp(sb.log_phi1); R = phi1[sb.nb] / phi1[:, None]
        errs = []
        for J in (50, 100):
            P01, _, _ = _am_fixed_point(mu, lambda P: R, ref, X, sb.nb, J)
            errs.append(np.abs(P01 - sb.joint).max())
        assert 3.5 < errs[0] / errs[1] < 4.5, errs
        assert errs[1] < 1e-5, errs


@pytest.mark.parametrize('sched', SCHEDULES)
def test_am_fixed_point_for_any_frozen_corrector(sched):
    """math_route2.md prop. 7.1 (1): for ANY frozen corrector phihat_1 (here all one and random), the damped (ctrl-AM)
    iteration converges to the forward half bridge with phi_1 = nu / phihat_1, which is not the SB (its terminal law
    is off nu). Remaining error: piecewise-constant time discretisation (J = 100); coupling TV to the half bridge
    < 1e-4 and at least 5x smaller than the half bridge's TV to the SB."""
    for gseed, n, tau in ((0, 3, 0.7), (1, 4, 0.5)):
        rng = np.random.default_rng(gseed); A = de.random_graph(n, 0.5, rng)
        X, mu, nu, ref, sb = setup(n, A, tau, sched)
        for log_phihat1 in (np.zeros(2 ** n), np.random.default_rng(5).normal(size=2 ** n)):
            P_half, log_phi1 = de.forward_half_bridge(mu, nu, log_phihat1, ref, X)
            phi1 = np.exp(log_phi1 - log_phi1.max()); R = phi1[sb.nb] / phi1[:, None]
            P_it, _, _ = _am_fixed_point(mu, lambda P: R, ref, X, sb.nb, 100)
            err = de.tv(P_it, P_half)
            assert err < 1e-4, err                                  # measured 8e-6 - 8e-5
            assert de.tv(P_half, sb.joint) > 5 * err                # a genuinely different fixed point (7e-4 - 0.34)


def test_alternation_with_iterated_controller():
    """The alternation with the controller obtained by the damped (ctrl-AM) iteration (not the closed form) and the
    corrector by (corr-DM) on the resulting coupling: follows the exact alternation stage by stage up to the time
    discretisation, and reaches nu."""
    rng = np.random.default_rng(0)
    n = 3; A = de.random_graph(n, 0.5, rng)
    for sched in SCHEDULES:
        X, mu, nu, ref, sb = setup(n, A, 0.7, sched)
        exact = de.ipf_alternation(mu, nu, ref, n, stages=6)
        log_phihat1 = np.zeros(2 ** n); Phis = None; got = []
        dH = de.mis_flip_delta(X, A, 2.0)
        for stage in range(6):
            R = np.exp(-dH / 0.7) / np.exp(log_phihat1[sb.nb] - log_phihat1[:, None])     # nu ratio / corrector
            P01, Phis, _ = _am_fixed_point(mu, lambda P: R, ref, X, sb.nb, 50, Phis=Phis)
            got.append(P01.sum(0))
            log_phihat1, _ = de.ratios_to_log_potential(de.corr_dm_map(P01, ref, X, sb.nb), sb.nb)
        for k in range(6):
            assert de.tv(got[k], exact[k]) < 5e-4, (k, de.tv(got[k], exact[k]))
        assert de.tv(exact[0], nu) > 1e-2 and de.tv(got[-1], nu) < 5e-4


def test_undamped_am_iteration_oscillates():
    """math_route2.md §7: without damping (eta = 1) the tabular (ctrl-AM) iteration does not settle (period-2
    oscillation); with eta = 0.5 it converges (test_am_iteration_reaches_fixed_point)."""
    rng = np.random.default_rng(0)
    n = 3; A = de.random_graph(n, 0.5, rng)
    for sched in SCHEDULES:
        X, mu, nu, ref, sb = setup(n, A, 0.7, sched)
        phi1 = np.exp(sb.log_phi1); R = phi1[sb.nb] / phi1[:, None]
        changes = []
        _am_fixed_point(mu, lambda P: R, ref, X, sb.nb, 50, iters=200, eta=1.0, changes=changes)
        assert min(changes[-20:]) > 1.0, changes[-5:]


def test_memoryless_reference_needs_no_corrector():
    """Corollary 7.2, directly with the memoryless log-linear reference (alpha = 0; before only a gamma = 40 proxy):
    corr-DM at t = 0 returns the all-one corrector for any coupling, and the controller fixed point with the all-one
    corrector is already the SB (terminal law nu)."""
    ref = de.Reference('loglinear', 1.0, 0.0)
    assert ref.signal(0.0, 1.0) == 0.0 and ref.signal(0.3, 1.0) > 0
    for n, A, tau in cases(6):
        X = de.states(n); nb = de.neighbours(n)
        mu = np.full(2 ** n, 1.0 / 2 ** n); nu = de.boltzmann(X, A, 2.0, tau)
        P01, _ = de.forward_half_bridge(mu, nu, np.zeros(2 ** n), ref, X)
        assert de.tv(P01.sum(0), nu) < 1e-12 and np.abs(P01.sum(1) - mu).max() < 1e-15
        rng = np.random.default_rng(n)
        Q = rng.random((2 ** n, 2 ** n)); Q /= Q.sum()
        assert np.abs(de.corr_dm_map(Q, ref, X, nb, 0.0) - 1).max() < 1e-12


# ------------------------------------------------------------------ wrong versions must fail

WRONG_CASES = [(0, 3, 0.7, SCHEDULES[0]), (0, 3, 0.7, SCHEDULES[1]), (1, 4, 0.5, SCHEDULES[0]),
               (1, 4, 0.5, SCHEDULES[1]), (2, 4, 1.0, SCHEDULES[0]), (2, 4, 1.0, SCHEDULES[1])]


@pytest.mark.parametrize('case', WRONG_CASES)
def test_wrong_versions_fail(case):
    """3 graphs x 2 schedules. Right version with the same grid: terminal TV 4e-5 - 1.4e-4; the wrong ones are
    100x or more off (log-ratio controller, CAM labels), or off nu by > 1e-4 where the right version is exact."""
    gseed, n, tau, sched = case
    rng = np.random.default_rng(gseed)
    A = de.random_graph(n, 0.5, rng)
    X, mu, nu, ref, sb = setup(n, A, tau, sched)
    nb = sb.nb
    phi1 = np.exp(sb.log_phi1); R_true = phi1[nb] / phi1[:, None]
    dH = de.mis_flip_delta(X, A, 2.0)
    # baseline: the right label with the same grid gets close
    P_ok, _, _ = _am_fixed_point(mu, lambda P: R_true, ref, X, nb, 50)
    ok = de.tv(P_ok.sum(0), nu)
    assert ok < 2e-4
    # (a) controller regresses the log-ratio (geometric instead of arithmetic mean)
    P_log, _, _ = _am_fixed_point(mu, lambda P: R_true, ref, X, nb, 50, log_version=True)
    assert de.tv(P_log.sum(0), nu) > 100 * ok, de.tv(P_log.sum(0), nu)
    # (a') corrector regresses the log-ratio, exact alternation otherwise
    hist = de.ipf_alternation(mu, nu, ref, n, stages=400, log_ratio_corrector=True)
    assert de.tv(hist[-1], nu) > 1e-3, de.tv(hist[-1], nu)
    assert de.tv(de.ipf_alternation(mu, nu, ref, n, stages=400)[-1], nu) < 1e-6         # the right alternation
    # (b) CAM sign-propagated label with the SB corrector frozen (controller iteration only, not an alternation):
    #     log Phi_t(x)_i = -E[ s_i(t,1) (dH_i(x1)/tau + log Phihat(x1)_i) | x_t ]
    log_Phihat = np.log(sb.corrector())
    s = np.linspace(0, 1, 51); grid = 1 - (1 - s) ** 2; mids = (grid[:-1] + grid[1:]) / 2
    Phis = [np.ones(nb.shape) for _ in mids]
    for _ in range(300):
        P01 = de.evolve_piecewise(mu, Phis, grid, ref, nb)
        new = []
        for t in mids:
            K0t, Kt1, K01 = ref.kernel(X, 0, t), ref.kernel(X, t, 1), ref.kernel(X, 0, 1)
            W = np.einsum('ab,ax,xb->xb', P01 / K01, K0t, Kt1)             # p(x_t, x1)
            W = W / W.sum(1, keepdims=True)
            sign = np.where(X[:, None, :] == X[None, :, :], 1.0, -1.0)     # (x_t, x1, i)
            lab = sign * (dH / tau + log_Phihat)[None, :, :]
            new.append(np.exp(-(W[:, :, None] * lab).sum(1)))
        Phis = [0.5 * a + 0.5 * b for a, b in zip(Phis, new)]
    P_cam = de.evolve_piecewise(mu, Phis, grid, ref, nb)
    assert de.tv(P_cam.sum(0), nu) > 100 * ok, de.tv(P_cam.sum(0), nu)
    # (c) no corrector with a non-memoryless reference: the controller fixed point ends off nu ...
    P_nc, _ = de.forward_half_bridge(mu, nu, np.zeros(2 ** n), ref, X)
    assert de.tv(P_nc.sum(0), nu) > 1e-4
    P_sb, _ = de.forward_half_bridge(mu, nu, np.log(sb.phihat(1.0)), ref, X)            # with the SB corrector
    assert de.tv(P_sb.sum(0), nu) < 1e-12
    # ... while with a (nearly) memoryless reference the same thing is fine (DASBS Prop. B.12 remark)
    ref_m = de.Reference('constant', 40.0)
    P_m, _ = de.forward_half_bridge(mu, nu, np.zeros(2 ** n), ref_m, X)
    assert de.tv(P_m.sum(0), nu) < 1e-8


# ------------------------------------------------------------------ one bit, symbolically

def test_one_bit_symbolic():
    sp = pytest.importorskip('sympy')
    t, s, g = sp.symbols('t s gamma', positive=True)
    u0, u1, v0, v1 = sp.symbols('u0 u1 v0 v1', positive=True)        # phi_1 and phihat_0, arbitrary positive
    def K(a):   # 2x2 channel, rows from, cols to
        return sp.Matrix([[(1 + a) / 2, (1 - a) / 2], [(1 - a) / 2, (1 + a) / 2]])
    a = lambda lo, hi: sp.exp(-g * (hi - lo))                           # constant schedule
    phi1 = sp.Matrix([u0, u1]); phihat0 = sp.Matrix([v0, v1])
    phi_t = K(a(t, 1)) * phi1
    phihat_t = K(a(0, t)).T * phihat0
    phihat1 = K(a(0, 1)).T * phihat0
    # (i) Kolmogorov: d/dt (phi_t phihat_t) = U^T (phi_t phihat_t), rates gamma/2 * phi_t(other)/phi_t(this)
    P = sp.Matrix([phi_t[0] * phihat_t[0], phi_t[1] * phihat_t[1]])
    r01 = g / 2 * phi_t[1] / phi_t[0]; r10 = g / 2 * phi_t[0] / phi_t[1]
    U = sp.Matrix([[-r01, r01], [r10, -r10]])
    assert sp.simplify(P.diff(t) - U.T * P) == sp.zeros(2, 1)
    # (ii) controller identity at x_t = 0: phi_t(1)/phi_t(0) = E[ phi_1(1 - x1)/phi_1(x1) | x_t = 0 ]
    Kt1 = K(a(t, 1))
    cond0 = [Kt1[0, j] * phi1[j] / phi_t[0] for j in (0, 1)]
    lhs = phi_t[1] / phi_t[0]
    rhs = cond0[0] * phi1[1] / phi1[0] + cond0[1] * phi1[0] / phi1[1]
    assert sp.simplify(lhs - rhs) == 0
    # (iii) corrector identity at y = 0: phihat_1(1)/phihat_1(0) = E[ K_{t,1}(x_t, 1)/K_{t,1}(x_t, 0) | x1 = 0 ]
    nu0 = phi1[0] * phihat1[0]
    cond = [phihat_t[x] * Kt1[x, 0] * phi1[0] / nu0 for x in (0, 1)]
    rhs = sum(cond[x] * Kt1[x, 1] / Kt1[x, 0] for x in (0, 1))
    assert sp.simplify(phihat1[1] / phihat1[0] - rhs) == 0


def test_cam_counterexample_one_bit():
    """math_route2.md §12 (re-check of sb_worktrees/CAM/docs/20261114/sb_mis_dasbs_cam_discussion.md §11): one bit,
    tau = 1, terminal cost F(0) = 0, F(1) = 2, reference flip rate 1 each way for the remaining time 1 (signal
    e^{-2}). At x_t = 0 the exact multiplier phi(1)/phi(0) equals the DASBS label mean, while the CAM label mean is
    2 and the log-ratio regression gives a third value."""
    import math
    a = math.exp(-2.0)
    p = (1 + a) / 2
    phi0 = p + (1 - p) * math.exp(-2.0); phi1 = (1 - p) + p * math.exp(-2.0)
    prob0 = p / phi0; prob1 = (1 - p) * math.exp(-2.0) / phi0              # P*(X_1 | X_t = 0)
    assert abs(prob0 + prob1 - 1) < 1e-15
    dasbs = prob0 * math.exp(-2.0) + prob1 * math.exp(2.0)                 # E[exp(-dF(X_1))]
    cam = prob0 * (+1) * 2.0 + prob1 * (-1) * (-2.0)                       # E[s * dF(X_1)]
    log_reg = math.exp(prob0 * (-2.0) + prob1 * 2.0)                       # exp(E[log ratio])
    assert abs(dasbs - phi1 / phi0) < 1e-12 and abs(phi1 / phi0 - 0.8131206389) < 1e-9
    assert abs(-math.log(phi1 / phi0) - 0.2068757931) < 1e-9
    assert abs(cam - 2.0) < 1e-12 and abs(math.exp(-cam) - phi1 / phi0) > 0.6
    assert abs(log_reg - phi1 / phi0) > 0.5
    # the same through the exact SB machinery: phi_1 = exp(-F), generic tools
    X = de.states(1); nb = de.neighbours(1)
    ref = de.Reference('constant', 2.0)                                     # rate gamma / 2 = 1 per direction
    phi_t = ref.kernel(X, 0.0, 1.0) @ np.exp(-np.array([0.0, 2.0]))
    assert abs(phi_t[nb[0, 0]] / phi_t[0] - dasbs) < 1e-12


@pytest.mark.parametrize('sched', SCHEDULES)
def test_alternation_stage_is_one_sinkhorn_step(sched):
    """math_route2.md §7: one stage of the alternation, written in potentials, is exactly one Sinkhorn (IPF) step:
    controller fixed point phi_1 = nu / phihat_1, then corr-DM gives phihat_1' ∝ K_{0,1}^T (mu / K_{0,1} phi_1),
    for any t in [0, 1) of the corr-DM regression."""
    for n, A, tau in cases():
        X, mu, nu, ref, sb = setup(n, A, tau, sched)
        nb = sb.nb; K01 = ref.kernel(X, 0.0, 1.0)
        rng = np.random.default_rng(n)
        log_phihat1 = rng.normal(size=2 ** n)                      # arbitrary current corrector
        P01, log_phi1 = de.forward_half_bridge(mu, nu, log_phihat1, ref, X)
        phi1 = np.exp(log_phi1 - log_phi1.max())
        phihat1_new = K01.T @ (mu / (K01 @ phi1))                   # Sinkhorn half-steps
        R_sink = phihat1_new[nb] / phihat1_new[:, None]
        for t in (0.0, 0.4, 0.9):
            assert rel(de.corr_dm_map(P01, ref, X, nb, t), R_sink) < 1e-9
