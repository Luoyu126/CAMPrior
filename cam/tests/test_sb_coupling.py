"""Exact-enumeration checks of the route-1 Schrödinger bridge formulas (utils/sb_coupling.py,
docs/20261125/math_route1.md). Every check enumerates all 2^n states (n <= 9) in float64.

Acceptance criteria of stage M1 (docs/20261125/plan.md section 4):
  1. on >= 20 random instances x p in {0.1, 0.3, 0.45}, errors < 1e-8 for the Sinkhorn marginals, the closed form
     of the coupling, P*(y | x_k) at >= 3 intermediate k, and the end distribution of the multi-step sampler;
  2. Monte Carlo Sinkhorn converges to the exact solution as the number of samples grows;
  3. at p = 0.5 the target equals label_sets.posterior_mean (< 1e-6; here < 1e-12 at every k);
  4. deliberately wrong versions fail the same checks.
"""
import itertools
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import sb_coupling as sb                       # noqa: E402
from utils.label_sets import posterior_mean as pm_difusco  # noqa: E402
from utils.diffusion_schedulers import CategoricalDiffusion  # noqa: E402

TOL = 1e-8
PS = (0.1, 0.3, 0.45)
N_INSTANCES = 20
T = 4


def states(n):
    return torch.tensor(list(itertools.product([0, 1], repeat=n)), dtype=torch.float64)


def kernel(a, X):
    """Full matrix K_a(x' | x) over all states, rows x, columns x'."""
    return sb.log_kernel(X, X, a).exp()


def index_of(X, rows):
    return [int(torch.nonzero((X == r).all(1))[0]) for r in rows]


def instance(seed, p):
    g = torch.Generator().manual_seed(seed)
    n = 6 + seed % 4
    M = 2 + seed % 5
    X = states(n)
    idx = torch.randperm(len(X), generator=g)[:M]
    labels = X[idx]
    pi1 = torch.rand(M, generator=g, dtype=torch.float64) + 0.2   # non-uniform target weights
    pi1 = pi1 / pi1.sum()
    a_T = 1 - 2 * p
    a = torch.tensor([1.0] + [a_T + (1 - a_T) * (1 - k / T) ** 1.5 for k in range(1, T)] + [a_T], dtype=torch.float64)
    return dict(n=n, M=M, X=X, labels=labels, lab_idx=idx.tolist(), pi1=pi1, a_T=a_T, a=a)


def exact_sinkhorn(inst):
    X = inst['X']
    w = torch.full((len(X),), 1.0 / len(X), dtype=torch.float64)
    log_g, err, it = sb.sinkhorn(inst['labels'], inst['pi1'], inst['a_T'], X, w, iters=200000, tol=1e-14)
    return log_g, w


def reference_sinkhorn(inst):
    """Textbook Sinkhorn on the matrix C[z, m] = K_{a_T}(s_m | z) with both scalings, written independently of
    sb.sinkhorn: P = diag(u) C diag(v), rows sum to 2^-n, columns to pi1."""
    X, labels, pi1 = inst['X'], inst['labels'], inst['pi1']
    C = kernel(inst['a_T'], X)[:, inst['lab_idx']]
    r = torch.full((len(X),), 1.0 / len(X), dtype=torch.float64)
    v = torch.ones(len(pi1), dtype=torch.float64)
    for _ in range(200000):
        u = r / (C @ v)
        v_new = pi1 / (C.T @ u)
        if (v_new - v).abs().max() < 1e-15 * v.abs().max():
            v = v_new
            break
        v = v_new
    u = r / (C @ v)
    return u[:, None] * C * v[None, :]


def joint_z_xk_m(inst, P_zm, a_k):
    """P(z, x_k, m) = pi*(z, m) * bridge(x_k | z, s_m), the bridge from the full matrices:
    K_{a_k}(x_k | s_m) K_{a_T / a_k}(z | x_k) / K_{a_T}(z | s_m)."""
    X = inst['X']; a_T = inst['a_T']
    K1 = kernel(a_k, X)[inst['lab_idx']]                  # (M, S): K_{a_k}(x | s_m)
    K2 = kernel(a_T / a_k, X)                             # (S, S): [x, z] = K(z | x)
    K3 = kernel(a_T, X)[:, inst['lab_idx']]               # (S, M): [z, m] = K(s_m | z) = K(z | s_m)
    # J[z, x, m]
    return P_zm[:, None, :] * K1.T[None, :, :] * K2.T[:, :, None] / K3[:, None, :]


# ---------------------------------------------------------------- 1. static coupling

@pytest.mark.parametrize('p', PS)
def test_sinkhorn_marginals_and_closed_form(p):
    for seed in range(N_INSTANCES):
        inst = instance(seed, p)
        log_g, w = exact_sinkhorn(inst)
        P = w[:, None] * sb.coupling_conditional(inst['X'], inst['labels'], log_g, inst['a_T'])
        assert (P.sum(1) - w).abs().max() < TOL
        assert (P.sum(0) - inst['pi1']).abs().max() < TOL
        # the coupling equals the textbook two-sided Sinkhorn solution (SB static problem, unique)
        P_ref = reference_sinkhorn(inst)
        assert (P - P_ref).abs().max() < TOL
        # closed form in the overlap |z ∧ s_m|: pi(m | z) ∝ g_m exp(-w |s_m| + 2 w |z ∧ s_m|)
        om = sb.omega(inst['a_T'])
        s = inst['labels']
        L = log_g[None, :] - om * s.sum(1)[None, :] + 2 * om * inst['X'] @ s.T
        assert (torch.softmax(L, 1) - P / w[:, None]).abs().max() < TOL


# ---------------------------------------------------------------- 2. dynamic: posterior given x_k, bridge

@pytest.mark.parametrize('p', PS)
def test_markov_posterior_and_bridge(p):
    for seed in range(N_INSTANCES):
        inst = instance(seed, p)
        log_g, w = exact_sinkhorn(inst)
        P_zm = w[:, None] * sb.coupling_conditional(inst['X'], inst['labels'], log_g, inst['a_T'])
        X = inst['X']
        # the channel composes: K_{a_k} then K_{a_T / a_k} is K_{a_T}
        for k in range(1, T):
            a_k = inst['a'][k].item()
            assert (kernel(a_k, X) @ kernel(inst['a_T'] / a_k, X) - kernel(inst['a_T'], X)).abs().max() < 1e-12
            J = joint_z_xk_m(inst, P_zm, a_k)
            assert abs(J.sum().item() - 1) < TOL
            assert (J.sum(1) - P_zm).abs().max() < TOL           # summing x_k back gives the coupling
            P_xm = J.sum(0)                                     # (S, M)
            post_true = P_xm / P_xm.sum(1, keepdim=True)
            post = sb.label_log_posterior(X, inst['labels'], log_g, a_k).exp()
            assert (post - post_true).abs().max() < TOL, f'P*(y | x_k) at k={k}'
            # Theorem 3.1: given x_k, y does not depend on z (checked on (z, x_k) with mass > 1e-12)
            Jzx = J.sum(2)
            mask = Jzx > 1e-12
            cond = J / torch.where(mask, Jzx, torch.ones_like(Jzx))[:, :, None]
            diffm = (cond - post[None, :, :]).abs().amax(2)
            assert diffm[mask].max() < 1e-7, f'conditional independence at k={k}'
            # the per-bit bridge equals the bridge from the full matrices
            for m in range(inst['M']):
                pb = sb.bridge_prob1(inst['labels'][m][None, :].expand(len(X), -1), X, a_k, inst['a_T'])
                # probability of every x given (z, s_m): product over bits
                Pb = torch.ones(len(X), len(X), dtype=torch.float64)     # [z, x]
                for i in range(inst['n']):
                    xi = X[:, i][None, :]
                    pi = pb[:, i][:, None]
                    Pb = Pb * (xi * pi + (1 - xi) * (1 - pi))
                ref = J[:, :, m] / P_zm[:, m][:, None]
                assert (Pb - ref).abs().max() < TOL


def test_bridge_endpoints():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (50, 8), generator=g).double(); z = torch.randint(0, 2, (50, 8), generator=g).double()
    for a_T in (0.0, 0.2, 0.8):
        assert torch.equal(sb.bridge_prob1(y, z, 1.0, a_T), y)
        assert torch.equal(sb.bridge_prob1(y, z, a_T, a_T), z)          # k = T is the noise end, also at a_T = 0
    with pytest.raises(AssertionError):
        sb.bridge_prob1(y, z, 0.1, 0.2)                                   # a_k < a_T is not a valid index
    # a_T = 0: the bridge is the forward marginal K_{a_k}(. | y)
    assert torch.allclose(sb.bridge_prob1(y, z, 0.3, 0.0), y * 0.65 + (1 - y) * 0.35, atol=1e-15)


# ---------------------------------------------------------------- 3. samplers end on pi1

def ancestral_end(inst, log_g, post_fn, bridge_fn, keep=None):
    """Exact distribution of the Markov posterior sampler x_T = z -> ... -> x_0 over all states:
    x_{k-1} ~ sum_m P(m | x_k) bridge(x_{k-1} | x_k, s_m). Returns the marginals over states; with keep = {}
    also stores keep['zx'][k] = P(z, x_k) and keep['pair'][k] = P(x_k, x_{k-1})."""
    X = inst['X']; a = inst['a']
    p = torch.full((len(X),), 1.0 / len(X), dtype=torch.float64)
    zx = torch.diag(p)
    margs = {T: p}
    if keep is not None:
        keep['zx'] = {T: zx}; keep['pair'] = {}
    for k in range(T, 0, -1):
        post = post_fn(X, a[k].item())                       # (S, M)
        Tr = torch.zeros(len(X), len(X), dtype=torch.float64)
        for m in range(inst['M']):
            Tr += post[:, m][:, None] * bridge_fn(inst['labels'][m], a[k - 1].item(), a[k].item())
        if keep is not None:
            keep['pair'][k] = p[:, None] * Tr
            zx = zx @ Tr
            keep['zx'][k - 1] = zx
        p = p @ Tr
        margs[k - 1] = p
    return margs


def bridge_back(inst):
    """bridge(x_{k-1} | x_k, y) from the full matrices: K_{a'}(x' | y) K_{a / a'}(x | x') / K_a(x | y), rows x, cols x'."""
    X = inst['X']
    def fn(y, a_prev, a_cur):
        Ky_prev = sb.log_kernel(y[None, :], X, a_prev).exp()[0]     # K(x' | y), (S,)
        Ky_cur = sb.log_kernel(y[None, :], X, a_cur).exp()[0]       # K(x | y)
        K_step = kernel(a_cur / a_prev, X)                          # [x', x] = K(x | x')
        return Ky_prev[None, :] * K_step.T / Ky_cur[:, None]
    return fn


@pytest.mark.parametrize('p', PS)
def test_markov_sampler_ends_on_pi1(p):
    for seed in range(N_INSTANCES):
        inst = instance(seed, p)
        log_g, w = exact_sinkhorn(inst)
        post = lambda X, a_k: sb.label_log_posterior(X, inst['labels'], log_g, a_k).exp()
        keep = {}
        margs = ancestral_end(inst, log_g, post, bridge_back(inst), keep)
        end = torch.zeros(len(inst['X']), dtype=torch.float64); end[inst['lab_idx']] = inst['pi1']
        assert (margs[0] - end).abs().max() < TOL
        # every intermediate marginal is the SB marginal, and so are the (z, x_k) joints, including (z, x_0) = pi*
        P_zm = w[:, None] * sb.coupling_conditional(inst['X'], inst['labels'], log_g, inst['a_T'])
        Jend = torch.zeros(len(inst['X']), len(inst['X']), dtype=torch.float64); Jend[:, inst['lab_idx']] = P_zm
        assert (keep['zx'][0] - Jend).abs().max() < TOL
        for k in range(1, T):
            J = joint_z_xk_m(inst, P_zm, inst['a'][k].item())
            assert (margs[k] - J.sum((0, 2))).abs().max() < TOL
            assert (keep['zx'][k] - J.sum(2)).abs().max() < TOL
        # adjacent pairs (x_k, x_{k-1}) equal the SB pair law sum_m P*(x_k, m) R(x_{k-1} | x_k, s_m)
        bb = bridge_back(inst)
        for k in range(2, T):
            J = joint_z_xk_m(inst, P_zm, inst['a'][k].item()).sum(0)          # P*(x_k, m)
            pair = sum(J[:, m][:, None] * bb(inst['labels'][m], inst['a'][k - 1].item(), inst['a'][k].item())
                       for m in range(inst['M']))
            assert (keep['pair'][k] - pair).abs().max() < TOL


def bridge_renoise_joint(inst, log_g, post_fn, bridge_zy):
    """Exact joint P(z, x_k) of the bridge re-noising sampler: x_{k-1} ~ sum_m P(m | x_k) bridge(x_{k-1} | z, s_m)."""
    X = inst['X']; a = inst['a']; S = len(X)
    J = torch.diag(torch.full((S,), 1.0 / S, dtype=torch.float64))   # P(z, x_T) = U(z) [x_T = z]
    out = {T: J}
    for k in range(T, 0, -1):
        post = post_fn(X, a[k].item())                                 # (S_x, M)
        new = torch.zeros(S, S, dtype=torch.float64)
        for m in range(inst['M']):
            B = bridge_zy(inst['labels'][m], a[k - 1].item())           # [z, x'] = bridge(x' | z, s_m)
            new += (J * post[:, m][None, :]).sum(1)[:, None] * B
        J = new
        out[k - 1] = J
    return out


def bridge_zy_exact(inst):
    X = inst['X']
    def fn(y, a_k):
        if a_k == 1.0:
            B = torch.zeros(len(X), len(X), dtype=torch.float64); B[:, index_of(X, [y])[0]] = 1
            return B
        pb = sb.bridge_prob1(y[None, :].expand(len(X), -1), X, a_k, inst['a_T'])   # rows z
        B = torch.ones(len(X), len(X), dtype=torch.float64)
        for i in range(inst['n']):
            xi = X[:, i][None, :]; pi = pb[:, i][:, None]
            B = B * (xi * pi + (1 - xi) * (1 - pi))
        return B
    return fn


@pytest.mark.parametrize('p', PS)
def test_bridge_renoise_sampler(p):
    for seed in range(N_INSTANCES):
        inst = instance(seed, p)
        log_g, w = exact_sinkhorn(inst)
        post = lambda X, a_k: sb.label_log_posterior(X, inst['labels'], log_g, a_k).exp()
        out = bridge_renoise_joint(inst, log_g, post, bridge_zy_exact(inst))
        P_zm = w[:, None] * sb.coupling_conditional(inst['X'], inst['labels'], log_g, inst['a_T'])
        for k in range(1, T):
            J = joint_z_xk_m(inst, P_zm, inst['a'][k].item())
            assert (out[k] - J.sum(2)).abs().max() < TOL, f'(z, x_k) joint at k={k}'
        end = torch.zeros_like(out[0]); end[:, inst['lab_idx']] = P_zm
        assert (out[0] - end).abs().max() < TOL


# ---------------------------------------------------------------- 4. Monte Carlo Sinkhorn

def test_monte_carlo_sinkhorn_converges():
    n = 10
    X = states(n)
    g = torch.Generator().manual_seed(1)
    labels = X[torch.randperm(len(X), generator=g)[:6]]
    pi1 = torch.full((6,), 1 / 6, dtype=torch.float64)
    a_T = 0.4
    w = torch.full((len(X),), 1.0 / len(X), dtype=torch.float64)
    errs = []
    for K in (256, 4096, 65536):
        e = []
        for rep in range(5):
            gz = torch.Generator().manual_seed(100 + rep)
            z = torch.randint(0, 2, (K, n), generator=gz).double()
            log_g, _, _ = sb.sinkhorn(labels, pi1, a_T, z, iters=5000, tol=1e-13)
            marg = (w[:, None] * sb.coupling_conditional(X, labels, log_g, a_T)).sum(0)   # under the exact pi0
            e.append((marg - pi1).abs().max().item())
        errs.append(float(np.mean(e)))
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[2] < 5e-3, errs


# ---------------------------------------------------------------- 5. the independent limit p = 0.5

def test_independent_limit_matches_difusco_soft_targets():
    diff = CategoricalDiffusion(T=50, schedule='cosine')
    a = sb.sb_signal_schedule(diff.Q_bar, 0.0)
    assert (torch.from_numpy(sb.signal_to_Q_bar(a, as_numpy=True)) - torch.from_numpy(diff.Q_bar)).abs().max() < 1e-12
    g = torch.Generator().manual_seed(0)
    n, M = 40, 7
    labels = torch.randint(0, 2, (M, n), generator=g)
    assert len({tuple(r.tolist()) for r in labels}) == M
    pi1 = torch.full((M,), 1 / M, dtype=torch.float64)
    z = torch.randint(0, 2, (4096, n), generator=g)
    log_g, err, _ = sb.sinkhorn(labels, pi1, a[-1].item(), z, tol=1e-14)
    assert err < 1e-6                       # Monte Carlo g at a[T] ~ 1e-6: within MC noise of uniform
    for k in (1, 10, 25, 49, 50):
        xt = torch.randint(0, 2, (64, n), generator=g)
        ours = sb.posterior_mean(xt, labels, log_g, a[k].item())
        theirs = torch.stack([pm_difusco(labels, x, torch.from_numpy(diff.Q_bar[k]).double()) for x in xt])
        assert (ours - theirs).abs().max() < 1e-6, k
    # exact Sinkhorn on all states at a_T = 0 with a non-uniform pi1: g ∝ pi1 (Prop 6.1). sinkhorn starts from
    # log pi1, so this alone would not catch a Sinkhorn that never updates; the a_T > 0 marginal tests do.
    X = states(6)
    lab6 = X[torch.randperm(64, generator=g)[:5]]
    pi6 = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3], dtype=torch.float64)
    log_g6, err6, _ = sb.sinkhorn(lab6, pi6, 0.0, X)
    assert err6 < 1e-12
    d6 = log_g6 - torch.log(pi6)
    assert (d6 - d6.mean()).abs().max() < 1e-12
    # continuity: as a_T -> 0 the SB target approaches the independent one
    gaps = []
    for aT in (1e-1, 1e-2, 1e-3):
        lg, _, _ = sb.sinkhorn(labels, pi1, aT, z, tol=1e-13)
        xt = torch.randint(0, 2, (64, n), generator=torch.Generator().manual_seed(5))
        gaps.append((sb.posterior_mean(xt, labels, lg, 0.5) - sb.posterior_mean(xt, labels, torch.zeros(M), 0.5)).abs().max().item())
    assert gaps[0] > gaps[1] > gaps[2], gaps


def test_signal_schedule_positive_a_T():
    diff = CategoricalDiffusion(T=50, schedule='cosine')
    for aT in (0.1, 0.5):
        a = sb.sb_signal_schedule(diff.Q_bar, aT)
        assert a[0] == 1.0 and bool((a[1:] <= a[:-1]).all())
        assert a[-1] >= aT and a[-1] - aT < 1e-5 and bool((a >= a[-1]).all())


def product_sampler_tv(inst, log_g, which, N):
    """Exact end law of samplers A and B when the posterior is replaced by its per-bit marginals (what a
    node-wise network outputs). Signals a_k = a_T + (1 - a_T)(1 - k/N). Returns TV(end, pi1)."""
    X = inst['X']; n = inst['n']; S = len(X); a_T = inst['a_T']
    a = [a_T + (1 - a_T) * (1 - k / N) for k in range(N + 1)]
    def bitprob(p1, lo, hi, x_cond):           # P(x_lo = 1) per bit for sampler A: sum_y p(y) R(x_lo | x_hi, y)
        out = 0
        for yv, py in ((1.0, p1), (0.0, 1 - p1)):
            yv_t = torch.full_like(p1, yv)
            num = sb.bridge_prob1(yv_t, x_cond, lo, hi)    # bridge between y (signal lo) and x_hi as the far end
            out = out + py * num
        return out
    def law(prob):                             # (R, n) per-bit probabilities -> (R, S) state probabilities
        return torch.prod(torch.where(X[None, :, :] == 1, prob[:, None, :], 1 - prob[:, None, :]), 2)
    if which == 'A':
        p = torch.full((S,), 1.0 / S, dtype=torch.float64)
        for k in range(N, 0, -1):
            p1 = sb.posterior_mean(X, inst['labels'], log_g, a[k])
            # bridge between y at signal a[k-1] and x_k at signal a[k] relative to y: use the per-bit bridge with
            # (a_k, a_T) -> (a[k-1], a[k]) and the 'noise end' x_k
            prob = bitprob(p1, a[k - 1], a[k], X)
            p = p @ law(prob)
        end = p
    else:
        J = torch.diag(torch.full((S,), 1.0 / S, dtype=torch.float64))   # P(z, x_k)
        for k in range(N, 0, -1):
            p1 = sb.posterior_mean(X, inst['labels'], log_g, a[k])           # (S_x, n)
            new = torch.zeros_like(J)
            for zi in range(S):
                z = X[zi][None, :].expand(S, -1)
                prob = p1 * sb.bridge_prob1(torch.ones_like(p1), z, a[k - 1], a_T) + \
                    (1 - p1) * sb.bridge_prob1(torch.zeros_like(p1), z, a[k - 1], a_T)
                new[zi] = J[zi] @ law(prob)
            J = new
        end = J.sum(0)
    target = torch.zeros(S, dtype=torch.float64); target[inst['lab_idx']] = inst['pi1']
    return 0.5 * (end - target).abs().sum().item()


def test_product_denoiser_samplers():
    """With per-bit (product) posteriors, sampler A's error vanishes as the number of steps grows, while sampler B
    (bridge re-noising from z, the generalisation of DIFUSCO's renoise) keeps an O(1) bias (math_route1.md §5)."""
    inst = instance(1, 0.3)
    inst['n'] = 7
    X = states(7)
    gg = torch.Generator().manual_seed(11)
    idx = torch.randperm(len(X), generator=gg)[:4]
    inst.update(X=X, labels=X[idx], lab_idx=idx.tolist(), M=4, pi1=torch.full((4,), 0.25, dtype=torch.float64))
    log_g, _, _ = sb.sinkhorn(inst['labels'], inst['pi1'], inst['a_T'], X, tol=1e-14)
    A = [product_sampler_tv(inst, log_g, 'A', N) for N in (4, 16, 64)]
    B = [product_sampler_tv(inst, log_g, 'B', N) for N in (4, 16, 64)]
    print('product samplers TV(end, pi1): A', A, 'B', B)
    assert A[0] > A[1] > A[2] and A[2] < A[1] / 2.5          # roughly 1/N
    assert B[2] > 2 * A[2] and B[2] > B[1] / 2.5              # B decays much slower than 1/N


# ---------------------------------------------------------------- 6. deliberately wrong versions must fail

@pytest.mark.parametrize('p', (0.1, 0.3))
def test_wrong_versions_fail(p):
    worst = {'no_g': 0.0, 'wrong_time': 0.0, 'bridge_without_z': 0.0}
    for seed in range(N_INSTANCES):
        inst = instance(seed, p)
        log_g, w = exact_sinkhorn(inst)
        X = inst['X']
        end = torch.zeros(len(X), dtype=torch.float64); end[inst['lab_idx']] = inst['pi1']
        # (a) the posterior without g (prior pi1 instead of the Sinkhorn potential)
        bad = lambda X_, a_k: sb.label_log_posterior(X_, inst['labels'], torch.log(inst['pi1']), a_k).exp()
        worst['no_g'] = max(worst['no_g'], (ancestral_end(inst, log_g, bad, bridge_back(inst))[0] - end).abs().max().item())
        # (b) the posterior with the kernel of the wrong remaining time (a_T instead of a_k)
        bad = lambda X_, a_k: sb.label_log_posterior(X_, inst['labels'], log_g, inst['a_T']).exp()
        worst['wrong_time'] = max(worst['wrong_time'], (ancestral_end(inst, log_g, bad, bridge_back(inst))[0] - end).abs().max().item())
        # (c) bridge re-noising that ignores z (the plain forward marginal K_{a_k}(. | y)), with the right posterior
        good = lambda X_, a_k: sb.label_log_posterior(X_, inst['labels'], log_g, a_k).exp()
        def fwd_only(y, a_k):
            K = sb.log_kernel(y[None, :], X, a_k).exp()[0]
            return K[None, :].expand(len(X), -1)
        out = bridge_renoise_joint(inst, log_g, good, fwd_only)
        P_zm = w[:, None] * sb.coupling_conditional(X, inst['labels'], log_g, inst['a_T'])
        J = joint_z_xk_m(inst, P_zm, inst['a'][2].item()).sum(2)
        worst['bridge_without_z'] = max(worst['bridge_without_z'], (out[2] - J).abs().max().item())
    print('wrong-version errors', worst)
    for k, v in worst.items():
        assert v > 1e3 * TOL, (k, v)


def test_categorical_posterior_is_sampler_A():
    """solvers/difusco.py categorical_posterior, fed Q_bar = signal_to_Q_bar(a, as_numpy=True) of an SB schedule,
    gives the per-bit transition of sampler A (math_route1.md §5), including skips k -> j < k - 1.
    The method is extracted from the source because solvers/difusco.py imports packages absent from this env."""
    import types
    import numpy as np
    import torch.nn.functional as F
    src = open(os.path.join(os.path.dirname(__file__), '..', 'solvers', 'difusco.py')).read()
    start = src.index('    def categorical_posterior'); stop = src.index('    def tour2adj')
    ns = {'np': np, 'torch': torch, 'F': F}
    exec('\n'.join(l[4:] for l in src[start:stop].splitlines()), ns)
    cp = ns['categorical_posterior']
    a = sb.sb_signal_schedule(CategoricalDiffusion(T=50, schedule='cosine').Q_bar, 0.3)
    obj = types.SimpleNamespace(diffusion=types.SimpleNamespace(Q_bar=sb.signal_to_Q_bar(a, as_numpy=True)))
    g = torch.Generator().manual_seed(0)
    xt = torch.randint(0, 2, (2000, 1), generator=g).float()
    p1 = torch.rand(2000, 1, generator=g)
    x0p = torch.stack([1 - p1, p1], -1)
    orig = torch.bernoulli
    torch.bernoulli = lambda p: p                    # read the probability instead of a sample
    try:
        for t, tt in ((50, 49), (50, 40), (20, 10), (5, 1)):
            out = cp(obj, tt, t, x0p, xt).double()
            x = xt.double()
            ref = p1.double() * sb.bridge_prob1(torch.ones_like(x), x, a[tt].item(), a[t].item()) + \
                (1 - p1.double()) * sb.bridge_prob1(torch.zeros_like(x), x, a[tt].item(), a[t].item())
            assert (out - ref).abs().max() < 1e-5, (t, tt)      # float32 inside categorical_posterior
    finally:
        torch.bernoulli = orig


def test_gpu_style_sinkhorn_matches_reference():
    """sinkhorn_distances (float32 chunks on distances, the form used for real graphs) agrees with sinkhorn
    (float64 on the starts) on the same starts; schedule_for_omega hits the requested end strength."""
    diff = CategoricalDiffusion(T=50, schedule='cosine')
    for om in (0.35, 1.0):
        a = sb.schedule_for_omega(diff.Q_bar, om)
        assert abs(sb.omega(a[-1].item()) - om) < 1e-12 and a[0] == 1.0
    assert (sb.schedule_for_omega(diff.Q_bar, 0.0) - sb.sb_signal_schedule(diff.Q_bar, 0.0)).abs().max() == 0
    g = torch.Generator().manual_seed(3)
    labels = torch.randint(0, 2, (9, 60), generator=g)
    pi1 = torch.rand(9, generator=g, dtype=torch.float64) + 0.2; pi1 /= pi1.sum()
    for om in (0.3, 1.0):
        D = sb.sample_distances(5000, labels, torch.Generator().manual_seed(7), chunk=1024)
        z = torch.randint(0, 2, (5000, 60), generator=torch.Generator().manual_seed(7))
        # same stream only if generated in one call; regenerate D from z directly for the comparison
        Dz = (z.sum(1, keepdim=True) + labels.sum(1)[None, :] - 2 * z @ labels.T).to(torch.int16)
        lg32, rel, _ = sb.sinkhorn_distances(Dz, om, pi1, tol=1e-6, chunk=1024)
        lg64, err, _ = sb.sinkhorn(labels, pi1, math.tanh(om / 2), z, tol=1e-13)
        assert rel < 1e-6 and err < 1e-12
        assert ((lg32 - lg32.mean()) - (lg64 - lg64.mean())).abs().max() < 1e-4
        assert D.dtype == torch.int16 and D.shape == (5000, 9) and int(D.min()) >= 0 and int(D.max()) <= 60
