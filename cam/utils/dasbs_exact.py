"""Exact (enumerated) discrete Schrödinger bridge tools for route 2 of docs/20261125/plan.md (DASBS, Guo et al.,
arXiv 2602.08243), specialised to binary states. For checking derivations on small graphs (n <= 9) only; the
training losses live in solvers/. Derivations: docs/20261125/math_route2.md. Local addition, UPSTREAM.md item 68.

Conventions follow DASBS (not the DIFUSCO index of utils/sb_coupling.py):
    time t in [0, 1]; source mu at t = 0 (here Bernoulli(0.5)^n), target nu at t = 1.
    Reference: every bit flips at rate gamma_t / 2 (DASBS eq. 16 with N = 2), Gamma(s, t) = int_s^t gamma, so the
    reference kernel over [s, t] is the binary symmetric channel with signal exp(-Gamma(s, t)).
    SB potentials: pi*(x0, x1) = phihat_0(x0) K_{0,1}(x0, x1) phi_1(x1); phi_t = K_{t,1} phi_1,
    phihat_t = K_{0,t}^T phihat_0; P*_t = phi_t * phihat_t; nu = phi_1 * phihat_1 (DASBS eq. 11-13).
    Controller Phi_t(x)_i = phi_t(x^i) / phi_t(x); corrector Phihat(y)_i = phihat_1(y^i) / phihat_1(y),
    where x^i is x with bit i flipped.
States are enumerated in itertools.product order; bit i has weight 2^(n-1-i).
"""
import itertools
import math

import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp


# ------------------------------------------------------------------ states and reference

def states(n):
    assert 1 <= n <= 12
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)


def neighbours(n):
    """nb[s, i] = index of state s with bit i flipped."""
    S = 2 ** n
    return np.arange(S)[:, None] ^ (1 << (n - 1 - np.arange(n)))[None, :]


class Reference(object):
    """Uniform flip reference: per-bit flip rate gamma_t / 2. schedule 'constant' (gamma_t = gamma) or
    'loglinear' (gamma_t = gamma / (t + alpha), DASBS App. B.5; alpha -> 0 is memoryless)."""

    def __init__(self, schedule='constant', gamma=1.0, alpha=1.0):
        assert schedule in ('constant', 'loglinear')
        self.schedule, self.gamma, self.alpha = schedule, float(gamma), float(alpha)

    def rate(self, t):
        return self.gamma if self.schedule == 'constant' else self.gamma / (t + self.alpha)

    def Gamma(self, s, t):
        assert 0 <= s <= t <= 1
        if self.schedule == 'constant':
            return self.gamma * (t - s)
        if s + self.alpha == 0:          # memoryless log-linear (alpha = 0): infinite noise on [0, t], t > 0
            return math.inf if t > 0 else 0.0
        return self.gamma * math.log((t + self.alpha) / (s + self.alpha))

    def signal(self, s, t):
        return math.exp(-self.Gamma(s, t))

    def kernel(self, X, s, t):
        """K[x, y] = P(x_t = y | x_s = x) over all states X (S, n)."""
        return channel(X, X, self.signal(s, t))


def channel(X, Y, a):
    """K_a(y | x) for rows x of X and y of Y: per-bit binary symmetric channel with signal a."""
    d = (X[:, None, :] != Y[None, :, :]).sum(-1)
    n = X.shape[1]
    same, diff = (1 + a) / 2, (1 - a) / 2
    return same ** (n - d) * diff ** d


def unit_generator(n):
    """L0 with rate 1/2 for every single-bit flip; the reference generator is gamma_t * L0."""
    S = 2 ** n
    L = np.zeros((S, S))
    nb = neighbours(n)
    for i in range(n):
        L[np.arange(S), nb[:, i]] = 0.5
    L[np.arange(S), np.arange(S)] = -n / 2
    return L


# ------------------------------------------------------------------ MIS energy and target

def mis_energy(X, A, lam):
    """H(x) = -sum_i x_i + lam * sum_{i<j} A_ij x_i x_j."""
    X = X.astype(np.float64)
    return -X.sum(1) + lam * 0.5 * np.einsum('si,ij,sj->s', X, A, X)


def mis_flip_delta(X, A, lam):
    """H(x^i) - H(x) = (1 - 2 x_i) (-1 + lam (A x)_i); (S, n)."""
    X = X.astype(np.float64)
    return (1 - 2 * X) * (-1 + lam * X @ A)


def boltzmann(X, A, lam, tau):
    logp = -mis_energy(X, A, lam) / tau
    return np.exp(logp - logsumexp(logp))


def random_graph(n, p, rng):
    A = np.triu(rng.random((n, n)) < p, 1).astype(np.float64)
    return A + A.T


# ------------------------------------------------------------------ exact SB

class ExactSB(object):
    def __init__(self, mu, nu, ref, n, tol=1e-14, iters=200000):
        self.n, self.ref = n, ref
        self.X = states(n)
        self.nb = neighbours(n)
        self.mu, self.nu = np.asarray(mu, float), np.asarray(nu, float)
        K = ref.kernel(self.X, 0.0, 1.0)
        logK = np.log(K)
        lu, lv = np.zeros(len(K)), np.zeros(len(K))
        lmu = np.log(self.mu)
        lnu = np.where(self.nu > 0, np.log(np.maximum(self.nu, 1e-300)), -np.inf)
        for it in range(iters):
            lu = lmu - logsumexp(logK + lv[None, :], axis=1)
            lv_new = lnu - logsumexp(logK + lu[:, None], axis=0)
            J = np.exp(lu[:, None] + logK + lv_new[None, :])
            err = max(np.abs(J.sum(1) - self.mu).max(), np.abs(J.sum(0) - self.nu).max())
            lv = lv_new
            if err < tol:
                break
        self.iterations, self.marginal_error = it + 1, err
        self.log_phihat0, self.log_phi1 = lu, lv          # pi* = phihat_0 K phi_1
        self.joint = np.exp(lu[:, None] + logK + lv[None, :])

    def phi(self, t):
        return self.ref.kernel(self.X, t, 1.0) @ np.exp(self.log_phi1)

    def phihat(self, t):
        return self.ref.kernel(self.X, 0.0, t).T @ np.exp(self.log_phihat0)

    def marginal(self, t):
        return self.phi(t) * self.phihat(t)

    def controller(self, t):
        ph = self.phi(t)
        return ph[self.nb] / ph[:, None]

    def corrector(self):
        ph = self.phihat(1.0)
        return ph[self.nb] / ph[:, None]

    def controlled_generator(self, t, Phi=None):
        Phi = self.controller(t) if Phi is None else Phi
        return generator_from_controller(Phi, self.nb, self.ref.rate(t))


def generator_from_controller(Phi, nb, rate):
    S, n = Phi.shape
    U = np.zeros((S, S))
    for i in range(n):
        U[np.arange(S), nb[:, i]] += rate / 2 * Phi[:, i]
    U[np.arange(S), np.arange(S)] = -U.sum(1)
    return U


# ------------------------------------------------------------------ matching maps (tabular "networks")

def joint_xt_x1(P01, ref, X, t):
    """p(x_t, x1) when (x0, x1) ~ P01 and x_t ~ reference bridge: sum_x0 P01 K_{0,t}(x0, x) K_{t,1}(x, x1) / K_{0,1}(x0, x1)."""
    K0t, Kt1, K01 = ref.kernel(X, 0.0, t), ref.kernel(X, t, 1.0), ref.kernel(X, 0.0, 1.0)
    W = P01 / K01                                       # (x0, x1)
    return np.einsum('ab,ax,xb->xb', W, K0t, Kt1)       # (x_t, x1)


def ctrl_am_map(P01, terminal_ratio, ref, X, t):
    """Tabular (ctrl-AM), DASBS eq. 20: Phi_t(x)_i = E[ terminal_ratio(x1)_i | x_t = x ] under P01 + reference bridge.
    terminal_ratio (S, n) = phi_1(x1^i) / phi_1(x1) built from the current corrector."""
    J = joint_xt_x1(P01, ref, X, t)
    px = J.sum(1, keepdims=True)
    return (J / px) @ terminal_ratio


def ctrl_am_map_log(P01, log_terminal_ratio, ref, X, t):
    """Wrong version (negative control): regress the log-ratio, Phi = exp(E[log ratio | x_t])."""
    J = joint_xt_x1(P01, ref, X, t)
    return np.exp((J / J.sum(1, keepdims=True)) @ log_terminal_ratio)


def corr_dm_map(P01, ref, X, nb, t=0.0):
    """Tabular (corr-DM), DASBS eq. 25: Phihat(y)_i = E[ K_{t,1}(x_t, y^i) / K_{t,1}(x_t, y) | x1 = y ], t in [0, 1)."""
    J = joint_xt_x1(P01, ref, X, t)                     # (x_t, y)
    Kt1 = ref.kernel(X, t, 1.0)
    py = J.sum(0)
    out = np.zeros(nb.shape)
    for i in range(nb.shape[1]):
        out[:, i] = (J * Kt1[:, nb[:, i]] / Kt1).sum(0) / np.where(py > 0, py, 1.0)
    return out


def ratios_to_log_potential(R, nb, support=None):
    """Integrate single-flip ratios R[s, i] = h(s^i) / h(s) into log h (up to a constant) along a spanning tree
    from state 0; returns (log h, max inconsistency over all edges) to detect non-gradient ratio fields."""
    S, n = R.shape
    logh = np.full(S, np.nan); logh[0] = 0.0
    order = [0]
    for s in order:
        for i in range(n):
            t = nb[s, i]
            if np.isnan(logh[t]):
                logh[t] = logh[s] + np.log(R[s, i]); order.append(t)
    incons = np.abs(logh[nb] - logh[:, None] - np.log(R)).max()
    return logh, incons


def forward_half_bridge(mu, nu, log_phihat1, ref, X):
    """Fixed point of (ctrl-AM) for a frozen corrector phihat_1 (DASBS Thm 5.2 (1)): the h-transform of the reference
    from mu with terminal function phi_1 = nu / phihat_1. Returns (P01, log phi_1)."""
    log_phi1 = np.where(nu > 0, np.log(np.maximum(nu, 1e-300)), -np.inf) - log_phihat1
    K01 = ref.kernel(X, 0.0, 1.0)
    phi1 = np.exp(log_phi1 - np.max(log_phi1[np.isfinite(log_phi1)]))
    phi0 = K01 @ phi1
    return mu[:, None] * K01 * phi1[None, :] / phi0[:, None], log_phi1


def controller_from_phi1(log_phi1, ref, X, nb, t):
    phi1 = np.exp(log_phi1 - np.max(log_phi1[np.isfinite(log_phi1)]))
    ph = ref.kernel(X, t, 1.0) @ phi1
    return ph[nb] / ph[:, None]


def ipf_alternation(mu, nu, ref, n, stages, dm_time=0.0, log_ratio_corrector=False):
    """Exact alternation of the two matching steps (DASBS sec. 4, 'Alternating Update'), each solved exactly:
    controller = fixed point of (ctrl-AM) for the frozen corrector; corrector = (corr-DM) under that controller.
    Starts from the all-one corrector. Returns the terminal marginals after each stage."""
    X, nb = states(n), neighbours(n)
    log_phihat1 = np.zeros(2 ** n)
    out = []
    for _ in range(stages):
        P01, _ = forward_half_bridge(mu, nu, log_phihat1, ref, X)
        out.append(P01.sum(0))
        R = corr_dm_map(P01, ref, X, nb, dm_time)
        if log_ratio_corrector:      # wrong version: geometric mean of the ratio labels
            J = joint_xt_x1(P01, ref, X, dm_time); Kt1 = ref.kernel(X, dm_time, 1.0)
            py = J.sum(0)
            for i in range(n):
                R[:, i] = np.exp((J * np.log(Kt1[:, nb[:, i]] / Kt1)).sum(0) / py)
        log_phihat1, _ = ratios_to_log_potential(R, nb)
    return out


def bridge_prob1(ref, x0, x1, t):
    """Per-bit reference bridge (DASBS Prop. B.6, N = 2): P(X_t,i = 1 | X_0 = x0, X_1 = x1) for 0 < t < 1, bits
    independent given both ends. P(b | x0_i, x1_i) ∝ K_{0,t}(x0_i, b) K_{t,1}(b, x1_i), one-bit channels."""
    a0, a1 = ref.signal(0.0, t), ref.signal(t, 1.0)
    x0 = np.asarray(x0); x1 = np.asarray(x1)
    def ch(b, c, a):
        return np.where(b == c, (1 + a) / 2, (1 - a) / 2)
    w1 = ch(x0, 1, a0) * ch(1, x1, a1)
    w0 = ch(x0, 0, a0) * ch(0, x1, a1)
    return w1 / (w0 + w1)


# ------------------------------------------------------------------ simulation of a tabular controller

def evolve_piecewise(mu, Phis, grid, ref, nb):
    """Exact P(x0, x1) of the CTMC whose controller is Phis[j] (S, n) on [grid[j], grid[j+1]] (rates
    gamma_t / 2 * Phis[j], the time profile of gamma integrated exactly). Returns P01 (S, S)."""
    S = len(mu)
    M = np.eye(S)
    for j in range(len(grid) - 1):
        G = generator_from_controller(Phis[j], nb, 1.0) * ref.Gamma(grid[j], grid[j + 1])
        M = M @ expm(G)
    return mu[:, None] * M


def tau_leaping_marginal(mu, controller_fn, ref, nb, steps, mode='linear', grid=None):
    """Exact distribution of DASBS's tau-leaping sampler (App. B.6, Prop. B.7) with N = 2: on [t, t + h] every bit i
    flips independently with probability Gamma(t, t + h) / 2 * Phi_t(x)_i (mode 'linear', clipped to [0, 1]).
    mode 'exp' uses 1 - exp(-Gamma(t, t + h) / 2 * Phi_t(x)_i) instead: the probability of at least one jump of a
    bit whose rate is frozen (not the net-flip probability r/(r + r')(1 - exp(-(r + r')h)) of the two-state chain);
    it agrees to first order and never exceeds 1 (local addition, not in DASBS). grid: optional increasing time
    points from 0 to 1 (steps + 1 of them) instead of the uniform grid. A step with Gamma = inf (the first step of the
    memoryless log-linear reference) is a uniform resample of every bit: the reference kernel itself, controller
    ignored. Returns (terminal marginal, max un-clipped linear flip probability seen on finite steps)."""
    assert mode in ('linear', 'exp')
    S, n = nb.shape
    X = states(n)
    p = mu.copy()
    worst = 0.0
    grid = np.linspace(0, 1, steps + 1) if grid is None else np.asarray(grid, float)
    assert len(grid) == steps + 1 and grid[0] == 0 and abs(grid[-1] - 1) < 1e-12
    for j in range(steps):
        G = ref.Gamma(grid[j], grid[j + 1])
        if math.isinf(G):
            q = np.full((S, n), 0.5)
        else:
            lin = G / 2 * controller_fn(grid[j])                                 # (S, n)
            worst = max(worst, lin.max())
            q = np.clip(lin, 0, 1) if mode == 'linear' else -np.expm1(-lin)
        diff = (X[:, None, :] != X[None, :, :])                               # (x, y, i)
        T = np.prod(np.where(diff, q[:, None, :], 1 - q[:, None, :]), axis=2)
        p = p @ T
    return p, worst


def tv(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum()
