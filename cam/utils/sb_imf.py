"""Exact (tabular) discrete-time Iterative Markovian Fitting for the route-1 Schrödinger bridge (stage 1D of
docs/20261125/plan.md; CSBM, Ksenofontov & Korotin 2025, arXiv 2502.01416, eq. 5-6 and §4.1; DDSBM, Kim et al. 2025).

Setting of math_route1.md: labels y_1..y_M in {0,1}^n with weights pi1, start z ~ pi0 = Bernoulli(0.5)^n, discrete
time k = 0..T with signals 1 = a_0 >= a_1 >= ... >= a_T > 0. Reference (Markov, per bit): X_0 = y, X_k | X_{k-1} ~
K_{a_k / a_{k-1}}, so X_k | y ~ K_{a_k}( . | y) and z = X_T. A coupling q(z, m) on {0,1}^n x labels is lifted to a path
law by the reference bridge (reciprocal projection) and projected on Markov chains (Markov projection), in either
direction:
  gen  (z -> label, the sampling direction): transitions P(x_{k-1} | x_k) of the path law, new coupling
       pi0(z) [P_T ... P_1](z, m);
  fwd  (label -> z): transitions P(x_k | x_{k-1}), new coupling pi1(m) [P_1 ... P_T](m, z).
Every state of {0,1}^n is enumerated (n <= ~9), float64. Pair marginals of the path law:
  J_k(x_{k-1}, x_k) = R_k(x_{k-1}, x_k) * sum_m F_{k-1}(x_{k-1}, m) G_k(x_k, m),
  R_k = K_{a_k/a_{k-1}}, F_{k-1}(x, m) = K_{a_{k-1}}(x | y_m), G_k(x, m) = sum_z K_{a_T/a_k}(z | x) q(z, m) / K_{a_T}(z | y_m).
The fixed point of the iteration (Markov and reciprocal with marginals pi0, pi1) is the static SB coupling
pi*(z, m) ∝ f(z) g_m K_{a_T}(z | y_m) of utils/sb_coupling.sinkhorn (math_route1.md §2; CSBM Thm 3.1).
"""
import math

import torch

from . import sb_coupling as sb


def all_states(n):
    """(2^n, n) float64, row j = binary digits of j, least significant bit first."""
    return ((torch.arange(2 ** n)[:, None] >> torch.arange(n)) & 1).to(torch.float64)


def kernel_matrix(X, Y, a):
    """K_a(Y_j | X_i) for all rows: (|X|, |Y|) float64 (per bit (1 + a)/2 if equal, (1 - a)/2 if not)."""
    return sb.log_kernel(Y, X, a).T.exp()


class TabularIMF(object):
    def __init__(self, labels, pi1, a, bridge=True):
        """labels (M, n) in {0,1}; pi1 (M,); a: signals a_0 = 1 >= ... >= a_T > 0 (T + 1,).
        bridge=False replaces the reference bridge by the forward marginals (x_k | y) only, ignoring z: a deliberately
        wrong reciprocal projection (negative control), whose iteration does not reach the SB."""
        self.Y = labels.to(torch.float64)
        self.M, self.n = self.Y.shape
        self.pi1 = pi1.to(torch.float64) / pi1.sum()
        self.a = [float(v) for v in a]
        assert abs(self.a[0] - 1) < 1e-15 and all(x >= y for x, y in zip(self.a, self.a[1:])) and self.a[-1] > 0
        self.T = len(self.a) - 1
        self.X = all_states(self.n)
        S = len(self.X)
        self.pi0 = torch.full((S,), 1.0 / S, dtype=torch.float64)
        self.R = [None] + [kernel_matrix(self.X, self.X, self.a[k] / self.a[k - 1]) for k in range(1, self.T + 1)]
        self.F = [kernel_matrix(self.Y, self.X, self.a[k]).T for k in range(self.T + 1)]        # (S, M): K_{a_k}(x|y_m)
        self.R1 = kernel_matrix(self.Y, self.X, self.a[1])                                       # (M, S): K(x_1 | y_m)
        self.W = [kernel_matrix(self.X, self.X, self.a[-1] / self.a[k]) for k in range(self.T + 1)]  # K(z | x_k)
        self.KT = kernel_matrix(self.Y, self.X, self.a[-1]).T                                     # (S, M): K_{a_T}(z|y)
        self.bridge = bridge

    def independent(self):
        return self.pi0[:, None] * self.pi1[None, :]

    def pair_marginals(self, q):
        """J_k for k = 1..T: J_1 (M, S) over (label, x_1), J_k (S, S) over (x_{k-1}, x_k) for k >= 2."""
        h = q / self.KT                                     # (S, M)
        J = [None]
        for k in range(1, self.T + 1):
            if self.bridge:
                G = self.W[k] @ h                           # (S, M): sum_z K(z | x_k) h(z, m)
            else:                                           # z ignored: x_k | y from the forward marginal only
                G = torch.ones_like(h) * q.sum(0)[None, :]
            if k == 1:
                J.append(self.R1 * G.T)                     # (M, S)
            else:
                J.append(self.R[k] * (self.F[k - 1] @ G.T))
        return J

    def gen(self, q):
        """Markov projection in the sampling direction: pi0(z) [P_T ... P_1](z, m)."""
        J = self.pair_marginals(q)
        v = torch.diag(self.pi0)                            # rows: start z, columns: current state x_k
        for k in range(self.T, 0, -1):
            P = J[k] / J[k].sum(0, keepdim=True)            # P(x_{k-1} | x_k), columns sum to 1
            v = v @ P.T
        return v                                            # (S, M)

    def fwd(self, q):
        """Markov projection in the noising direction: pi1(m) [P_1 ... P_T](m, z), returned as (S, M)."""
        J = self.pair_marginals(q)
        v = torch.diag(self.pi1)                            # rows: label m
        for k in range(1, self.T + 1):
            P = J[k] / J[k].sum(1, keepdim=True)            # P(x_k | x_{k-1}), rows sum to 1
            v = v @ P
        return v.T

    def sb_coupling(self, iters=100000, tol=1e-14):
        """Exact static SB coupling pi*(z, m) (Sinkhorn over all 2^n starts)."""
        log_g, err, _ = sb.sinkhorn(self.Y, self.pi1, self.a[-1], self.X, self.pi0, iters=iters, tol=tol)
        P = sb.coupling_conditional(self.X, self.Y, log_g, self.a[-1])
        return self.pi0[:, None] * P

    def run(self, iters, order='alternate', q0=None):
        """D-IMF from q0 (default independent): order 'alternate' (gen, fwd, gen, ...) or 'gen' only. Returns the list
        of couplings after each projection."""
        q = self.independent() if q0 is None else q0
        out = []
        for l in range(iters):
            q = self.gen(q) if (order == 'gen' or l % 2 == 0) else self.fwd(q)
            out.append(q)
        return out


def tv(p, q):
    return 0.5 * float((p - q).abs().sum())


def kl(p, q):
    m = p > 0
    return float((p[m] * (p[m].log() - q[m].log())).sum())


def mean_hamming(q, X, Y):
    D = X.sum(1, keepdim=True) + Y.sum(1)[None, :] - 2 * X @ Y.T
    return float((q * D).sum())
