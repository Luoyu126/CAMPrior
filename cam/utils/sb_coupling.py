"""Static and dynamic Schrödinger bridge between Bernoulli(0.5)^n and a finite label set (route 1 of
docs/20261125/plan.md; derivations in docs/20261125/math_route1.md, local addition, UPSTREAM.md item 67).

Conventions (the DIFUSCO index convention of solvers/difusco.py):
    y   the data end, one of the labels s_1..s_M of a graph, with target weights pi1 (sum 1)
    z   the noise end, z ~ Bernoulli(0.5)^n
    x_k the state at diffusion index k = 0..T; x_0 = y, x_T = z
The reference process flips every bit independently. Between the data end and index k it is the binary
symmetric channel with signal a_k (P(x_k,i != y_i) = (1 - a_k) / 2), a_0 = 1 > a_1 > ... > a_T >= 0; between
index k and the noise end the signal is a_T / a_k. a_T > 0 is a reference that does not fully mix (not
memoryless); a_T = 0 is the usual diffusion, for which the bridge reduces to the independent coupling.

Everything below follows from (math_route1.md, sections 2-5):
    pi*(y = s_m | z)   ∝ g_m K_{a_T}(s_m | z)                   (static SB, Sinkhorn for g)
    P*(y = s_m | x_k)  ∝ g_m K_{a_k}(s_m | x_k)                 (Markov property of the SB)
    x_k | (z, y)       per-bit reference bridge                 (closed form, bridge_prob1)
with K_a(s | x) = prod_i ((1 + a)/2)^[s_i = x_i] ((1 - a)/2)^[s_i != x_i].
"""
import math

import torch


def log_kernel(x, labels, a):
    """log K_a(s_m | x) for all rows of x (K, n) and labels (M, n), both in {0, 1}; returns (K, M).
    a = 1 gives -inf for every disagreeing pair; a = 0 gives the constant -n log 2."""
    x = x.to(torch.float64); s = labels.to(torch.float64)
    n = x.shape[1]
    d = x.sum(1, keepdim=True) + s.sum(1).unsqueeze(0) - 2 * x @ s.T          # Hamming distances (K, M)
    a = float(a)
    log_same = math.log((1 + a) / 2)
    log_diff = math.log((1 - a) / 2) if a < 1 else -math.inf
    out = (n - d) * log_same
    if a < 1:
        return out + d * log_diff
    return torch.where(d > 0, torch.full_like(out, -math.inf), out)


def label_log_posterior(x, labels, log_g, a):
    """log P*(y = s_m | x_k = x) = log_softmax_m(log g_m + log K_{a_k}(s_m | x)); x (K, n) -> (K, M)."""
    LK = log_kernel(x, labels, a)
    return torch.log_softmax(log_g.to(device=LK.device, dtype=torch.float64).unsqueeze(0) + LK, dim=1)


def posterior_mean(x, labels, log_g, a):
    """E[y | x_k = x] = sum_m P*(m | x) s_m, per node; x (K, n) -> (K, n), float64."""
    w = label_log_posterior(x, labels, log_g, a).exp()
    return w @ labels.to(device=w.device, dtype=torch.float64)


def sinkhorn(labels, pi1, a_T, z, z_weights=None, iters=10000, tol=1e-12):
    """Solve for log g on the label side so that the coupling
        pi(z, m) = w(z) softmax_m(log g_m + log K_{a_T}(s_m | z))
    has label marginal pi1. z (K, n) are states of the noise end with weights w (default uniform 1/K):
    all 2^n states with weights 2^-n give the exact solution, K uniform samples the Monte Carlo one.
    The z marginal holds by construction (the softmax normalises each row), so only g is iterated.
    Returns (log_g, max_abs_marginal_error, iterations). log_g is defined up to an additive constant;
    it is normalised to logsumexp 0."""
    assert 0.0 <= float(a_T) < 1.0
    LK = log_kernel(z, labels, a_T)
    pi1 = pi1.to(device=LK.device, dtype=torch.float64)
    assert bool((pi1 > 0).all()) and abs(pi1.sum().item() - 1) < 1e-9
    w = torch.full((z.shape[0],), 1.0 / z.shape[0], dtype=torch.float64, device=LK.device) \
        if z_weights is None else z_weights.to(device=LK.device, dtype=torch.float64)
    assert abs(w.sum().item() - 1) < 1e-9
    log_pi1 = torch.log(pi1)
    log_g = log_pi1.clone()
    err = math.inf
    for it in range(1, iters + 1):
        P = torch.softmax(log_g.unsqueeze(0) + LK, dim=1)
        marg = (w.unsqueeze(1) * P).sum(0)
        err = (marg - pi1).abs().max().item()
        if err < tol:
            break
        log_g = log_g + log_pi1 - torch.log(marg)
        log_g = log_g - torch.logsumexp(log_g, 0)
    else:   # no break: report the error of the returned log_g
        marg = (w.unsqueeze(1) * torch.softmax(log_g.unsqueeze(0) + LK, dim=1)).sum(0)
        err = (marg - pi1).abs().max().item()
    return log_g, err, it


def coupling_conditional(z, labels, log_g, a_T):
    """pi*(m | z) as probabilities (K, M)."""
    return label_log_posterior(z, labels, log_g, a_T).exp()


def bridge_prob1(y, z, a_k, a_T):
    """P(x_k,i = 1 | y_i, z_i) under the reference bridge, per bit (same shape as y and z).

    P(x_k = b | y, z) ∝ K_{a_k}(b | y) K_{a_T / a_k}(z | b). At a_k = a_T (k = T) it is z itself;
    with a_T = 0 the second factor is constant and the bridge is the forward marginal K_{a_k}(. | y)."""
    y = y.to(torch.float64); z = z.to(torch.float64)
    a_k = float(a_k); a_T = float(a_T)
    assert 0.0 <= a_T <= a_k <= 1.0, (a_T, a_k)
    if a_k == a_T:          # the noise end itself (also at a_T = 0, where any law of x_T given y is uniform)
        return z.clone()
    r = a_T / a_k
    def ch(b, c, a):          # K_a(b | c) elementwise
        same = (b == c).to(torch.float64)
        return same * (1 + a) / 2 + (1 - same) * (1 - a) / 2
    one = torch.ones_like(y); zero = torch.zeros_like(y)
    p1 = ch(one, y, a_k) * ch(z, one, r)
    p0 = ch(zero, y, a_k) * ch(z, zero, r)
    return p1 / (p1 + p0)


def sb_signal_schedule(Q_bar, a_T):
    """Signals a_0..a_T of the SB reference built from an existing DIFUSCO schedule:
        a_k = a_T + (1 - a_T) a~_k   for every k = 1..T, a_0 = 1,
    where a~_k is the signal of Q_bar[k] (Q_bar[k][0,0] - Q_bar[k][0,1]). The endpoint signal used everywhere
    (Sinkhorn, bridge, omega) is a[T] = a_T + (1 - a_T) a~_T, slightly above the parameter a_T because the cosine
    table does not reach 0 exactly (a~_T is about 1e-6 at T = 50). a_T = 0 reproduces Q_bar at every k.
    Returns a float64 tensor of length T + 1."""
    assert 0.0 <= a_T < 1.0
    Qb = torch.as_tensor(Q_bar, dtype=torch.float64)
    base = Qb[:, 0, 0] - Qb[:, 0, 1]
    a = a_T + (1 - a_T) * base
    a[0] = 1.0
    assert bool((a[1:] <= a[:-1]).all()), 'the signal schedule must be non-increasing'
    return a


def signal_to_Q_bar(a, as_numpy=False):
    """2x2 transition matrices Q_bar[k][y, x] = K_{a_k}(x | y) for a signal schedule a (T + 1,).
    as_numpy=True gives the numpy array that CategoricalDiffusion.Q_bar holds (solvers/difusco.py uses
    torch.from_numpy on it)."""
    a = torch.as_tensor(a, dtype=torch.float64)
    same = (1 + a) / 2; diff = (1 - a) / 2
    Qb = torch.stack([torch.stack([same, diff], -1), torch.stack([diff, same], -1)], -2)
    return Qb.numpy() if as_numpy else Qb


def omega(a_T):
    """Strength of the Hamming cost of the static SB, log((1 - p) / p) with p = (1 - a_T) / 2."""
    return math.log((1 + a_T) / (1 - a_T))


def effective_labels(P):
    """Effective number of labels per row of a probability matrix (K, M): 1 / sum_m P_m^2."""
    return 1.0 / (P ** 2).sum(1)
