"""Paths between a start sample x0 and the label x1 for binary node variables (local addition, see
UPSTREAM.md item 43): discrete flow matching with an arbitrary start distribution, and bridge
matching with a uniform-flip reference process.

Each node is independent given its two endpoints a = x0_i and b = x1_i, t in [0, 1].

Reference process (bridge matching): every node flips at rate lam. Over a time s,
    P_s(flip) = (1 - exp(-2 lam s)) / 2,   P_s(stay) = 1 - P_s(flip).
The bridge is this process conditioned on x_0 = a, x_1 = b:
    P(x_t = c | a, b) = P_t(a -> c) P_{1-t}(c -> b) / P_1(a -> b).
lam -> 0 is the mixture path of discrete flow matching with kappa(t) = t (Gat et al., NeurIPS 2024):
a node with a != b switches once at a uniform time, a node with a == b never moves:
    x_t = x1 with probability t, else x0.

Sampling with a denoiser p1 = p_theta(x1_i = 1 | x_t, t). The generator of the marginal path is
the bridge rate averaged over the posterior of x1_i (exact because the reference factorises over
nodes, so the Doob ratio of node i only involves x1_i):
    rate_i = lam * [ q (1+e)/(1-e) + (1-q) (1-e)/(1+e) ],   e = exp(-2 lam (1-t)),
    q = P(x1_i != x_t_i) under p1.
For lam -> 0 this is q / (1 - t), the mixture-path rate (and the Euler step of solvers/fm.py).
Neither formula involves the start distribution, so any start (e.g. MCMC samples) is allowed.
"""
import math
import torch


def flip_prob_ref(lam, s):
    """P_s(flip) of the reference process; s may be a tensor."""
    if lam == 0:
        return torch.zeros_like(s)
    return (1 - torch.exp(-2 * lam * s)) / 2


def sample_path(x0, x1, t, lam=0.0):
    """x_t given endpoints x0, x1 (same shape, 0/1 floats) and t (broadcastable, in [0, 1])."""
    t = torch.as_tensor(t, dtype=x0.dtype, device=x0.device).expand_as(x0)
    if lam == 0:
        take_x1 = torch.rand_like(x0) < t
        return torch.where(take_x1, x1, x0)
    pf_a = flip_prob_ref(lam, t)            # a -> c over [0, t]
    pf_b = flip_prob_ref(lam, 1 - t)        # c -> b over [t, 1]
    same = (x0 == x1)
    # c = a: stay then (a -> b);  c = 1 - a: flip then (1-a -> b)
    w_stay = (1 - pf_a) * torch.where(same, 1 - pf_b, pf_b)
    w_flip = pf_a * torch.where(same, pf_b, 1 - pf_b)
    p_flip = w_flip / (w_stay + w_flip).clamp_min(1e-12)
    flip = torch.rand_like(x0) < p_flip
    return torch.where(flip, 1 - x0, x0)


def step_flip_prob(x, p1, t1, t2, lam=0.0):
    """Probability that each node of x_{t1} flips by t2, given the denoiser's p1 = P(x1 = 1 | x_{t1})."""
    q = torch.where(x > 0.5, 1 - p1, p1)    # P(x1 != current value)
    h = t2 - t1
    if lam == 0:
        return (q * h / max(1 - t1, 1e-12)).clamp(0, 1)
    e = math.exp(-2 * lam * (1 - t1))
    rate = lam * (q * (1 + e) / max(1 - e, 1e-12) + (1 - q) * (1 - e) / (1 + e))
    return (1 - torch.exp(-rate * h)).clamp(0, 1)
