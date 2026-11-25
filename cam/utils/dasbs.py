"""Single-graph discrete adjoint Schrödinger bridge sampler (DASBS, Guo et al., arXiv 2602.08243) for binary MIS:
the parts that do not depend on the network. Stage 2A of docs/20261125/plan.md; derivations in
docs/20261125/math_route2.md. Local addition, UPSTREAM.md item 70.

Ported from the DASBS worktree used on 2026-11-15 (`sb_worktrees/DASBS/mis_sb`: graph.py, reference.py, losses.py,
dasbs.py, sampling.py, exact.py, metrics.py). With the default arguments every function computes exactly what the
worktree computes (same operations in the same order), so a run reproduces the 11-15 numbers for the same seed and
torch build. Additions, all off by default:
  - corrector_loss(..., time='zero'): corr-DM at t = 0 only (math_route2.md §5: the identity holds at every t, and the
    variance of the kernel ratio grows as t -> 1).
  - sample_paths(..., grid='quadratic'): outer grid t = 1 - (1 - s)^2, denser near t = 1 (math_route2.md §10).

Conventions (DASBS, as in utils/dasbs_exact.py): time t in [0, 1], source Bernoulli(p)^n at t = 0, target
nu ∝ exp(-H / temperature) at t = 1; every bit flips at rate gamma_t / 2; controller output = log Phi_t(x)_i =
log phi_t(x^i) / phi_t(x); corrector output = log Phihat(y)_i = log phihat_1(y^i) / phihat_1(y).
States in exact enumeration are least-significant-bit first: state index = sum_i x_i 2^i (unlike dasbs_exact).
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import logsumexp


# ------------------------------------------------------------------ graph and MIS energy

class MISGraph(object):
    """Undirected graph with the penalised MIS energy H(x) = -|x| + penalty * (number of violated edges)."""

    def __init__(self, adjacency, penalty=2.0):
        a = torch.as_tensor(adjacency, dtype=torch.float32)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
            raise ValueError('adjacency must be a nonempty square matrix')
        if not torch.equal(a, a.T) or not torch.all((a == 0) | (a == 1)):
            raise ValueError('adjacency must be symmetric and binary')
        if torch.any(a.diag() != 0):
            raise ValueError('self loops are not permitted')
        if not np.isfinite(penalty) or penalty <= 1:
            raise ValueError('MIS penalty must be finite and greater than one')
        self.adjacency, self.penalty = a, penalty

    @property
    def n(self):
        return self.adjacency.shape[0]

    def to(self, device):
        return MISGraph(self.adjacency.to(device), self.penalty)

    def _adj(self, x):
        return self.adjacency.to(device=x.device, dtype=x.dtype)

    def energy(self, x):
        x = x.float() if not x.is_floating_point() else x
        return -x.sum(-1) + self.penalty / 2 * (x * (x @ self._adj(x))).sum(-1)

    def flip_delta(self, x):
        """Exact H(x with bit i flipped) - H(x) for every i."""
        x = x.float() if not x.is_floating_point() else x
        return (1 - 2 * x) * (-1 + self.penalty * (x @ self._adj(x)))

    def violations(self, x):
        x = x.float()
        return (x * (x @ self._adj(x))).sum(-1) / 2

    def decode(self, scores):
        """Greedy maximal independent set by descending score (ties: lower index first)."""
        shape = scores.shape
        values = scores.detach().reshape(-1, self.n).cpu()
        a = self.adjacency.cpu().bool()
        out = torch.zeros_like(values)
        for k, row in enumerate(values):
            available = torch.ones(self.n, dtype=torch.bool)
            for i in torch.argsort(row, descending=True, stable=True):
                if available[i]:
                    out[k, i] = 1
                    available[i] = False
                    available[a[i]] = False
        return out.reshape(shape).to(scores.device)

    @classmethod
    def from_edges(cls, n, edges, penalty=2.0):
        a = torch.zeros(n, n)
        for i, j in edges:
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError('edge outside node range')
            a[i, j] = a[j, i] = 1
        return cls(a, penalty)

    @classmethod
    def path(cls, n, penalty=2.0):
        return cls.from_edges(n, [(i, i + 1) for i in range(n - 1)], penalty)

    @classmethod
    def cycle(cls, n, penalty=2.0):
        return cls.from_edges(n, [(i, (i + 1) % n) for i in range(n)], penalty)

    @classmethod
    def erdos_renyi(cls, n, p, seed=0, penalty=2.0):
        rng = np.random.default_rng(seed)
        a = np.triu(rng.random((n, n)) < p, 1)
        return cls(torch.from_numpy(a | a.T), penalty)


# ------------------------------------------------------------------ reference process

@dataclass(frozen=True)
class UniformFlipReference(object):
    """Every bit flips at rate gamma_t / 2. 'constant': gamma_t = gamma; 'log_linear': gamma_t = gamma / (t + alpha)
    (DASBS App. B.5; alpha = 0 is memoryless: Gamma(0, t) = inf, every bit uniform at any t > 0). Gamma(s, t) = int_s^t gamma; the kernel over [s, t] flips each bit w.p. (1 - e^-Gamma) / 2."""
    schedule: str = 'log_linear'
    gamma: float = 1.0
    alpha: float = 0.5

    def __post_init__(self):
        if self.schedule not in ('constant', 'log_linear'):
            raise ValueError('unknown noise schedule')
        if not math.isfinite(self.gamma) or self.gamma <= 0 or not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('gamma must be positive and alpha non-negative, both finite')
        if self.alpha == 0 and self.schedule != 'log_linear':
            raise ValueError('alpha = 0 (memoryless) needs the log-linear schedule')

    def integral(self, s, t):
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        if torch.any(s < 0) or torch.any(t > 1) or torch.any(t < s):
            raise ValueError('expected 0 <= s <= t <= 1')
        if self.schedule == 'constant':
            return self.gamma * (t - s)
        return self.gamma * torch.log1p((t - s) / (s + self.alpha))

    def rate(self, t):
        """Per-bit flip rate gamma_t / 2."""
        t = torch.as_tensor(t)
        return torch.ones_like(t) * self.gamma / 2 if self.schedule == 'constant' else self.gamma / (2 * (t + self.alpha))

    def flip_probability(self, s, t):
        return -torch.expm1(-self.integral(s, t)) / 2

    @staticmethod
    def _column(v, x):
        v = torch.as_tensor(v, dtype=torch.float64, device=x.device)
        if v.ndim == x.ndim - 1:
            v = v.unsqueeze(-1)
        return v

    def log_kernel(self, x, y, s=0.0, t=1.0):
        p = self.flip_probability(self._column(s, x), self._column(t, x))
        return torch.where(x == y, torch.log1p(-p), torch.log(p)).sum(-1)

    def bridge_prob_one(self, x0, x1, t):
        """P(X_t,i = 1 | X_0 = x0, X_1 = x1) of the reference, per bit."""
        t = self._column(t, x0)
        p0 = self.flip_probability(torch.zeros_like(t), t)
        p1 = self.flip_probability(t, torch.ones_like(t))
        w0 = torch.where(x0 == 0, 1 - p0, p0) * torch.where(x1 == 0, 1 - p1, p1)
        w1 = torch.where(x0 == 1, 1 - p0, p0) * torch.where(x1 == 1, 1 - p1, p1)
        return w1 / (w0 + w1)

    def sample_bridge(self, x0, x1, t, generator=None):
        p = self.bridge_prob_one(x0, x1, t)
        return (torch.rand(p.shape, device=p.device, generator=generator) < p).long()

    def endpoint_flip_log_ratio(self, x, y, s=0.0, t=1.0):
        """log K_{s,t}(x, y^i) / K_{s,t}(x, y) per bit i (math_route2.md §5, binary closed form)."""
        p = self.flip_probability(self._column(s, x), self._column(t, x))
        if torch.any(p <= 0):
            raise ValueError('denoising ratios require a positive time interval')
        ratio = torch.log(p) - torch.log1p(-p)
        return torch.where(x == y, ratio, -ratio)


# ------------------------------------------------------------------ losses

def ratio_bregman(log_prediction, log_target):
    """Generalised KL D(a || b) with a = exp(target), b = exp(prediction), averaged. Its minimiser is the conditional
    mean of the RATIO a (math_route2.md §6); regressing log a instead gives exp(E log a), which is not the SB.
    Targets are detached; float64; non-finite values raise instead of being clipped."""
    p = log_prediction.double()
    t = log_target.detach().double()
    a, b = t.exp(), p.exp()
    value = a * (t - p) - a + b
    if not torch.isfinite(value).all():
        raise FloatingPointError('ratio overflow: increase temperature or inspect model logits')
    return value.mean()


def controller_loss(controller, corrector, rollout, graph, reference, temperature, replicas=8):
    """ctrl-AM (math_route2.md §4): regress Phi_t(X_t)_i on exp(-dH_i(X_1)/tau) / Phihat(X_1)_i with X_t from the
    reference bridge between the rollout's endpoints."""
    x0 = rollout.x0.repeat_interleave(replicas, 0)
    x1 = rollout.x1.repeat_interleave(replicas, 0)
    t = torch.rand(len(x0), device=x0.device)
    xt = reference.sample_bridge(x0, x1, t)
    with torch.no_grad():
        target = -graph.flip_delta(x1) / temperature - corrector(x1, None, graph)
    return ratio_bregman(controller(xt, t, graph), target)


def corrector_loss(corrector, rollout, graph, reference, replicas=8, epsilon=.01, time='uniform'):
    """corr-DM (math_route2.md §5): regress Phihat(X_1)_i on K_{t,1}(X_t, X_1^i) / K_{t,1}(X_t, X_1).
    time='uniform': t ~ U[0, 1 - epsilon] (the 11-15 runs); time='zero': t = 0, so X_t = X_0."""
    x0 = rollout.x0.repeat_interleave(replicas, 0)
    x1 = rollout.x1.repeat_interleave(replicas, 0)
    if time == 'uniform':
        t = torch.rand(len(x0), device=x0.device) * (1 - epsilon)
        xt = reference.sample_bridge(x0, x1, t)
    elif time == 'zero':
        t = torch.zeros(len(x0), device=x0.device)
        xt = x0
    else:
        raise ValueError(time)
    target = reference.endpoint_flip_log_ratio(xt, x1, t, torch.ones_like(t))
    return ratio_bregman(corrector(x1, None, graph), target)


# ------------------------------------------------------------------ sampling

@dataclass
class Rollout(object):
    x0: torch.Tensor
    x1: torch.Tensor
    times: torch.Tensor
    states: torch.Tensor
    substeps: int


def time_grid(steps, grid='uniform', device='cpu'):
    s = torch.linspace(0, 1, steps + 1, dtype=torch.float64, device=device)
    if grid == 'uniform':
        return s
    if grid == 'quadratic':      # t = 1 - (1 - s)^2: steps shrink linearly towards t = 1
        t = 1 - (1 - s) ** 2
        t[-1] = 1.0
        return t
    raise ValueError(grid)


@torch.no_grad()
def sample_paths(model, graph, reference, batch_size, steps=100, source_probability=0.5, max_hazard=0.1,
                 max_substeps=100000, device='cpu', generator=None, keep_path=True, grid='uniform'):
    """Controlled CTMC rollout. Within each outer step the rates are re-evaluated at every substep and frozen over it;
    substeps are short enough that every per-bit rate * dt <= max_hazard, and a bit flips over a substep with the
    exact probability (1 - exp(-2 rate dt)) / 2 of a frozen-rate two-state chain. No clamping.
    Model interface: (binary [B, n], time [B], graph) -> log rate ratio [B, n]."""
    if steps < 1 or batch_size < 1 or not 0 <= source_probability <= 1 or max_hazard <= 0:
        raise ValueError('invalid sampling parameters')
    x = (torch.rand((batch_size, graph.n), device=device, generator=generator) < source_probability).long()
    x0 = x.clone()
    times = time_grid(steps, grid, device)
    history = [x.clone()] if keep_path else []
    total = 0
    was_training = model.training
    model.eval()
    try:
        for k in range(steps):
            t, end = float(times[k]), float(times[k + 1])
            while end - t > 1e-12:
                ts = torch.full((batch_size,), t, device=device)
                rates = reference.rate(ts).unsqueeze(-1) * model(x, ts, graph).double().exp()
                if not torch.isfinite(rates).all():
                    raise FloatingPointError('nonfinite controlled rate')
                maximum = float(rates.max())
                dt = min(end - t, max_hazard / maximum) if maximum > 0 else end - t
                if not math.isfinite(dt) or dt <= 0 or total >= max_substeps:
                    raise RuntimeError('sampling substep budget exceeded; inspect rates/temperature')
                prob = -torch.expm1(-2 * rates * dt) / 2
                flip = torch.rand(prob.shape, device=device, generator=generator) < prob
                x = x ^ flip.long()
                t += dt
                total += 1
            if keep_path:
                history.append(x.clone())
    finally:
        model.train(was_training)
    path = torch.stack(history) if keep_path else torch.empty(0, device=device, dtype=torch.long)
    return Rollout(x0, x, times, path, total)


# ------------------------------------------------------------------ exact enumeration (small graphs only)

def states(n, max_nodes=10):
    if not 1 <= n <= max_nodes:
        raise ValueError(f'enumeration limited to {max_nodes} nodes')
    return ((torch.arange(2 ** n)[:, None] >> torch.arange(n)) & 1).long()


def state_indices(x):
    return (x.long() * (2 ** torch.arange(x.shape[-1], device=x.device))).sum(-1)


@dataclass
class ExactBridge(object):
    states: torch.Tensor
    source: np.ndarray
    target: np.ndarray
    joint: np.ndarray
    log_a: np.ndarray        # log phihat_0
    log_b: np.ndarray        # log phi_1
    corrector: np.ndarray    # log phihat_1
    iterations: int
    marginal_error: float

    def controller_log_ratio(self, reference, t):
        """log Phi_t(x)_i for every state x and bit i, shape (2^n, n)."""
        log_k = reference.log_kernel(self.states[:, None], self.states[None, :], t, 1.0).numpy()
        log_phi = logsumexp(log_k + self.log_b[None, :], axis=1)
        ids = np.arange(len(self.states))[:, None] ^ (1 << np.arange(self.states.shape[1]))
        return log_phi[ids] - log_phi[:, None]


def solve_bridge(graph, reference, temperature=1.0, source_probability=0.5, tolerance=1e-11, max_iterations=20000,
                 max_nodes=10):
    """Sinkhorn on the enumerated states: the exact SB from Bernoulli(p)^n to exp(-H / temperature)."""
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError('positive finite temperature required')
    if not 0 < source_probability < 1:
        raise ValueError('exact solver requires a full-support Bernoulli source')
    x = states(graph.n, max_nodes)
    log_mu = (x * np.log(source_probability) + (1 - x) * np.log1p(-source_probability)).sum(-1).numpy()
    log_nu = -graph.energy(x.double()).numpy() / temperature
    log_nu -= logsumexp(log_nu)
    log_k = reference.log_kernel(x[:, None], x[None, :]).numpy()
    b = np.zeros(len(x))
    for iteration in range(1, max_iterations + 1):
        a = log_mu - logsumexp(log_k + b[None, :], axis=1)
        b = log_nu - logsumexp(log_k + a[:, None], axis=0)
        joint = np.exp(a[:, None] + log_k + b[None, :])
        error = max(np.max(abs(joint.sum(1) - np.exp(log_mu))), np.max(abs(joint.sum(0) - np.exp(log_nu))))
        if error < tolerance:
            break
    else:
        raise RuntimeError(f'Sinkhorn did not converge: marginal error={error}')
    corrector = logsumexp(log_k + a[:, None], axis=0)
    return ExactBridge(x, np.exp(log_mu), np.exp(log_nu), joint, a, b, corrector, iteration, float(error))


# ------------------------------------------------------------------ metrics

def evaluate_samples(graph, samples, exact=None):
    """Raw-sample fidelity (energy, feasibility, TV to the exact target if given) and greedy-decoded MIS sizes."""
    if samples.ndim != 2 or samples.shape[1] != graph.n or len(samples) == 0:
        raise ValueError('samples must have shape [batch, nodes]')
    if not torch.all((samples == 0) | (samples == 1)):
        raise ValueError('expected binary samples')
    decoded = graph.decode(samples.float())
    result = {'samples': len(samples), 'raw_energy_mean': graph.energy(samples).mean().item(),
              'raw_feasible_fraction': (graph.violations(samples) == 0).float().mean().item(),
              'decoded_size_mean': decoded.sum(-1).mean().item(),
              'decoded_size_best': int(decoded.sum(-1).max().item()),
              'decoded_feasible': bool((graph.violations(decoded) == 0).all())}
    if exact is not None:
        counts = torch.bincount(state_indices(samples.cpu()), minlength=len(exact.target)).double()
        target = torch.as_tensor(exact.target)
        result['raw_total_variation'] = ((counts / counts.sum() - target).abs().sum() / 2).item()
        result['target_energy_mean'] = float((graph.energy(exact.states.double()).cpu() * target).sum())
        feasible = graph.violations(exact.states) == 0
        result['exact_mis_size'] = int(exact.states.sum(-1)[feasible].max())
    return result
