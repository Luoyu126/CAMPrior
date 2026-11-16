"""Energy-heated forward process for DIFUSCO on MIS (local addition, see UPSTREAM.md item 42).

DIFUSCO's forward flips every node independently with probability beta_t / 2 per step. Here the flip
probability of node i at step s (x_{s-1} -> x_s) is

    beta_s * sigmoid(lambda_s * dE_i(x_{s-1})),

where dE_i is the change of the MIS energy E(x) = sum_i x_i (beta_pen (Ax)_i + b_i) when node i is
flipped: dE_i = (1 - 2 x_i)(2 beta_pen (Ax)_i + b_i), the same gradient as `envs/mis.py`. With
lambda > 0, flips that raise the energy are more likely (the reverse of an RLSA step). lambda = 0
gives the uniform kernel beta_s / 2 exactly. The expected number of flips per step is not normalized.

lambda_s decays linearly from `heat_lambda` at s = 0 to 0 at s = heat_until * T and stays 0 after,
so the late steps are the uniform kernel and x_T is still close to Bernoulli(0.5). Those uniform
steps are applied in one closed-form jump.

There is no closed-form marginal, so x_t is simulated step by step from x_0. The same simulation is
used for the training inputs and, in the `renoise` sampler, to noise a predicted x_0 down to the next
inference step, so training and inference see the same marginals q(x_t | x_0).
"""
import numpy as np
import torch


class HeatedForward(object):
    def __init__(self, beta_schedule, heat_lambda, heat_until, penalty):
        self.beta = np.asarray(beta_schedule, dtype=np.float64)  # beta[s-1] is the rate of step s
        self.T = len(self.beta)
        self.heat_lambda = float(heat_lambda)
        self.penalty = float(penalty)
        heat_T = int(round(heat_until * self.T))
        # last step with lambda_s > 0; steps after it are uniform and done in one jump
        self.H = 0 if self.heat_lambda == 0 else max(0, min(heat_T - 1, self.T))
        self.heat_T = heat_T
        cp = np.concatenate([[1.0], np.cumprod(1.0 - self.beta)])  # cp[s] = prod_{r<=s} (1 - beta_r)
        self._cp = torch.from_numpy(cp)

    def lam(self, s):
        if s >= self.heat_T:
            return 0.0
        return self.heat_lambda * (1.0 - s / self.heat_T)

    @staticmethod
    def delta_energy(A, b, x, penalty):
        """Energy change of flipping each node; x is (N, K), b is (N, 1)."""
        return (1 - 2 * x) * (2 * penalty * (A @ x) + b)

    def step(self, A, b, x, s):
        p = self.beta[s - 1] * torch.sigmoid(self.lam(s) * self.delta_energy(A, b, x, self.penalty))
        return torch.abs(x - torch.bernoulli(p))

    def uniform_jump(self, x, s_from, s_to):
        """Uniform steps s_from+1 .. s_to (s_to: LongTensor (M,)) in closed form."""
        cp = self._cp.to(x.device)
        keep = (cp[s_to] / cp[s_from]).float()
        p = ((1 - keep) / 2).unsqueeze(1).expand_as(x)
        return torch.abs(x - torch.bernoulli(p))

    @torch.no_grad()
    def sample(self, A, b, x0, t):
        """x_t for every node; x0 (N, K) in {0,1}, t LongTensor (N,) in 0..T (per node, constant per graph)."""
        x0 = x0.float()
        x = x0.clone()
        xt = x0.clone()
        n_heat = min(int(t.max().item()), self.H)
        for s in range(1, n_heat + 1):
            x = self.step(A, b, x, s)
            xt = torch.where((t == s).unsqueeze(1), x, xt)
        far = t > self.H
        if bool(far.any()):
            xt[far] = self.uniform_jump(x[far], self.H, t[far])
        return xt
