"""Tests for utils/bridge_paths.py (UPSTREAM.md item 43)."""
import os
import sys
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.bridge_paths import sample_path, step_flip_prob, flip_prob_ref  # noqa: E402

N = 400_000


def test_mixture_marginal():
    torch.manual_seed(0)
    x0 = torch.zeros(N); x1 = torch.ones(N)
    for t in [0.0, 0.3, 0.7, 1.0]:
        xt = sample_path(x0, x1, t, 0.0)
        assert abs(xt.mean().item() - t) < 0.005, (t, xt.mean())
    same = sample_path(torch.ones(N), torch.ones(N), 0.4, 0.0)
    assert same.min().item() == 1.0


def test_bridge_endpoints_and_marginal():
    torch.manual_seed(1)
    lam = 0.7
    for a, b in [(0., 0.), (0., 1.), (1., 0.), (1., 1.)]:
        x0 = torch.full((N,), a); x1 = torch.full((N,), b)
        assert torch.equal(sample_path(x0, x1, 0.0, lam), x0)
        assert torch.equal(sample_path(x0, x1, 1.0, lam), x1)
        # closed form vs simulating the reference CTMC and conditioning on the end value
        t = 0.35
        pf = lambda s: (1 - math.exp(-2 * lam * s)) / 2
        c1 = pf(t) if a == 0 else 1 - pf(t)                       # P(x_t = 1 | a)
        e1 = (1 - pf(1 - t)) if b == 1 else pf(1 - t)              # P(end = b | x_t = 1)
        e0 =(1 - pf(1 - t)) if b == 0 else pf(1 - t)              # P(end = b | x_t = 0)
        expect = c1 * e1 / (c1 * e1 + (1 - c1) * e0)
        got = sample_path(x0, x1, t, lam).mean().item()
        assert abs(got - expect) < 0.005, (a, b, got, expect)


def test_bridge_tends_to_mixture():
    torch.manual_seed(2)
    x0 = torch.zeros(N); x1 = torch.ones(N)
    a = sample_path(x0, x1, 0.6, 1e-4).mean().item()
    assert abs(a - 0.6) < 0.005
    x = torch.zeros(5); p1 = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
    f0 = step_flip_prob(x, p1, 0.3, 0.32, 0.0)
    f1 = step_flip_prob(x, p1, 0.3, 0.32, 1e-5)
    assert torch.allclose(f0, f1, atol=5e-4), (f0, f1)   # Euler vs 1-exp(-rate h): O(h^2)


def test_sampler_reaches_label_with_oracle():
    # with the exact denoiser (p1 = x1), both samplers transport any start to x1
    torch.manual_seed(3)
    n = 20000
    x1 = (torch.rand(n) < 0.1).float()
    for lam in [0.0, 0.5]:
        x = (torch.rand(n) < 0.5).float()
        ts = [math.sin(i / 50 * math.pi / 2) for i in range(51)]
        for t1, t2 in zip(ts[:-1], ts[1:]):
            if t2 >= 1:
                x = x1.clone()   # last step outputs the prediction itself
                break
            flip = step_flip_prob(x, x1, t1, t2, lam)
            x = torch.where(torch.rand(n) < flip, 1 - x, x)
        assert (x != x1).float().mean().item() == 0.0
        # one step before the end most nodes must already agree
        x = (torch.rand(n) < 0.5).float()
        for t1, t2 in zip(ts[:-2], ts[1:-1]):
            x = torch.where(torch.rand(n) < step_flip_prob(x, x1, t1, t2, lam), 1 - x, x)
        assert (x != x1).float().mean().item() < 0.05, lam


def test_marginal_matches_path_under_oracle_lambda0():
    # simulate the sampler with the exact posterior for a 2-point label distribution and compare
    # the marginal at t with the mixture-path marginal
    torch.manual_seed(4)
    n = 200000
    x0 = (torch.rand(n) < 0.5).float()
    x1 = torch.ones(n)
    x = x0.clone()
    ts = [i / 100 for i in range(61)]
    for t1, t2 in zip(ts[:-1], ts[1:]):
        x = torch.where(torch.rand(n) < step_flip_prob(x, x1, t1, t2, 0.0), 1 - x, x)
    expect = 0.6 + 0.4 * 0.5
    assert abs(x.mean().item() - expect) < 0.01, (x.mean(), expect)


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('ok', name)
