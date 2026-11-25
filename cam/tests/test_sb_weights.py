"""Size-weighted label distribution pi1 (stage 1E of docs/20261125/plan.md) for the route-1 SB coupling.

With pi1(m) ∝ exp(|s_m| / T) instead of uniform, the static SB (math_route1.md §2) is the same Sinkhorn problem with
a different label marginal; at omega = 0 (the existing cosine table, signal ~1e-6 at the noise end) the coupling is
independent and the potential is log g = log pi1 (scripts/sb/precompute_logg.py --omega 0 --weight_temp T, read by
solvers/difusco.py through --sb_logg). Exact enumeration over all 2^n starts, float64.

    /usr0/home/shengyuf/yunyi/envs/sb/bin/python -m pytest cam/tests/test_sb_weights.py
"""
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import sb_coupling as sb                       # noqa: E402
from utils.diffusion_schedulers import CategoricalDiffusion  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'precompute_logg', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'sb', 'precompute_logg.py'))
pl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pl)


def all_states(n):
    return ((torch.arange(2 ** n)[:, None] >> torch.arange(n)) & 1).to(torch.float64)


def labels_of(n, M, seed):
    g = torch.Generator().manual_seed(seed)
    s = (torch.rand(M, n, generator=g) < 0.4).to(torch.uint8)
    return torch.unique(s, dim=0)


def exact_label_marginal(labels, log_g, a_T):
    Z = all_states(labels.shape[1])
    return sb.coupling_conditional(Z, labels, log_g, a_T).mean(0)


def test_label_target():
    s = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.uint8)
    w = pl.label_target(s, 2.0)
    want = torch.exp(torch.tensor([2.0, 1.0, 3.0]) / 2.0); want /= want.sum()
    assert torch.allclose(w, want.double(), atol=1e-15)
    assert torch.allclose(pl.label_target(s), torch.full((3,), 1 / 3, dtype=torch.float64))


def test_weighted_sinkhorn_marginal():
    """omega > 0: the exact Sinkhorn with pi1 = size weights reproduces pi1; uniform log g does not."""
    for seed, n, M, T, om in ((0, 8, 10, 1.0, 0.5), (1, 9, 14, 0.5, 1.0), (2, 7, 8, 2.0, 0.35)):
        s = labels_of(n, M, seed)
        w = pl.label_target(s, T)
        a = sb.schedule_for_omega(CategoricalDiffusion(T=50, schedule='cosine').Q_bar, om)[-1].item()
        lg, err, _ = sb.sinkhorn(s, w, a, all_states(n), iters=20000, tol=1e-13)
        assert (exact_label_marginal(s, lg, a) - w).abs().max() < 1e-12
        lg_u, _, _ = sb.sinkhorn(s, torch.full_like(w, 1 / len(w)), a, all_states(n), iters=20000, tol=1e-13)
        assert (exact_label_marginal(s, lg_u, a) - w).abs().max() > 1e-3          # negative control
        # Monte Carlo version (the one precompute_logg.py runs) on the exact distance table of all starts
        D = all_states(n) @ (1 - 2 * s.double().T) + s.double().sum(1)              # Hamming distances
        lg_mc, rel, _ = sb.sinkhorn_distances(D.round().to(torch.int16), sb.omega(a), w, tol=1e-7, iters=20000)
        assert ((exact_label_marginal(s, lg_mc, a) - w).abs() / w).max() < 1e-5


def test_omega0_weighted_is_independent_with_prior_w():
    """omega = 0: log g = log pi1 gives label marginal pi1 up to the cosine table's ~1e-6 end signal, draws
    independent of the start, and the soft target is the Bayes posterior with prior pi1."""
    a = sb.schedule_for_omega(CategoricalDiffusion(T=50, schedule='cosine').Q_bar, 0.0)
    for seed, n, M, T in ((3, 8, 12, 1.0), (4, 9, 10, 0.5)):
        s = labels_of(n, M, seed)
        w = pl.label_target(s, T)
        lg = w.log()
        assert (exact_label_marginal(s, lg, a[-1].item()) - w).abs().max() < 1e-5
        Z = all_states(n)
        P = sb.coupling_conditional(Z, s, lg, a[-1].item())
        assert (P - w[None]).abs().max() < 1e-4                                 # (nearly) independent of z
        for k in (5, 25, 45):
            x = Z[torch.randperm(len(Z), generator=torch.Generator().manual_seed(k))[:16]]
            got = sb.posterior_mean(x, s, lg, a[k].item())
            p = torch.tensor((1 + a[k].item()) / 2, dtype=torch.float64)
            lik = torch.stack([torch.where(x == s[m].double(), p, 1 - p).prod(1) for m in range(len(s))], 1)
            post = w[None] * lik; post /= post.sum(1, keepdim=True)
            assert (got - post @ s.double()).abs().max() < 1e-12
            unif = sb.posterior_mean(x, s, torch.zeros_like(lg), a[k].item())
            if k == 5:
                assert (unif - got).abs().max() > 1e-3                          # negative control
