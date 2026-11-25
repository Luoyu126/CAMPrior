"""Stage 1D pieces of solvers/difusco.py (learned coupling, math_route1.md §12.4). CPU only, cam environment:

    $CAM_PYTHON cam/tests/test_difusco_sb_reverse.py

1. cache_posterior_mean (soft target under the empirical coupling of cached pairs) equals the posterior computed
   independently from the per-bit bridge probabilities (sb_coupling.bridge_prob1).
2. The reversed schedule b_k = a_T / a_{T-k} of --sb_reverse gives the same bridge as the forward one read backwards:
   P(x_k | y, z) on signals a = P(x_{T-k} | z, y) on signals b, for every k.
3. The backward sampler (DIFUSCO.renoise with --sb_reverse: chains start at labels y ~ pi1, the "prediction" is a
   start z drawn from the exact posterior P*(z | x) under the SB coupling) ends with pairs (y, z) ~ pi*, the SB
   coupling (chi-square), i.e. the backward network's sampler is exact when its posterior is. Negative control: the
   same with the posterior of the independent coupling ends far from pi*.
"""
import os
import sys
import time
import traceback
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''
CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from solvers.difusco import DIFUSCO, cache_posterior_mean  # noqa: E402
from utils import sb_coupling as sb  # noqa: E402
from utils import sb_imf as imf  # noqa: E402
from utils.diffusion_schedulers import CategoricalDiffusion, DiffusionInferenceSchedule  # noqa: E402

torch.set_num_threads(4)


def schedule(omega, T=50):
    return sb.schedule_for_omega(CategoricalDiffusion(T=T, schedule='cosine').Q_bar, omega)


def test_cache_posterior_mean():
    g = torch.Generator().manual_seed(0)
    a = schedule(0.7)
    for n, J in ((6, 5), (9, 12)):
        Y = (torch.rand(J, n, generator=g) < 0.4).double()
        Z = (torch.rand(J, n, generator=g) < 0.5).double()
        for k in (1, 10, 30, 49, 50):
            x = (torch.rand(n, generator=g) < 0.5).double()
            got = cache_posterior_mean(x, Z, Y, a[k].item(), a[-1].item())
            p1 = sb.bridge_prob1(Y, Z, a[k].item(), a[-1].item())                       # (J, n)
            lik = torch.where(x[None] == 1, p1, 1 - p1).prod(1)
            if lik.sum() == 0:                                                         # k = T: x must equal some z_j
                continue
            want = (lik / lik.sum()) @ Y
            assert (got - want).abs().max() < 1e-10, (n, k, (got - want).abs().max())
        # at k = T only pairs whose start equals x_T carry weight
        x = Z[2].clone()
        got = cache_posterior_mean(x, Z, Y, a[-1].item(), a[-1].item())
        same = (Z == x).all(1)
        assert (got - Y[same].mean(0)).abs().max() < 1e-12


def test_reverse_schedule_bridge():
    a = schedule(0.5)
    b = a[-1] / a.flip(0)
    assert abs(b[0].item() - 1) < 1e-15 and abs(b[-1].item() - a[-1].item()) < 1e-15
    assert bool((b[1:] <= b[:-1]).all())
    y = torch.tensor([0., 0, 1, 1]); z = torch.tensor([0., 1, 0, 1])
    T = len(a) - 1
    for k in range(T + 1):
        fwd = sb.bridge_prob1(y, z, a[k].item(), a[-1].item())
        rev = sb.bridge_prob1(z, y, b[T - k].item(), b[-1].item())
        assert (fwd - rev).abs().max() < 1e-12, k


def chisq(counts, probs):
    counts = np.asarray(counts, float); probs = np.asarray(probs, float)
    exp = counts.sum() * probs
    small = exp < 5
    if small.any():
        counts = np.append(counts[~small], counts[small].sum()); exp = np.append(exp[~small], exp[small].sum())
    stat = ((counts - exp) ** 2 / exp).sum()
    return chi2.sf(stat, len(exp) - 1)


def run_reverse(P, coupling, b, steps, N, gen):
    """Backward sampler of DIFUSCO.renoise (sampler B, --sb_reverse) with a tabular joint posterior over starts."""
    n = P.n
    fake = types.SimpleNamespace(sb=True, sb_a=b, sb_z=None, sampler='renoise', solution_type='node',
                                 diffusion=types.SimpleNamespace(Q_bar=sb.signal_to_Q_bar(b, as_numpy=True)))
    graph = types.SimpleNamespace(num_nodes=n)
    sched = DiffusionInferenceSchedule(inference_schedule='linear', T=len(b) - 1, inference_T=steps)
    m0 = torch.multinomial(P.pi1, N, replacement=True, generator=gen)
    x = P.Y[m0].T.clone()                                                                    # (n, N): labels
    fake.sb_z = x.clone()
    Zs = P.X                                                                                 # all starts (S, n)
    a_T = b[-1].item()
    for t in range(steps):
        t1, t2 = sched(t)
        # reverse time t1 <-> forward time T - t1; P*(z | x) ∝ sum_m pi*(z, m) bridge(x | y_m, z) (forward signals)
        ak = (a_T / b[int(t1)]).item()
        p1 = sb.bridge_prob1(P.Y[:, None, :].expand(-1, len(Zs), -1), Zs[None].expand(P.M, -1, -1), ak, a_T)  # (M,S,n)
        lik = torch.where(x.T[:, None, None, :] == 1, p1[None], 1 - p1[None]).prod(-1)     # (N, M, S)
        post = (lik * coupling.T[None]).sum(1)                                               # (N, S)
        post = post / post.sum(1, keepdim=True)
        zi = torch.multinomial(post, 1, generator=gen).squeeze(1)
        p = Zs[zi].T                                                                         # (n, N) one-hot "pred"
        prob = torch.stack([1 - p, p], -1).float()
        x = DIFUSCO.renoise(fake, graph, prob, int(t2)).reshape(n, N).double()
    return m0, x


def test_backward_sampler_ends_on_sb_coupling():
    gen = torch.Generator().manual_seed(1)
    torch.manual_seed(1)
    n = 4
    Y = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]], dtype=torch.float64)
    a = schedule(1.0, T=20)
    P = imf.TabularIMF(Y, torch.full((4,), 0.25, dtype=torch.float64), a)
    star = P.sb_coupling()                                                                   # (S, M)
    b = a[-1] / a.flip(0)
    N = 40000
    for steps in (20, 5):
        m0, x = run_reverse(P, star, b, steps, N, gen)
        zi = (x.T.round().long() * (2 ** torch.arange(n))).sum(1)                          # state index, LSB first
        cell = (m0 * len(P.X) + zi).numpy()
        counts = np.bincount(cell, minlength=P.M * len(P.X))
        probs = star.T.reshape(-1).numpy()
        pv = chisq(counts, probs)
        tvv = 0.5 * np.abs(counts / N - probs).sum()
        print(f'  backward sampler, {steps} steps: chi2 p {pv:.3f}, TV {tvv:.4f}')
        assert pv > 1e-3, pv
    # negative control: driven by the independent coupling's posterior, the pairs are off pi*
    m0, x = run_reverse(P, P.independent(), b, 20, N, gen)
    zi = (x.T.round().long() * (2 ** torch.arange(n))).sum(1)
    counts = np.bincount((m0 * len(P.X) + zi).numpy(), minlength=P.M * len(P.X))
    assert chisq(counts, star.T.reshape(-1).numpy()) < 1e-6


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f'PASS {name} ({time.time()-t0:.1f}s)', flush=True)
        except Exception:
            failed += 1
            print(f'FAIL {name}\n{traceback.format_exc()}', flush=True)
    print(f'{len(tests)-failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
