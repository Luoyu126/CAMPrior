"""Stage 1B criterion 3 of docs/20261125/plan.md: run the SB samplers exactly as solvers/difusco.py does
(DIFUSCO.renoise = sampler B, DIFUSCO.categorical_posterior = sampler A, the inference schedule of evaluate_single),
with the network replaced by a table: at every step each chain draws a label from the exact posterior
P*(m | x_t) ∝ g_m K_{a_t}(s_m | x_t) and outputs it as a one-hot "prediction". Both samplers must then end on the
labels with law pi1, and the pair (start z, end y) must follow the SB coupling pi* (math_route1.md Props 5.1, 5.2).
Checked with Pearson chi-square tests on many chains. For comparison, the same with the exact per-bit marginal
posterior (what a node-wise network can at best output) shows the product-denoiser error (math_route1.md §5).

Run with the cam environment (solvers/difusco.py imports packages absent from the sb env):
  CUDA_VISIBLE_DEVICES=<gpu> $CAM_PYTHON scripts/sb/check_difusco_sb_sampler.py
"""
import math
import os
import sys
import types

import numpy as np
import torch
from scipy.stats import chi2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from solvers.difusco import DIFUSCO                                                 # noqa: E402
from utils import sb_coupling as sb                                                 # noqa: E402
from utils.diffusion_schedulers import CategoricalDiffusion, DiffusionInferenceSchedule  # noqa: E402


def chisq(counts, probs):
    """Pearson chi-square p-value of counts against probs (cells with expected count < 5 pooled)."""
    counts = np.asarray(counts, float); probs = np.asarray(probs, float)
    exp = counts.sum() * probs
    small = exp < 5
    if small.any():
        counts = np.append(counts[~small], counts[small].sum()); exp = np.append(exp[~small], exp[small].sum())
    stat = ((counts - exp) ** 2 / exp).sum()
    return chi2.sf(stat, len(exp) - 1)


def run(sampler, joint, n, labels, pi1, log_g, a, steps, N, dev, gen):
    Q_bar = sb.signal_to_Q_bar(a, as_numpy=True)
    fake = types.SimpleNamespace(sb=True, sb_a=a, sb_z=None, sampler=sampler, solution_type='node',
                                 diffusion=types.SimpleNamespace(Q_bar=Q_bar))
    graph = types.SimpleNamespace(num_nodes=n)
    sched = DiffusionInferenceSchedule(inference_schedule='linear', T=len(a) - 1, inference_T=steps)
    x = torch.randint(0, 2, (n, N), device=dev, generator=gen).float()
    fake.sb_z = x.clone()
    lab = labels.to(dev).double()
    for t in range(steps):
        t1, t2 = sched(t)
        post = sb.label_log_posterior(x.T, lab, log_g, a[int(t1)].item()).exp()          # (N, M)
        if joint:
            m = torch.multinomial(post, 1, generator=gen).squeeze(1)
            p1 = lab[m].T                                                                  # (n, N) one-hot labels
        else:
            p1 = (post @ lab).T                                                            # per-bit marginals
        prob = torch.stack([1 - p1, p1], -1).float()                                       # (n, N, 2)
        if sampler == 'posterior':
            x = DIFUSCO.categorical_posterior(fake, int(t2), int(t1), prob, x)
        else:
            x = DIFUSCO.renoise(fake, graph, prob, int(t2))
        x = x.reshape(n, N)
    return fake.sb_z, x


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    n, M = 6, 4
    X = torch.tensor(np.array(np.meshgrid(*[[0, 1]] * n, indexing='ij')).reshape(n, -1).T)   # all 2^n states
    gen = torch.Generator().manual_seed(1)
    labels = X[torch.randperm(len(X), generator=gen)[:M]]
    pi1 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    a = sb.schedule_for_omega(CategoricalDiffusion(T=50, schedule='cosine').Q_bar, 1.0)
    a_end = a[-1].item()
    log_g, err, _ = sb.sinkhorn(labels, pi1, a_end, X, tol=1e-14)
    pi_star = (sb.coupling_conditional(X, labels, log_g, a_end) / len(X)).numpy()           # (2^n, M)
    print(f'n={n} M={M} pi1={pi1.tolist()} omega=1.0 a_T={a_end:.4f} Sinkhorn err {err:.1e}')
    N = 400000
    ok = True
    for si, sampler in enumerate(('renoise', 'posterior')):
        for steps in (50, 10):
            g = torch.Generator(dev).manual_seed(100 * si + steps)
            for joint in (True, False):
                z, y = run(sampler, joint, n, labels, pi1, log_g.to(dev), a, steps, N, dev, g)
                y = y.round().long().cpu(); z = z.long().cpu()
                code = lambda v: (v.T * (2 ** torch.arange(n - 1, -1, -1))).sum(1)
                lab_code = (labels * (2 ** torch.arange(n - 1, -1, -1))).sum(1)
                yc = code(y); zc = code(z)
                idx = torch.full((len(yc),), -1, dtype=torch.long)
                for m in range(M):
                    idx[yc == lab_code[m]] = m
                on_labels = (idx >= 0).float().mean().item()
                cnt = torch.bincount(idx[idx >= 0], minlength=M).numpy()
                tv = 0.5 * np.abs(cnt / N - pi1.numpy()).sum()
                if joint:
                    pj = chisq(torch.bincount(zc[idx >= 0] * M + idx[idx >= 0], minlength=len(X) * M).numpy(),
                               pi_star.reshape(-1))
                    pm = chisq(cnt, pi1.numpy())
                    good = on_labels == 1.0 and pm > 1e-3 and pj > 1e-3
                    ok &= good
                    print(f'{sampler:9s} steps={steps:2d} joint posterior : on labels {on_labels:.4f} | TV(end, pi1) {tv:.4f}'
                          f' | chi2 p (end ~ pi1) {pm:.3f} | chi2 p ((z, end) ~ pi*) {pj:.3f} | {"PASS" if good else "FAIL"}')
                else:
                    print(f'{sampler:9s} steps={steps:2d} per-bit posterior: on labels {on_labels:.4f} | TV(end, pi1) {tv:.4f}'
                          f'   (product-denoiser error, math_route1.md §5; not an acceptance check)')
    print('ALL PASS' if ok else 'SOME FAILED')


if __name__ == '__main__':
    main()
