"""Effect of clipping the ctrl-AM regression target on the EXACT SB sampler (stage 2C, docs/20261125/status.md).

math_route2.md §6: a clipped target min(xi, e^L) moves the regression's fixed point from E[xi | x_t] to
E[min(xi, e^L) | x_t]; if 2C needs it, it is a bias that must be recorded and measured on small graphs. This script
measures it exactly: for every small graph (n <= 12, all 2^n states enumerated) it solves the SB from
Bernoulli(0.5)^n to nu ∝ exp(-H / tau) (Sinkhorn, float64, on the GPU), and propagates the terminal law of the
tau-leaping sampler driven by
  exact     Phi_t(x)_i   = E[xi_i(Y) | X_t = x]                (xi_i(y) = phi_1(y^i) / phi_1(y), the ctrl-AM target)
  clip L    Phi^L_t(x)_i = E[min(xi_i(Y), e^L) | X_t = x]      (the fixed point of the clipped regression)
with p(y | x) = K_{t,1}(x, y) phi_1(y) / phi_t(x). Self-check: the "exact" field equals ExactBridge.controller_log_ratio
(max abs log difference printed as check).
Reported per (graph, tau, alpha, L, steps): TV(law, nu), TV(clip law, exact law), expected greedy-decoded MIS size
(the decode of solvers / env applied to the terminal state; the main metric), P(feasible).
Usage (from cam/): CUDA_VISIBLE_DEVICES=0 python scripts/sb/dasbs_clip_effect.py --data DIR > OUT.jsonl
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from datasets.mis import Dataset                   # noqa: E402
from solvers.dasbs import gamma_integral           # noqa: E402
from utils import dasbs as D                       # noqa: E402


def sinkhorn(log_mu, log_nu, log_k, tol=1e-11, max_iter=20000):
    b = torch.zeros_like(log_nu)
    for it in range(1, max_iter + 1):
        a = log_mu - torch.logsumexp(log_k + b[None, :], 1)
        b = log_nu - torch.logsumexp(log_k + a[:, None], 0)
        if it % 10 == 0 or it == 1:
            joint = (a[:, None] + log_k + b[None, :]).exp()
            err = max(float((joint.sum(1) - log_mu.exp()).abs().max()), float((joint.sum(0) - log_nu.exp()).abs().max()))
            if err < tol:
                return a, b, it, err
    raise RuntimeError(f'Sinkhorn did not converge: {err}')


def law_step(p, q, X):
    T = torch.ones(len(X), len(X), dtype=torch.float64, device=p.device)
    for i in range(X.shape[1]):
        diff = X[:, i][:, None] != X[None, :, i]
        T *= torch.where(diff, q[:, i:i + 1], 1 - q[:, i:i + 1])
    return p @ T


def run_graph(data, tau, alpha, beta, clips, steps_list, dev, check_cpu):
    n = data.num_nodes
    A = torch.zeros(n, n)
    A[data.edge_index[0], data.edge_index[1]] = 1
    g = D.MISGraph(A, penalty=beta)
    Xc = D.states(n, max_nodes=12)
    X = Xc.to(dev)
    S = len(X)
    H = g.energy(Xc.double()).to(dev)
    log_nu = torch.log_softmax(-H / tau, 0)
    nu = log_nu.exp()
    feas = (g.violations(Xc) == 0).double().to(dev)
    dec = g.decode(Xc.double()).sum(-1).double().to(dev)          # greedy-decoded size of every terminal state
    ref = D.UniformFlipReference('log_linear', 1.0, alpha)
    log_mu = torch.full((S,), -n * np.log(2.0), dtype=torch.float64, device=dev)
    log_k1 = ref.log_kernel(Xc[:, None], Xc[None, :]).to(dev)
    a, b, iters, err = sinkhorn(log_mu, log_nu, log_k1)
    ids = (torch.arange(S, device=dev)[:, None] ^ (1 << torch.arange(n, device=dev))[None, :])
    log_xi = b[ids] - b[:, None]                                  # (S_y, n): log xi_i(y)
    grid = D.time_grid(max(steps_list), 'quadratic')
    fields = {}                                                    # t index -> {L: (S, n) log Phi}
    check = 0.0
    for k in range(len(grid) - 1):
        t0 = float(grid[k])
        if np.isinf(gamma_integral(ref, t0, float(grid[k + 1]))):
            continue
        lk = ref.log_kernel(Xc[:, None], Xc[None, :], t0, 1.0).to(dev)
        lp = lk + b[None, :]
        lp = lp - torch.logsumexp(lp, 1, keepdim=True)            # log p(y | x)
        f = {}
        for L in [None] + list(clips):
            lx = log_xi if L is None else log_xi.clamp(max=L)
            f[L] = torch.stack([torch.logsumexp(lp + lx[:, i][None, :], 1) for i in range(n)], 1)
        fields[k] = f
        if check_cpu and k in (1, len(grid) // 2):
            ex = D.ExactBridge(Xc, None, None, None, None, b.cpu().numpy(), None, iters, err)
            check = max(check, float(np.abs(ex.controller_log_ratio(ref, t0) - f[None].cpu().numpy()).max()))
    rows = []
    tv = lambda u, v: 0.5 * float((u - v).abs().sum())
    for steps in steps_list:
        stride = max(steps_list) // steps
        laws = {}
        for L in [None] + list(clips):
            p = torch.full((S,), 1.0 / S, dtype=torch.float64, device=dev)
            for kk in range(steps):
                k0, k1 = kk * stride, (kk + 1) * stride
                G = gamma_integral(ref, float(grid[k0]), float(grid[k1]))
                if np.isinf(G):
                    q = torch.full((S, n), 0.5, dtype=torch.float64, device=dev)
                else:
                    q = (G / 2 * fields[k0][L].exp()).clamp(max=1.0)
                p = law_step(p, q, X)
            laws[L] = p
        for L in [None] + list(clips):
            p = laws[L]
            rows.append(dict(n=n, tau=tau, alpha=alpha, steps=steps, clip=L, tv=tv(p, nu), tv_to_exact=tv(p, laws[None]),
                             mis=float(p @ dec), mis_target=float(nu @ dec), feasible=float(p @ feas),
                             max_log_xi=float(log_xi.max()), sinkhorn_iters=iters, check=check))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--beta', type=float, default=1.2)
    ap.add_argument('--taus', default='1,0.3,0.1,0.05')
    ap.add_argument('--alphas', default='0,0.25,0.5,1.0')
    ap.add_argument('--clips', default='10,15,20,30')
    ap.add_argument('--steps', default='50,10')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = Dataset(args.data, torch.float32, args.limit)
    taus = [float(v) for v in args.taus.split(',')]
    alphas = [float(v) for v in args.alphas.split(',')]
    clips = [float(v) for v in args.clips.split(',')]
    steps = [int(v) for v in args.steps.split(',')]
    assert all(max(steps) % s == 0 for s in steps)
    for idx in range(len(ds)):
        for tau, alpha in itertools.product(taus, alphas):
            for r in run_graph(ds[idx], tau, alpha, args.beta, clips, steps, dev, check_cpu=(idx == 0)):
                r['idx'] = idx
                print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
