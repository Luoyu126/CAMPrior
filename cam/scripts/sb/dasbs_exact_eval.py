"""Exact evaluation of a trained cross-graph DASBS (solvers/dasbs.py) on small graphs (stage 2B, docs/20261125/plan.md).

For every graph (n <= 12) the network is evaluated on all 2^n states at every step of the tau-leaping sampler, and the
sampler's terminal law is propagated exactly (no sampling noise; the only errors are the network's and the time
grid's). Compared with the exact target nu ∝ exp(-H / tau) (H = env energy, -|x| + beta * #violated edges):
  tv          TV(terminal law of the learned sampler, nu)
  tv_source   TV(Bernoulli(0.5)^n, nu)
  tv_floor    TV of the same sampler driven by the EXACT SB controller (time discretisation alone)
  energy, energy_target, feasible, feasible_target   mean energy and P(no violated edge) under the two laws
  corr_loop   mean |log Phihat(y)_i + log Phihat(y^i)_j - log Phihat(y)_j - log Phihat(y^j)_i| over all y, i < j:
              0 for any ratio field that comes from a potential (math_route2.md §7, corollary 7.2)
  corr_err    mean |log Phihat - log Phihat_SB| over all states and bits
Output: one JSON line per graph and a summary line (means). With --pairs FILE (lines "i j name", the wl/ set of
scripts/sb/gen_small_graphs.py) also the per-pair numbers.

Usage (from cam/): python scripts/sb/dasbs_exact_eval.py --ckpt RUN/best.pt --data DIR [--num_t 50] [model and
reference flags as in training] > OUT.jsonl
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import models                                      # noqa: E402
from datasets.mis import Dataset                   # noqa: E402
from solvers.dasbs import DASBSNet, corrector_log_bound, gamma_integral   # noqa: E402
from utils import dasbs as D                       # noqa: E402


def build_net(args, reference):
    ctrl = getattr(models, args.model)(n_layers=args.num_l, hidden_dim=args.num_h, out_channels=1, problem='mis',
                                      time_scale=args.time_scale)
    corr = models.EncodeProcessDecode(n_layers=args.num_l, hidden_dim=args.num_h, out_channels=1, problem='mis',
                                      time_scale=args.time_scale)
    net = DASBSNet(ctrl, corr, corrector_log_bound(reference))
    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True))
    return net.eval()


def law_step(p, q, X):
    """One tau-leaping step on the exact law: bit i of state x flips w.p. q[x, i], independently. p (S,), q (S, n)."""
    T = torch.ones(len(X), len(X), dtype=torch.float64)
    for i in range(X.shape[1]):
        diff = X[:, i][:, None] != X[None, :, i]
        T *= torch.where(diff, q[:, i:i + 1], 1 - q[:, i:i + 1])
    return p @ T


@torch.no_grad()
def evaluate_graph(net, data, reference, grid, tau, beta, exact_floor=True):
    n = data.num_nodes
    X = D.states(n, max_nodes=12)                                     # (S, n), LSB first
    S = len(X)
    data.batch = torch.zeros(n, dtype=torch.long)
    A = torch.zeros(n, n)
    A[data.edge_index[0], data.edge_index[1]] = 1
    g = D.MISGraph(A, penalty=beta)
    H = g.energy(X.double())
    nu = torch.softmax(-H / tau, 0)
    feas = (g.violations(X) == 0).double()
    xs = X.T.float()                                                  # (n, S): every state is one chain
    ex = D.solve_bridge(g, reference, tau, max_nodes=12) if exact_floor else None
    p = torch.full((S,), 1.0 / S, dtype=torch.float64)
    p_ex = p.clone()
    for k in range(len(grid) - 1):
        t0, t1 = float(grid[k]), float(grid[k + 1])
        G = gamma_integral(reference, t0, t1)
        if math.isinf(G):
            q = torch.full((S, n), 0.5, dtype=torch.float64); q_ex = q
        else:
            lphi = net.ctrl(data, xs, torch.full_like(xs, t0)).double().T          # (S, n)
            q = (G / 2 * lphi.exp()).clamp(max=1.0)
            if ex is not None:
                q_ex = (G / 2 * torch.from_numpy(np.exp(ex.controller_log_ratio(reference, t0)))).clamp(max=1.0)
        p = law_step(p, q, X)
        if ex is not None:
            p_ex = law_step(p_ex, q_ex, X)
    tv = lambda a, b: 0.5 * float((a - b).abs().sum())
    out = dict(n=n, edges=int(A.sum() // 2), tv=tv(p, nu), tv_source=tv(torch.full_like(nu, 1.0 / S), nu),
               energy=float(p @ H), energy_target=float(nu @ H), feasible=float(p @ feas),
               feasible_target=float(nu @ feas))
    if ex is not None:
        out['tv_floor'] = tv(p_ex, nu)
        lc = net.corr(data, xs).double().T                                         # (S, n) log Phihat
        ids = torch.arange(S)[:, None] ^ (1 << torch.arange(n))[None, :]
        true = torch.from_numpy(ex.corrector)[ids] - torch.from_numpy(ex.corrector)[:, None]
        out['corr_err'] = float((lc - true).abs().mean())
        loops = []
        for i in range(n):
            for j in range(i + 1, n):
                yi, yj = ids[:, i], ids[:, j]
                loops.append((lc[:, i] + lc[yi, j] - lc[:, j] - lc[yj, i]).abs())
        out['corr_loop'] = float(torch.stack(loops).mean()) if loops else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None, help='best.pt of a DASBS run; none = untrained network')
    ap.add_argument('--data', required=True)
    ap.add_argument('--pairs', default=None)
    ap.add_argument('--model', default='EncodeProcessDecodeTime2')
    ap.add_argument('--num_h', type=int, default=64)
    ap.add_argument('--num_l', type=int, default=8)
    ap.add_argument('--time_scale', type=int, default=1000)
    ap.add_argument('--dasbs_schedule', default='log_linear')
    ap.add_argument('--dasbs_gamma', type=float, default=1.0)
    ap.add_argument('--dasbs_alpha', type=float, default=0.5)
    ap.add_argument('--dasbs_grid', default='quadratic')
    ap.add_argument('--num_t', type=int, default=50)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=2.0)
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--no_floor', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    reference = D.UniformFlipReference(args.dasbs_schedule, args.dasbs_gamma, args.dasbs_alpha)
    grid = D.time_grid(args.num_t, args.dasbs_grid)
    net = build_net(args, reference)
    ds = Dataset(args.data, torch.float32, args.limit)
    rows = []
    for k in range(len(ds)):
        r = evaluate_graph(net, ds[k], reference, grid, args.tau, args.beta, not args.no_floor)
        r['idx'] = k; r['file'] = os.path.basename(ds.files[k])
        rows.append(r)
        print(json.dumps(r), flush=True)
    keys = [k for k in rows[0] if k not in ('idx', 'file', 'n', 'edges')]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary['graphs'] = len(rows)
    summary['tv_le_half_source'] = float(np.mean([r['tv'] <= r['tv_source'] / 2 for r in rows]))
    summary['energy_rel_gap'] = abs(summary['energy'] - summary['energy_target']) / abs(summary['energy_target'])
    if args.pairs:
        by_file = {r['file']: r for r in rows}
        pairs = []
        for line in open(args.pairs):
            i, j, name = line.split()
            a, b = by_file[f'idx_{i}.gpickle'], by_file[f'idx_{j}.gpickle']
            pairs.append(dict(name=name, tv_a=a['tv'], tv_b=b['tv'], floor_a=a.get('tv_floor'), floor_b=b.get('tv_floor')))
        summary['pairs'] = pairs
    print(json.dumps({'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
