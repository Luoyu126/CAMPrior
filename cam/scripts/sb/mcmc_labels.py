"""Perturbed end distributions for route 1' (D4 of docs/20261128/plan_route1prime.md): per training graph, M samples of

    pi_B(s) ∝ exp(|s| / T)   over independent sets s with min_m d(s, k_m) <= r,

k_1..k_K the graph's KaMIS solutions (the label set), d the Hamming distance, drawn by a Metropolis chain that starts
from the KaMIS solutions. The samples are stored as a label set (rows with repeats are the empirical pi_B; use it with
temperature None, i.e. uniform over rows), in the format and node order of the existing *_loaderorder.pt sets.

Moves (one node v drawn uniformly per step; each move's reverse is proposed with the same probability, so the
proposal is symmetric and the acceptance is min(1, exp(Δ|s| / T)) times the radius constraint):
  v in s                          remove v                   (Δ|s| = -1)
  v not in s, no neighbour in s   add v                      (Δ|s| = +1)
  v not in s, one neighbour u     swap: add v, remove u      (Δ|s| = 0)
  v not in s, >= 2 neighbours     stay
Add/remove alone connect all independent sets (through the empty set), so without the radius constraint the chain is
irreducible on independent sets and its stationary law is pi_B (tests/test_dsbm.py). With a small radius the region
can split into pieces around different KaMIS solutions; chains start from every KaMIS solution in turn, so each
piece is visited, but the relative mass between pieces then follows the starts, not pi_B exactly (recorded in the
statistics as the number of distinct samples and their distance to the nearest KaMIS solution).

Statistics per graph (meta['stats'], printed as means): distinct samples, mean size, fraction smaller than the
largest KaMIS solution, mean Hamming distance to the nearest KaMIS solution, acceptance rate.

Usage: python scripts/sb/mcmc_labels.py --out OUT.pt [--label_set LS] [--r 6] [--T 1.0] [--M 128] [--chains 8]
       [--burn 50] [--thin 2] [--graphs N] [--workers 16] [--seed 0]   (burn and thin in sweeps of n steps)
"""
import argparse
import math
import multiprocessing as mp
import os
import sys

import numpy as np

CAM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
TRAIN = '/usr1/shengyuf/MIS/RB_small_train'
LS_DEFAULT = '/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/matched_kamis_loaderorder.pt'


def run_chain(adj, K, x0, r, T, burn, thin, n_samples, rng):
    """One Metropolis chain. adj: list of neighbour lists; K (m, n) int8 KaMIS solutions; x0 (n,) start (an
    independent set within the radius). Returns (samples (n_samples, n) int8, acceptance rate)."""
    n = len(adj)
    x = x0.astype(np.int8).copy()
    cnt = np.zeros(n, dtype=np.int64)                    # neighbours of v in the set
    for v in range(n):
        if x[v]:
            for u in adj[v]:
                cnt[u] += 1
    d = (K != x[None]).sum(1).astype(np.int64)           # distance to every KaMIS solution
    assert d.min() <= r
    Kc = K.T.astype(np.int64)                            # (n, m) columns
    out = np.empty((n_samples, n), dtype=np.int8)
    acc = 0; steps = burn + thin * n_samples
    vs = rng.integers(0, n, steps); us = rng.random(steps)
    k = 0
    for s in range(steps):
        v = vs[s]
        if x[v]:
            delta, flips = -1, (v,)
        elif cnt[v] == 0:
            delta, flips = 1, (v,)
        elif cnt[v] == 1:
            u = next(w for w in adj[v] if x[w])
            delta, flips = 0, (v, u)
        else:
            flips = None
        if flips is not None:
            dd = d.copy()
            for w in flips:                              # flipping bit w: +1 where it agreed with K, -1 otherwise
                dd += np.where(Kc[w] == x[w], 1, -1)
            if dd.min() <= r and (delta >= 0 or us[s] < math.exp(delta / T)):
                for w in flips:
                    x[w] ^= 1
                    inc = 1 if x[w] else -1
                    for u in adj[w]:
                        cnt[u] += inc
                d = dd; acc += 1
        if s >= burn and (s - burn) % thin == thin - 1:
            out[k] = x; k += 1
    assert k == n_samples
    return out, acc / steps


def graph_samples(job):
    i, edges, n, K, a = job
    rng = np.random.default_rng([a['seed'], i])
    adj = [[] for _ in range(n)]
    for u, v in zip(*edges):
        adj[u].append(v); adj[v].append(u)
    per = [a['M'] // a['chains'] + (c < a['M'] % a['chains']) for c in range(a['chains'])]
    outs, accs = [], []
    for c, m in enumerate(per):
        if m == 0:
            continue
        s, ac = run_chain(adj, K, K[c % len(K)], a['r'], a['T'], a['burn'] * n, a['thin'] * n, m, rng)
        outs.append(s); accs.append(ac)
    S = np.concatenate(outs)
    D = (S[:, None, :] != K[None]).sum(-1).min(1)
    st = {'distinct': len(np.unique(S, axis=0)), 'size': float(S.sum(1).mean()),
          'subopt': float((S.sum(1) < K.sum(1).max()).mean()), 'd_kamis': float(D.mean()), 'acc': float(np.mean(accs)),
          'kamis_size': int(K.sum(1).max())}
    return i, S, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--label_set', default=LS_DEFAULT)
    ap.add_argument('--train_path', default=TRAIN)
    ap.add_argument('--r', type=int, default=6)
    ap.add_argument('--T', type=float, default=1.0)
    ap.add_argument('--M', type=int, default=128)
    ap.add_argument('--chains', type=int, default=8)
    ap.add_argument('--burn', type=float, default=50)
    ap.add_argument('--thin', type=float, default=2)
    ap.add_argument('--graphs', type=int, default=None)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, CAM)
    import torch
    from datasets.mis import Dataset
    d = torch.load(args.label_set, weights_only=False)
    sets = d['sets'][:args.graphs] if args.graphs else d['sets']
    ds = Dataset(args.train_path)
    a = {k: getattr(args, k) for k in ('r', 'T', 'M', 'chains', 'seed')}
    a['burn'] = int(args.burn); a['thin'] = max(1, int(args.thin))
    jobs = []
    for i, S in enumerate(sets):
        ei = ds[i].edge_index.numpy()
        keep = ei[0] < ei[1]
        assert ds[i].num_nodes == S.shape[1]
        jobs.append((i, (ei[0][keep], ei[1][keep]), S.shape[1], S.numpy().astype(np.int8), a))
    res = [None] * len(jobs)
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for i, S, st in pool.imap_unordered(graph_samples, jobs, chunksize=4):
            res[i] = (torch.from_numpy(S.astype(np.uint8)), st)
    stats = [r[1] for r in res]
    meta = dict(vars(args), stats=stats, source_label_set=args.label_set)
    files = d.get('files', [])[:len(res)]
    torch.save({'files': files, 'sets': [r[0] for r in res], 'meta': meta}, args.out)
    mean = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    print(f'wrote {args.out}: {len(res)} graphs, {args.M} samples each; means {mean}', flush=True)


if __name__ == '__main__':
    main()
