"""Training-dynamics metrics of route 1' (DSBM, docs/20261128/plan_route1prime.md §4) from pair traces on training
graphs (solvers/difusco.py CAM_SB_PAIR_TRACE with CAM_SB_TRACE_NOSHARE=1; one JSON line per graph with its dataset
index, the chains' starts and finals as hex bit strings in loader node order).

Forward network (generation, trace of uniform starts z -> decoded solutions y), against the target pi_B (label set +
weights):
  size        mean size of the decoded solutions
  in_labels   fraction of solutions that are rows of the label set
  d_near      mean Hamming distance to the nearest label
  tv_near     per graph, total variation between the frequencies of the nearest label (ties: lower index) among
              the J chains and pi_B, averaged over graphs; tv_floor is the same statistic for J i.i.d. draws from
              pi_B (the finite-sample floor; tv_near - tv_floor is the excess)
  cost        mean Hamming distance d(z, y) of the pairs
Backward network (noising, trace of labels y -> sampled starts z), against Bernoulli(0.5)^n independent of y:
  density     mean fraction of ones of z (0.5)
  conflict    mean fraction of edges with both ends 1 (0.25)
  agree       mean fraction of nodes with z_i = y_i (0.5)
  ber_gap     mean over (graph, chain) of |density - 0.5| + |conflict - 0.25| + |agree - 0.5| (Shengyu's SBPD
              statistic, solvers/sb_utils.py terminal_stats in CAM4CO; it has a finite-n floor, see bgap_floor, the
              same statistic for z ~ Bernoulli(0.5)^n)
  cost        mean Hamming distance d(y, z) of the pairs

Usage: python scripts/sb/dsbm_metrics.py [--fwd TRACE] [--bwd TRACE] --label_set LS [--temp T] [--graphs 256]
       [--train_path DIR] [--out FILE.json]   (prints one JSON object; --out also writes it)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils import sb_coupling as sb                              # noqa: E402

TRAIN = '/usr1/shengyuf/MIS/RB_small_train'


def bits(h, n):
    return np.array(list(bin(int(h, 16))[2:].zfill(n)), dtype=np.int8)


def read_trace(path, graphs):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r['idx'] < graphs:
            out[r['idx']] = (np.stack([bits(h, r['n']) for h in r['start']]), np.stack([bits(h, r['n']) for h in r['final']]))
    return out


def nearest(X, S):
    """Index of the nearest row of S (M, n) for every row of X (J, n) (ties: lower index) and the distance."""
    D = (X[:, None, :] != S[None]).sum(-1)
    return D.argmin(1), D.min(1)


def tv_counts(idx, p):
    c = np.bincount(idx, minlength=len(p)) / len(idx)
    return 0.5 * np.abs(c - p).sum()


def fwd_metrics(trace, sets, temp, seed=0):
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ('size', 'in_labels', 'd_near', 'tv_near', 'tv_floor', 'cost')}
    for i, (Z, Y) in sorted(trace.items()):
        S = sets[i].numpy().astype(np.int8)
        p = sb.label_log_weights(sets[i], temp).exp().numpy()
        k, d = nearest(Y, S)
        acc['size'].append(Y.sum(1).mean())
        acc['in_labels'].append((d == 0).mean())
        acc['d_near'].append(d.mean())
        acc['tv_near'].append(tv_counts(k, p))
        acc['tv_floor'].append(tv_counts(rng.choice(len(p), size=len(Y), p=p), p))
        acc['cost'].append((Z != Y).sum(1).mean())
    return {k: float(np.mean(v)) for k, v in acc.items()} | {'graphs': len(trace)}


def bwd_metrics(trace, edges, seed=0):
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ('density', 'conflict', 'agree', 'ber_gap', 'bgap_floor', 'cost')}
    for i, (Y, Z) in sorted(trace.items()):          # backward trace: start y (a label), final z
        e0, e1 = edges[i]
        def stats(z):
            dens = z.mean(1); conf = (z[:, e0] * z[:, e1]).mean(1); agr = (z == Y).mean(1)
            return dens, conf, agr, np.abs(dens - 0.5) + np.abs(conf - 0.25) + np.abs(agr - 0.5)
        dens, conf, agr, gap = stats(Z)
        gap0 = stats(rng.integers(0, 2, Z.shape).astype(np.int8))[3]
        acc['density'].append(dens.mean()); acc['conflict'].append(conf.mean()); acc['agree'].append(agr.mean())
        acc['ber_gap'].append(gap.mean()); acc['bgap_floor'].append(gap0.mean()); acc['cost'].append((Z != Y).sum(1).mean())
    return {k: float(np.mean(v)) for k, v in acc.items()} | {'graphs': len(trace)}


def graph_edges(train_path, idxs):
    """Undirected edges (each once) of training graphs in loader node order (datasets.mis.Dataset)."""
    from datasets.mis import Dataset
    ds = Dataset(train_path)
    out = {}
    for i in idxs:
        ei = ds[i].edge_index.numpy()
        keep = ei[0] < ei[1]
        out[i] = (ei[0][keep], ei[1][keep])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fwd', default=None)
    ap.add_argument('--bwd', default=None)
    ap.add_argument('--label_set', required=True)
    ap.add_argument('--temp', default=None)
    ap.add_argument('--graphs', type=int, default=256)
    ap.add_argument('--train_path', default=TRAIN)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    res = {}
    if args.fwd:
        sets = torch.load(args.label_set, weights_only=False)['sets']
        res['fwd'] = fwd_metrics(read_trace(args.fwd, args.graphs), sets, sb.parse_temp(args.temp), args.seed)
    if args.bwd:
        tr = read_trace(args.bwd, args.graphs)
        res['bwd'] = bwd_metrics(tr, graph_edges(args.train_path, sorted(tr)), args.seed)
    s = json.dumps(res)
    print(s)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(s + '\n')


if __name__ == '__main__':
    main()
