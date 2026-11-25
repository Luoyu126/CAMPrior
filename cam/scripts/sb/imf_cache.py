"""Pair caches of the learned coupling (stage 1D of docs/20261125/plan.md, math_route1.md §12.4), read by
solvers/difusco.py --sb_pair_cache. A cache holds, per training graph, J pairs as two uint8 tensors (J, n): 'starts'
(the end the network does NOT predict) and 'labels' (the end it predicts). Forward network: start = z, label = y.
Backward network (--sb_reverse): start = y, label = z.

  independent  z ~ Bernoulli(0.5)^n, y uniform from the label set (iteration 0 of the IMF): writes both caches.
  from_trace   a pair trace (solvers/difusco.py CAM_SB_PAIR_TRACE on the training graphs, CAM_SB_TRACE_NOSHARE=1):
               --direction fwd: trace of the forward network (start z ~ pi0, final = decoded solution y) -> cache
               for the BACKWARD network (true z); --direction bwd: trace of the backward network (start = label y ~
               pi1, final = sampled z) -> cache for the FORWARD network (true y).
Reports the mean Hamming cost of the pairs; with --logg (precompute_logg.py output for the same omega) also the
closed-form SB coupling's expected cost E_{pi*} d(z, y) on the same graphs (Monte Carlo, --mc starts per graph),
plan.md 1D criterion 2.

  python scripts/sb/imf_cache.py independent --label_set LS --J 16 --out_fwd F.pt --out_bwd B.pt [--logg G.pt]
  python scripts/sb/imf_cache.py from_trace --trace T.jsonl --direction fwd|bwd --out F.pt [--label_set LS --logg G.pt]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils import sb_coupling as sb                              # noqa: E402

TRAIN = '/usr1/shengyuf/MIS/RB_small_train'


def bits(h, n):
    return np.array(list(bin(int(h, 16))[2:].zfill(n)), dtype=np.uint8)


def train_files():
    return [os.path.basename(f) for f in sorted(glob.glob(TRAIN + '/*gpickle'))]


def closed_form_cost(label_set, logg, mc, seed=0):
    """Per graph E_{pi*} d(z, y): z ~ Bernoulli(0.5)^n (mc samples), m ~ pi*(m | z)."""
    sets = torch.load(label_set, weights_only=False)['sets']
    d = torch.load(logg, weights_only=False)
    a_end = d['a_end']
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, (S, lg) in enumerate(zip(sets, d['log_g'])):
        S = S.double()
        z = torch.randint(0, 2, (mc, S.shape[1]), generator=g).double()
        P = sb.coupling_conditional(z, S, lg, a_end)                     # (mc, M)
        D = z.sum(1, keepdim=True) + S.sum(1)[None] - 2 * z @ S.T
        out.append(float((P * D).sum(1).mean()))
    return np.array(out), d['omega']


def save(path, starts, labels, meta):
    files = train_files()[:len(labels)]
    torch.save({'files': files, 'starts': starts, 'labels': labels, 'meta': meta}, path)
    cost = np.array([float((s != l).sum(1).float().mean()) for s, l in zip(starts, labels)])
    print(f'wrote {path}: {len(labels)} graphs, {np.mean([len(l) for l in labels]):.1f} pairs per graph, '
          f'mean Hamming cost {cost.mean():.3f}', flush=True)
    return cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['independent', 'from_trace'])
    ap.add_argument('--label_set', default='/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/matched_kamis_loaderorder.pt')
    ap.add_argument('--J', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--trace', default=None)
    ap.add_argument('--direction', choices=['fwd', 'bwd'], default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--out_fwd', default=None)
    ap.add_argument('--out_bwd', default=None)
    ap.add_argument('--logg', default=None)
    ap.add_argument('--mc', type=int, default=4096)
    args = ap.parse_args()
    meta = vars(args).copy()
    if args.mode == 'independent':
        sets = torch.load(args.label_set, weights_only=False)['sets']
        g = torch.Generator().manual_seed(args.seed)
        Z, Y = [], []
        for S in sets:
            Y.append(S[torch.randint(S.shape[0], (args.J,), generator=g)].to(torch.uint8))
            Z.append(torch.randint(0, 2, (args.J, S.shape[1]), generator=g).to(torch.uint8))
        cost = save(args.out_fwd, Z, Y, dict(meta, network='forward'))
        save(args.out_bwd, Y, Z, dict(meta, network='backward'))
    else:
        rows = [json.loads(l) for l in open(args.trace)]
        rows.sort(key=lambda r: r['idx'])
        assert [r['idx'] for r in rows] == list(range(len(rows))), 'trace must cover graphs 0..N-1 once'
        A = [torch.from_numpy(np.stack([bits(h, r['n']) for h in r['start']])) for r in rows]
        B = [torch.from_numpy(np.stack([bits(h, r['n']) for h in r['final']])) for r in rows]
        if args.direction == 'fwd':     # forward trace: start z, final y  ->  backward cache: start y, label z
            cost = save(args.out, B, A, dict(meta, network='backward'))
        else:                           # backward trace: start y, final z ->  forward cache: start z, label y
            cost = save(args.out, B, A, dict(meta, network='forward'))
    if args.logg:
        ref, om = closed_form_cost(args.label_set, args.logg, args.mc, args.seed)
        k = min(len(ref), len(cost))
        print(f'closed-form SB coupling (omega {om}): mean cost {ref[:k].mean():.3f} on the same {k} graphs; '
              f'pairs here {cost[:k].mean():.3f} (ratio {cost[:k].mean() / ref[:k].mean():.3f})', flush=True)


if __name__ == '__main__':
    main()
