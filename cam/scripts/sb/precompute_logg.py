"""Precompute the Sinkhorn potential log g of the static SB coupling (math_route1.md §2, §7) for every training graph
of a label set, for one cost strength omega. Read by solvers/difusco.py (--sb_coupling --sb_logg <file>).

The noise-end signal is the one DIFUSCO uses with --sb_omega omega (utils.sb_coupling.schedule_for_omega): a[T] =
tanh(omega / 2). Each graph: K uniform starts (int16 distances, sample_distances), float32 Monte Carlo Sinkhorn
(sinkhorn_distances) to a relative label-marginal error < tol on those starts; pi1 uniform over the graph's labels.

Usage: python scripts/sb/precompute_logg.py --omega 0.5 --out <file.pt> [--K 4194304]
"""
import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils import sb_coupling as sb                              # noqa: E402
from utils.label_sets import load_label_sets                     # noqa: E402
from utils.diffusion_schedulers import CategoricalDiffusion      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label_set', default='/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/matched_kamis_loaderorder.pt')
    ap.add_argument('--train_path', default='/usr1/shengyuf/MIS/RB_small_train')
    ap.add_argument('--omega', type=float, required=True)
    ap.add_argument('--K', type=int, default=1 << 22)
    ap.add_argument('--tol', type=float, default=1e-4)
    ap.add_argument('--num_tp', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    assert args.omega > 0
    dev = 'cuda'
    sets = load_label_sets(args.label_set, args.train_path)
    files = torch.load(args.label_set, weights_only=False)['files'][:len(sets)]
    a = sb.schedule_for_omega(CategoricalDiffusion(T=args.num_tp, schedule='cosine').Q_bar, args.omega)
    a_end = a[-1].item()
    log_gs, rels, iters = [], [], []
    torch.cuda.synchronize(); t0 = time.time()
    for i, s in enumerate(sets):
        M = s.shape[0]
        if M == 1:
            log_gs.append(torch.zeros(1, dtype=torch.float64)); rels.append(0.0); iters.append(0)
            continue
        labels = s.to(dev)
        gen = torch.Generator(dev).manual_seed(args.seed * 100003 + i)
        D = sb.sample_distances(args.K, labels, gen)
        pi1 = torch.full((M,), 1.0 / M, dtype=torch.float64, device=dev)
        lg, rel, it = sb.sinkhorn_distances(D, sb.omega(a_end), pi1, tol=args.tol)
        assert rel < args.tol, (i, rel)
        log_gs.append(lg.cpu()); rels.append(rel); iters.append(it)
        if i % 500 == 0:
            print(f'{i}/{len(sets)} M={M} iters={it} rel={rel:.1e} elapsed {time.time() - t0:.0f}s', flush=True)
    torch.cuda.synchronize(); elapsed = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'files': files, 'label_set': args.label_set, 'omega': args.omega, 'a_end': a_end, 'num_tp': args.num_tp,
                'K': args.K, 'tol': args.tol, 'seed': args.seed, 'log_g': log_gs, 'rel_err': rels, 'iters': iters,
                'seconds': elapsed}, args.out)
    print(f'done: {len(sets)} graphs in {elapsed:.0f}s ({elapsed / 60:.1f} min), max rel err {max(rels):.1e}, '
          f'max iters {max(iters)} -> {args.out}')


if __name__ == '__main__':
    main()
