"""Equalise the number of labels per graph across label sources, so that a comparison of sources (e.g. KaMIS
vs DiffUCO) is not confounded by how many distinct labels each one has: for every graph, each source keeps
M = min over sources of its number of sets, chosen uniformly at random. Local addition, UPSTREAM.md item 58.

    python cam/scripts/mis/match_label_counts.py OUT_DIR NAME1=SETS1.pt NAME2=SETS2.pt [...] [--seed 0]
writes OUT_DIR/matched_NAME.pt for each source.
"""
import argparse
import os

import numpy as np
import torch


def hamming(t):
    if t.shape[0] < 2:
        return 0.0
    f = t.float()
    return ((f[:, None] - f[None]).abs().sum(-1).sum() / (t.shape[0] * (t.shape[0] - 1))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('sources', nargs='+')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    src = dict(s.split('=', 1) for s in a.sources)
    data = {k: torch.load(v, weights_only=False) for k, v in src.items()}
    files = next(iter(data.values()))['files']
    for k, d in data.items():
        assert d['files'] == files, f'{k}: file order differs'
    g = torch.Generator().manual_seed(a.seed)
    out = {k: [] for k in data}
    for i in range(len(files)):
        m = min(d['sets'][i].shape[0] for d in data.values())
        for k, d in data.items():
            t = d['sets'][i]
            out[k].append(t[torch.randperm(t.shape[0], generator=g)[:m]])
    os.makedirs(a.out_dir, exist_ok=True)
    for k, sets in out.items():
        meta = {'from': src[k], 'rule': f'random subset, per graph as many as the smallest of {sorted(src)}',
                'seed': a.seed}
        torch.save({'files': files, 'sets': sets, 'meta': meta}, os.path.join(a.out_dir, f'matched_{k}.pt'))
        print(f'{k}: sets/graph {np.mean([t.shape[0] for t in sets]):.1f}, '
              f'size {np.mean([float(t[0].sum()) for t in sets]):.3f}, '
              f'mean pairwise Hamming {np.mean([hamming(t) for t in sets[:500]]):.2f}')


if __name__ == '__main__':
    main()
