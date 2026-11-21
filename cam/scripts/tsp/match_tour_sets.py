"""Equal numbers of tours per instance across tour-set sources (local addition, UPSTREAM.md item 61), the TSP
counterpart of scripts/mis/match_label_counts.py: per instance every source keeps its M shortest distinct
tours, M = min(--max_keep, number of tours of each source), so that a comparison of sources (e.g. LKH-3 vs a
trained model's samples) is not confounded by how many labels each has. Tours are continuous in quality, so
the shortest are kept instead of a random subset.

    python cam/scripts/tsp/match_tour_sets.py OUT_DIR NAME1=SETS1.pt NAME2=SETS2.pt [...] --max_keep 8
writes OUT_DIR/matched_NAME.pt for each source.
"""
import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('sources', nargs='+')
    ap.add_argument('--max_keep', type=int, default=8)
    a = ap.parse_args()
    src = dict(s.split('=', 1) for s in a.sources)
    data = {k: torch.load(v, weights_only=False) for k, v in src.items()}
    digest = next(iter(data.values()))['points_digest']
    assert all(d['points_digest'] == digest for d in data.values()), 'sources are for different instances'
    n = len(next(iter(data.values()))['tours'])
    ms = [min([a.max_keep] + [d['tours'][i].shape[0] for d in data.values()]) for i in range(n)]
    os.makedirs(a.out_dir, exist_ok=True)
    for k, d in data.items():
        tours = [d['tours'][i][:m] for i, m in enumerate(ms)]
        lengths = [d['lengths'][i][:m] for i, m in enumerate(ms)]
        meta = {'from': src[k], 'rule': f'per instance the M shortest tours, M = min({a.max_keep}, counts of {sorted(src)})',
                'source_meta': d.get('meta')}
        torch.save({'points_digest': digest, 'tours': tours, 'lengths': lengths, 'meta': meta},
                   os.path.join(a.out_dir, f'matched_{k}.pt'))
        print(f'{k}: tours/instance {np.mean(ms):.2f} (min {min(ms)}), best length '
              f'{np.mean([l[0].item() for l in lengths]):.4f}, mean kept length {np.mean([l.mean().item() for l in lengths]):.4f}')


if __name__ == '__main__':
    main()
