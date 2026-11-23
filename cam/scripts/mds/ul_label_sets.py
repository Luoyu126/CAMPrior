"""MDS label sets from an unsupervised solver's samples (local addition, UPSTREAM.md item 64).

Input: the output of scripts/mis/ul_samples.py run on the training graphs with --problem mds (unique decoded sets with
counts and sizes per graph). Keeps, per graph, all distinct sets of the smallest size found (the MDS counterpart of the
MIS 'all distinct largest sets' DiffUCO labels), in the label-set format of utils/label_sets.py. If an ILP label file
(scripts/mds/ilp_labels.py) is given, also reports how far the kept size is from the ILP optimum.
    python cam/scripts/mds/ul_label_sets.py <ul_samples.pt> <out.pt> [--ilp ilp_labels.pt] [--max_sets 0]
"""
import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('samples'); ap.add_argument('out')
    ap.add_argument('--ilp', default=None)
    ap.add_argument('--max_sets', type=int, default=0, help='keep at most this many per graph (random), 0 = all')
    a = ap.parse_args()
    d = torch.load(a.samples, weights_only=False)
    rng = np.random.default_rng(0)
    sets, sizes = [], []
    for s in d['samples']:
        u = s['sets'].to(torch.uint8); sz = u.sum(1)
        keep = u[sz == sz.min()]
        if a.max_sets > 0 and keep.shape[0] > a.max_sets:
            keep = keep[torch.from_numpy(rng.choice(keep.shape[0], a.max_sets, replace=False))]
        sets.append(keep); sizes.append(int(sz.min()))
    out = {'files': d['files'], 'sets': sets, 'k': sizes,
           'meta': {'source': a.samples, 'rule': 'all distinct smallest sets', 'max_sets': a.max_sets}}
    m = np.array([x.shape[0] for x in sets])
    msg = f'{len(sets)} graphs: smallest size {np.mean(sizes):.3f}, sets/graph mean {m.mean():.1f} median {np.median(m):.0f}'
    if a.ilp:
        il = torch.load(a.ilp, weights_only=False)
        assert il['files'] == d['files'], 'ILP labels are for other graphs'
        k = np.array(il['k']); s = np.array(sizes)
        msg += f'; ILP optimum {k.mean():.3f}, UL - ILP {np.mean(s - k):+.3f}, UL optimal on {np.mean(s == k):.3f}'
        out['ilp_k'] = il['k']
    torch.save(out, a.out)
    print(msg)


if __name__ == '__main__':
    main()
