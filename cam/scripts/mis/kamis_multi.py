"""Many KaMIS labels per training graph: redumis is run with K seeds per graph and the distinct sets of the
best size (together with the existing KaMIS label from <graphs>/anno when it is as large) are kept, in the
label-set format of diverse_labels.py ({'files', 'sets', 'meta'}) for --label_choice random / soft.
Local addition, UPSTREAM.md item 58.

On RB-[200-300] almost every seed returns a different maximum set (30-31 distinct of 32 on the first graphs).

    python cam/scripts/mis/kamis_multi.py --graphs DIR --out FILE --k 128 --time_limit 0.2 --procs 60 \
        [--files_order LABELSET.pt]
"""
import argparse
import glob
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_kamis import REDUMIS, metis  # noqa: E402


def solve(job):
    path, k, tl, tmp = job
    stem = os.path.basename(path)[:-len('.gpickle')]
    with open(path, 'rb') as f:
        G = pickle.load(f)
    assert sorted(G.nodes) == list(range(G.number_of_nodes()))
    gfile = os.path.join(tmp, f'{stem}.graph')
    with open(gfile, 'w') as f:
        f.write(metis(G))
    sols = []
    anno = os.path.join(os.path.dirname(path), 'anno', f'{stem}_unweighted.result')
    if os.path.exists(anno):
        sols.append([int(l) for l in open(anno) if l.strip()])
    for s in range(k):
        out = os.path.join(tmp, f'{stem}_{s}.out')
        subprocess.run([REDUMIS, gfile, '--output', out, '--seed', str(s), '--time_limit', str(tl)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        sols.append([int(l) for l in open(out) if l.strip()])
        os.remove(out)
    os.remove(gfile)
    x = torch.tensor(sols, dtype=torch.uint8)
    assert x.shape[1] == G.number_of_nodes()
    e = torch.tensor(list(G.edges), dtype=torch.long)
    if len(e):
        assert not (x[:, e[:, 0]] & x[:, e[:, 1]]).any(), f'{stem}: not independent'
    z = x.sum(1)
    top = torch.unique(x[z == z.max()], dim=0)
    return os.path.basename(path), top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=128)
    ap.add_argument('--time_limit', type=float, default=0.2)
    ap.add_argument('--procs', type=int, default=60)
    ap.add_argument('--files_order', default=None, help='label-set file whose file order to follow')
    ap.add_argument('--tmp', default='/usr0/home/shengyuf/yunyi/.cache/kamis_multi_tmp')
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    if a.files_order:
        files = torch.load(a.files_order, weights_only=False)['files']
    else:
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.graphs, '*.gpickle')))
    jobs = [(os.path.join(a.graphs, f), a.k, a.time_limit, a.tmp) for f in files]
    res, t0 = {}, time.time()
    with Pool(a.procs) as pool:
        for i, (f, top) in enumerate(pool.imap_unordered(solve, jobs, chunksize=4)):
            res[f] = top
            if (i + 1) % 200 == 0:
                print(f'# {i + 1}/{len(jobs)} {(time.time() - t0) / 60:.1f} min', flush=True)
    sets = [res[f] for f in files]
    meta = {'rule': f'distinct best sets of redumis over {a.k} seeds (time_limit {a.time_limit}) + anno label',
            'graphs': a.graphs}
    torch.save({'files': files, 'sets': sets, 'meta': meta}, a.out)
    m = np.array([t.shape[0] for t in sets])
    print(f'sets/graph mean {m.mean():.1f} median {np.median(m):.0f}, size {np.mean([float(t[0].sum()) for t in sets]):.3f}')


if __name__ == '__main__':
    main()
