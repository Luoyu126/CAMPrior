"""KaMIS optima + solutions from another sampler, optionally polished by local search, as a label set for a
Boltzmann-weighted SB end distribution (docs/20261126/plan_endpoint.md §2). No training.

Sources of extra solutions per training graph (nodes in the loader order of datasets/mis.py):
  --ul FILE[,FILE]   raw samples of an unsupervised solver (scripts/mis/ul_samples.py: {'files', 'samples':
                     [{'sets', 'counts', ...}]}), e.g. DiffUCO's 1000 samples per graph
  --sl FILE          samples of a trained DIFUSCO model (scripts/mis/sl_samples.py: {graph idx: {'sets', ...}})
Each extra solution is (with --polish) run through the ARW local search of neighborhood_labels.State (add free nodes,
(1,2)-swaps until none: a maximal, (1,2)-swap-free local optimum); then kept if its size >= best - --max_gap (best =
the KaMIS size; anything larger than KaMIS is reported and kept). Among the kept extra solutions at most --max_extra
distinct ones per graph (the largest first, ties in source order). Output rows: the KaMIS labels first, then the
extra solutions not already among them. pi1 weights exp(|s| / T) are applied by precompute_logg.py --weight_temp.
  python scripts/sb/union_labels.py --ul A.pt,B.pt --polish --max_gap 1 --max_extra 128 --out FILE
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neighborhood_labels import K128, TRAIN, State, load_adj  # noqa: E402


def job(args):
    idx, path, base, extra, polish, max_gap, max_extra, seed = args
    rng = np.random.default_rng(seed + idx)
    A = load_adj(path)
    base = base.numpy().astype(bool)
    best = int(base[0].sum())
    seen = {np.packbits(x).tobytes() for x in base}
    cand = {}
    n_bigger = 0
    for x in extra.numpy().astype(bool):
        if polish:
            st = State(A, x)
            st.local_search(rng)
            x = st.x
        s = int(x.sum())
        n_bigger += s > best
        if s < best - max_gap:
            continue
        k = np.packbits(x).tobytes()
        if k not in seen and k not in cand:
            cand[k] = x.copy()
    ext = sorted(cand.values(), key=lambda x: -int(x.sum()))[:max_extra]
    X = torch.from_numpy(np.concatenate([base, np.stack(ext)]) if ext else base).to(torch.uint8)
    e = np.argwhere(np.triu(A))
    assert not (X[:, e[:, 0]] & X[:, e[:, 1]]).any(), f'{path}: not independent'
    return idx, X, n_bigger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=K128)
    ap.add_argument('--ul', default='')
    ap.add_argument('--sl', default='')
    ap.add_argument('--polish', action='store_true')
    ap.add_argument('--max_gap', type=int, default=1)
    ap.add_argument('--max_extra', type=int, default=128)
    ap.add_argument('--procs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    B = torch.load(a.base, weights_only=False)
    files = [os.path.basename(f) for f in sorted(glob.glob(TRAIN + '/*gpickle'))]
    assert B['files'] == files
    N = len(files)
    extra = [[] for _ in range(N)]
    for f in [f for f in a.ul.split(',') if f]:
        U = torch.load(f, weights_only=False)
        assert U['files'] == files, f
        for i, s in enumerate(U['samples']):
            extra[i].append(s['sets'])
    if a.sl:
        S = torch.load(a.sl, weights_only=False)
        assert sorted(S) == list(range(N))
        for i in range(N):
            extra[i].append(S[i]['sets'])
    extra = [torch.cat(e) if e else torch.zeros(0, B['sets'][i].shape[1], dtype=torch.uint8) for i, e in enumerate(extra)]
    t0 = time.time()
    jobs = [(i, os.path.join(TRAIN, files[i]), B['sets'][i], extra[i], a.polish, a.max_gap, a.max_extra, a.seed)
            for i in range(N)]
    sets, bigger = [None] * N, 0
    with Pool(a.procs) as pool:
        for k, (i, X, nb) in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            sets[i] = X
            bigger += nb > 0
            if k % 500 == 0:
                print(f'{k}/{N} rows {X.shape[0]} elapsed {time.time() - t0:.0f}s', flush=True)
    sizes = [X.sum(1).to(torch.int16) for X in sets]
    gaps = torch.cat([s - s[0] for s in sizes])
    meta = dict(rule=f'union_labels.py: base {a.base} + ul {a.ul} + sl {a.sl}; polish {a.polish}; size >= KaMIS - '
                     f'{a.max_gap}; <= {a.max_extra} extra per graph, largest first; loader node order',
                graphs=TRAIN, seed=a.seed, graphs_with_larger_than_kamis=bigger)
    torch.save(dict(files=files, sets=sets, sizes=sizes, meta=meta), a.out)
    print(f'done {N} graphs, mean rows {np.mean([X.shape[0] for X in sets]):.1f} (base '
          f'{np.mean([x.shape[0] for x in B["sets"]]):.1f}), size - KaMIS counts '
          f'{ {int(k): int(v) for k, v in zip(*torch.unique(gaps, return_counts=True))} }, graphs with a larger '
          f'solution than KaMIS {bigger}, {time.time() - t0:.0f}s -> {a.out}')


if __name__ == '__main__':
    main()
