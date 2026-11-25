"""Label perturbations for stage 1E of docs/20261125/plan.md (no training): statistics, and label-set files.

Perturbations of an independent set s on graph G (all return independent sets):
  swap     equal-size (1,1)-swap: pick uniformly a pair (u in s, v not in s) whose only neighbour in s is u, and
           replace u by v. Repeated `--steps` times (each step from the previous result), so the size is unchanged.
  delete   remove k selected nodes, k uniform in {1, 2}: a smaller feasible solution.
  repair   flip r random bits (r = `--flip_frac` * n), then repair: while some edge is violated, drop the selected
           node with the most selected neighbours (ties at random); then add free nodes (no selected neighbour) in
           increasing degree order (ties at random) until maximal.
Statistics per perturbation over `--graphs` training graphs and `--draws` draws per label: size relative to the label,
Hamming distance to the label, distinct results per graph, and the fraction of results that are themselves in the
KaMIS multi-solution set (all of the matched set are optimal).

  python scripts/sb/perturb_labels.py stats --graphs 400 --draws 32 > OUT.txt
  python scripts/sb/perturb_labels.py build --base single|multi --kind swap|delete|repair --per_label N --out FILE

`build` writes a label-set file in the layout of utils/label_sets.py ({'files', 'sets', 'meta'}; rows = the base
labels first, then their perturbations, duplicates removed); with --with_sizes the file also holds 'sizes'.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch

TRAIN = '/usr1/shengyuf/MIS/RB_small_train'
MULTI = '/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/matched_kamis_loaderorder.pt'


def load_graph(path):
    """Nodes indexed in G's insertion order, the order of datasets/mis.py and of the label-set files."""
    with open(path, 'rb') as f:
        G = pickle.load(f)
    nodes = list(G.nodes())
    pos = {v: j for j, v in enumerate(nodes)}
    nbrs = [np.array(sorted(pos[u] for u in G.neighbors(v) if u != v), dtype=np.int64) for v in nodes]
    return len(nodes), nbrs, nodes


def single_label(path, nodes):
    """The single KaMIS label (anno/, indexed by sorted node id), realigned to insertion order as datasets/mis.py."""
    stem = os.path.basename(path)[:-len('.gpickle')]
    x = np.loadtxt(os.path.join(os.path.dirname(path), 'anno', f'{stem}_unweighted.result'), dtype=np.int64)
    assert len(x) == len(nodes)
    rank = {v: j for j, v in enumerate(sorted(nodes))}
    return np.array([x[rank[v]] for v in nodes], dtype=np.uint8)


def is_independent(s, nbrs):
    return all(not s[nb].any() for i in np.flatnonzero(s) for nb in [nbrs[i]])


def sel_count(s, nbrs):
    """Number of selected neighbours of every node."""
    return np.array([int(s[nb].sum()) for nb in nbrs])


def swap(s, nbrs, rng, steps=1):
    s = s.copy()
    for _ in range(steps):
        c = sel_count(s, nbrs)
        cand = []
        for v in np.flatnonzero((s == 0) & (c == 1)):
            u = nbrs[v][s[nbrs[v]] == 1][0]
            cand.append((u, v))
        if not cand:
            break
        u, v = cand[rng.integers(len(cand))]
        s[u], s[v] = 0, 1
    return s


def delete(s, nbrs, rng):
    s = s.copy()
    on = np.flatnonzero(s)
    k = int(rng.integers(1, 3))
    s[rng.choice(on, size=min(k, len(on)), replace=False)] = 0
    return s


def repair(s, nbrs, rng, flip_frac):
    n = len(s)
    s = s.copy()
    r = max(1, int(round(flip_frac * n)))
    idx = rng.choice(n, size=r, replace=False)
    s[idx] = 1 - s[idx]
    deg = np.array([len(nb) for nb in nbrs])
    while True:
        c = sel_count(s, nbrs) * s
        if c.max() == 0:
            break
        worst = np.flatnonzero(c == c.max())
        s[worst[rng.integers(len(worst))]] = 0
    c = sel_count(s, nbrs)
    order = np.lexsort((rng.random(n), deg))
    for v in order:
        if s[v] == 0 and c[v] == 0:
            s[v] = 1
            c[nbrs[v]] += 1
    return s


def perturb(kind, s, nbrs, rng, args):
    if kind == 'swap':
        return swap(s, nbrs, rng, args.steps)
    if kind == 'delete':
        return delete(s, nbrs, rng)
    if kind == 'repair':
        return repair(s, nbrs, rng, args.flip_frac)
    raise ValueError(kind)


def stats(args):
    files = sorted(glob.glob(TRAIN + '/*gpickle'))
    multi = torch.load(MULTI, weights_only=False)
    assert [os.path.basename(f) for f in files] == multi['files'][:len(files)]
    rng = np.random.default_rng(args.seed)
    kinds = ['swap', 'delete', 'repair']
    rec = {k: dict(dsize=[], ham=[], distinct=[], in_multi=[], feasible=[]) for k in kinds}
    base = dict(ham_multi=[], m=[], single_in_multi=[])
    for gi in range(args.graphs):
        n, nbrs, nodes = load_graph(files[gi])
        lab = single_label(files[gi], nodes)
        assert is_independent(multi['sets'][gi].numpy()[0], nbrs)
        assert is_independent(lab, nbrs)
        M = multi['sets'][gi].numpy()
        keys = {r.tobytes() for r in M}
        base['m'].append(len(M))
        base['single_in_multi'].append(lab.tobytes() in keys)
        if len(M) > 1:
            d = (M[:, None, :] != M[None, :, :]).sum(-1)
            base['ham_multi'].append(d[np.triu_indices(len(M), 1)].mean())
        for k in kinds:
            outs = [perturb(k, lab, nbrs, rng, args) for _ in range(args.draws)]
            rec[k]['dsize'] += [int(o.sum()) - int(lab.sum()) for o in outs]
            rec[k]['ham'] += [int((o != lab).sum()) for o in outs]
            rec[k]['distinct'].append(len({o.tobytes() for o in outs}))
            rec[k]['in_multi'] += [o.tobytes() in keys for o in outs]
            if gi < 50:
                rec[k]['feasible'] += [is_independent(o, nbrs) for o in outs]
    print(f'graphs {args.graphs}, draws per label {args.draws}, swap steps {args.steps}, flip_frac {args.flip_frac}')
    print(f'KaMIS matched multi-set: mean solutions per graph {np.mean(base["m"]):.1f}, mean pairwise Hamming '
          f'{np.mean(base["ham_multi"]):.1f}, single label in multi-set {np.mean(base["single_in_multi"]):.3f}')
    for k in kinds:
        r = rec[k]
        ds = np.array(r['dsize'])
        hist = {int(v): float((ds == v).mean()) for v in np.unique(ds)}
        print(f'{k:7s} size change mean {ds.mean():+.3f} dist {hist} | Hamming mean {np.mean(r["ham"]):.2f} '
              f'(min {min(r["ham"])}, max {max(r["ham"])}) | distinct / {args.draws}: {np.mean(r["distinct"]):.1f} | '
              f'in KaMIS multi-set {np.mean(r["in_multi"]):.3f} | independent (first 50 graphs) '
              f'{np.mean(r["feasible"]):.3f}')
    sys.stdout.flush()


def build(args):
    files = sorted(glob.glob(TRAIN + '/*gpickle'))
    if args.limit > 0:
        files = files[:args.limit]
    multi = torch.load(MULTI, weights_only=False) if args.base == 'multi' else None
    rng = np.random.default_rng(args.seed)
    sets, sizes = [], []
    for gi, f in enumerate(files):
        n, nbrs, nodes = load_graph(f)
        base = multi['sets'][gi].numpy() if multi is not None else single_label(f, nodes)[None]
        rows, seen = [], set()
        for r in base:
            if r.tobytes() not in seen:
                seen.add(r.tobytes()); rows.append(r)
        for r in base:
            for _ in range(args.per_label):
                o = perturb(args.kind, r, nbrs, rng, args)
                if o.tobytes() not in seen:
                    seen.add(o.tobytes()); rows.append(o)
        S = np.stack(rows)
        if gi < 20:
            assert all(is_independent(r, nbrs) for r in S)
        sets.append(torch.from_numpy(S.astype(np.uint8)))
        sizes.append(torch.from_numpy(S.sum(1).astype(np.int64)))
        if gi % 500 == 0:
            print(f'{gi}/{len(files)} rows {len(S)}', flush=True)
    out = dict(files=[os.path.basename(f) for f in files], sets=sets,
               meta=dict(base=args.base, kind=args.kind, per_label=args.per_label, steps=args.steps,
                         flip_frac=args.flip_frac, seed=args.seed, multi=MULTI if multi is not None else None))
    if args.with_sizes:
        out['sizes'] = sizes
    torch.save(out, args.out)
    m = np.array([len(s) for s in sets])
    print(f'wrote {args.out}: {len(sets)} graphs, rows per graph mean {m.mean():.1f} (min {m.min()}, max {m.max()}), '
          f'mean size {np.mean([float(s.float().mean()) for s in sizes]):.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['stats', 'build'])
    ap.add_argument('--graphs', type=int, default=400)
    ap.add_argument('--draws', type=int, default=32)
    ap.add_argument('--steps', type=int, default=1, help='swap: number of successive (1,1)-swaps')
    ap.add_argument('--flip_frac', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--base', choices=['single', 'multi'], default='single')
    ap.add_argument('--kind', choices=['swap', 'delete', 'repair'], default='swap')
    ap.add_argument('--per_label', type=int, default=8)
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--with_sizes', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    stats(args) if args.mode == 'stats' else build(args)


if __name__ == '__main__':
    main()
