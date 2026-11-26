"""Neighbourhoods of the KaMIS optima, for Boltzmann-weighted end distributions pi1 of the route-1 SB
(docs/20261126/plan_endpoint.md). No training.

All states are independent sets of a training graph, nodes in the loader (insertion) order of datasets/mis.py, the
order of the *_loaderorder label files. Moves on an independent set x (cnt_v = number of selected neighbours of v):
  swap   (1,1)-swap: pick uniformly a free-of-others node v not in x with cnt_v = 1, put it in and take out its single
         selected neighbour u. Size unchanged; repeated swaps walk over a plateau of equally large solutions.
  kick   pick uniformly a node v not in x with cnt_v >= 2, put it in and take out its cnt_v selected neighbours, then
         local search (below). A perturbation of one node: a nearby local optimum, usually smaller by cnt_v - 1 minus
         what the local search adds back.
  local search  until none applies: add any free node (cnt = 0), or a (1,2)-swap (take out u, put in two
         non-adjacent nodes whose only selected neighbour is u; size + 1). The result is maximal and (1,2)-swap-free,
         the local optimum of Andrade, Resende & Werneck (2012, ARW), the local search inside KaMIS's ILS.
Subcommands
  stats   per graph (first --graphs training graphs): plateau walk from every KaMIS label (how many distinct optima,
          whether any (1,2)-improvement exists), ILS (--ils iterations of kick + local search, accept if not smaller;
          does anything beat KaMIS?), and the kick neighbourhood (size, Hamming distance to the nearest optimum).
  build   a label-set file ({'files', 'sets', 'sizes', 'meta'}, layout of utils/label_sets.py) per training graph:
          the KaMIS labels, + plateau-walk optima up to --max_opt in total, + up to --kicks distinct kick results per
          optimum (--kick_from optima used), all distinct. Any solution larger than KaMIS found on the way replaces
          the optimum set (reported). pi1 weights exp(|s| / T) are applied later (precompute_logg.py --weight_temp).
  python scripts/sb/neighborhood_labels.py stats --graphs 200 > OUT.txt
  python scripts/sb/neighborhood_labels.py build --max_opt 256 --kicks 0 --out FILE
"""
import argparse
import glob
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

TRAIN = '/usr1/shengyuf/MIS/RB_small_train'
K128 = '/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/k128_loaderorder.pt'


def load_adj(path):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    nodes = list(G.nodes())
    pos = {v: j for j, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=bool)
    for a, b in G.edges():
        if a != b:
            A[pos[a], pos[b]] = A[pos[b], pos[a]] = True
    return A


class State:
    def __init__(self, A, x):
        self.A = A
        self.x = x.astype(bool).copy()
        self.cnt = A[:, self.x].sum(1).astype(np.int32)
        assert not (self.x & (self.cnt > 0)).any(), 'not independent'

    def put(self, v):
        self.x[v] = True
        self.cnt += self.A[v]

    def take(self, u):
        self.x[u] = False
        self.cnt -= self.A[u]

    def size(self):
        return int(self.x.sum())

    def key(self):
        return np.packbits(self.x).tobytes()

    def swap(self, rng):
        cand = np.flatnonzero(~self.x & (self.cnt == 1))
        if len(cand) == 0:
            return False
        v = cand[rng.integers(len(cand))]
        u = np.flatnonzero(self.A[v] & self.x)[0]
        self.take(u)
        self.put(v)
        return True

    def local_search(self, rng):
        while True:
            free = np.flatnonzero(~self.x & (self.cnt == 0))
            if len(free):
                self.put(free[rng.integers(len(free))])
                continue
            if not self.improve_12(rng):
                return

    def improve_12(self, rng):
        """One (1,2)-swap if any exists (random u order). True if applied."""
        tight = ~self.x & (self.cnt == 1)
        for u in rng.permutation(np.flatnonzero(self.x)):
            t = np.flatnonzero(tight & self.A[u])
            if len(t) < 2:
                continue
            sub = self.A[np.ix_(t, t)]
            free_pairs = np.argwhere(~sub & ~np.eye(len(t), dtype=bool))
            if len(free_pairs):
                i, j = free_pairs[rng.integers(len(free_pairs))]
                self.take(u)
                self.put(t[i])
                self.put(t[j])
                return True
        return False

    def kick(self, rng):
        cand = np.flatnonzero(~self.x & (self.cnt >= 2))
        if len(cand) == 0:
            return False
        v = cand[rng.integers(len(cand))]
        for u in np.flatnonzero(self.A[v] & self.x):
            self.take(u)
        self.put(v)
        self.local_search(rng)
        return True


def plateau(A, labels, max_opt, walk, rng):
    """Distinct optima reachable by (1,1)-swap walks from the labels; also any larger set met on the way."""
    seen = {}
    best = labels[0].sum()
    for x in labels:
        seen[np.packbits(x.astype(bool)).tobytes()] = x.astype(bool)
    bigger = []
    starts = list(seen.values())
    i = 0
    while len(seen) < max_opt and i < 4 * max_opt:
        st = State(A, starts[i % len(starts)])
        i += 1
        for _ in range(walk):
            if not st.swap(rng):
                break
            st.local_search(rng)                  # only acts if the swap opened a free node or a (1,2)-swap
            if st.size() > best:
                bigger.append(st.x.copy())
                break
            k = st.key()
            if k not in seen:
                seen[k] = st.x.copy()
                if len(seen) >= max_opt:
                    break
    return list(seen.values()), bigger


def ils(A, x0, iters, rng):
    """Kick + local search, accept if not smaller (ARW-style); returns the best size found."""
    st = State(A, x0)
    st.local_search(rng)
    best = st.size()
    for _ in range(iters):
        keep = (st.x.copy(), st.cnt.copy())
        if not st.kick(rng):
            break
        if st.size() < keep[0].sum():
            st.x, st.cnt = keep
        best = max(best, st.size())
    return best


def kick_neighbours(A, optima, per_opt, rng, tries_factor=4):
    out = {}
    for x in optima:
        got = 0
        for _ in range(tries_factor * per_opt):
            st = State(A, x)
            st.kick(rng)
            k = st.key()
            if k not in out:
                out[k] = st.x.copy()
                got += 1
                if got >= per_opt:
                    break
    return list(out.values())


def stats_job(job):
    idx, path, labels, walk, max_opt, ils_iters, seed = job
    rng = np.random.default_rng(seed + idx)
    A = load_adj(path)
    labels = labels.numpy().astype(bool)
    kamis = int(labels[0].sum())
    t0 = time.time()
    has12 = sum(State(A, x).improve_12(np.random.default_rng(0)) for x in labels[:8])
    opt, bigger = plateau(A, labels, max_opt, walk, rng)
    best_ils = max(ils(A, labels[j % len(labels)], ils_iters, rng) for j in range(4)) if ils_iters else kamis
    nb = kick_neighbours(A, opt[:16], 8, rng)
    O = np.stack(opt).astype(np.int16)
    sizes = np.array([x.sum() for x in nb])
    ham = np.array([int(np.abs(O - x.astype(np.int16)).sum(1).min()) for x in nb])
    return dict(idx=idx, n=len(A), kamis=kamis, labels=len(labels), plateau=len(opt), bigger=len(bigger),
                has12=has12, ils_best=best_ils, kick_dsize=float((sizes - kamis).mean()),
                kick_frac_opt=float((sizes == kamis).mean()), kick_ham=float(ham.mean()), secs=time.time() - t0)


def build_job(job):
    idx, path, labels, walk, max_opt, kicks, kick_from, seed = job
    rng = np.random.default_rng(seed + idx)
    A = load_adj(path)
    labels = labels.numpy().astype(bool)
    opt, bigger = plateau(A, labels, max_opt, walk, rng) if max_opt > len(labels) else (list(labels), [])
    if bigger:                                    # a larger solution than KaMIS: it becomes the optimum set
        b = max(int(y.sum()) for y in bigger)
        opt, more = plateau(A, np.stack([y for y in bigger if y.sum() == b]), max_opt, walk, rng)
    near = kick_neighbours(A, opt[:kick_from], kicks, rng) if kicks else []
    seen, rows = set(), []
    for x in list(opt) + near:
        k = np.packbits(x).tobytes()
        if k not in seen:
            seen.add(k)
            rows.append(x)
    X = torch.from_numpy(np.stack(rows).astype(np.uint8))
    e = np.argwhere(np.triu(A))
    assert not (X[:, e[:, 0]] & X[:, e[:, 1]]).any(), f'{path}: not independent'
    return idx, X, len(opt), len(bigger) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['stats', 'build'])
    ap.add_argument('--labels', default=K128)
    ap.add_argument('--graphs', type=int, default=-1)
    ap.add_argument('--walk', type=int, default=50)
    ap.add_argument('--max_opt', type=int, default=256)
    ap.add_argument('--ils', type=int, default=0)
    ap.add_argument('--kicks', type=int, default=0)
    ap.add_argument('--kick_from', type=int, default=64)
    ap.add_argument('--procs', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out')
    a = ap.parse_args()
    L = torch.load(a.labels, weights_only=False)
    files = [os.path.basename(f) for f in sorted(glob.glob(TRAIN + '/*gpickle'))]
    assert L['files'] == files
    N = len(files) if a.graphs < 0 else a.graphs
    paths = [os.path.join(TRAIN, f) for f in files[:N]]
    t0 = time.time()
    if a.cmd == 'stats':
        jobs = [(i, p, L['sets'][i], a.walk, a.max_opt, a.ils, a.seed) for i, p in enumerate(paths)]
        rows = []
        with Pool(a.procs) as pool:
            for r in pool.imap_unordered(stats_job, jobs):
                rows.append(r)
                print(r, flush=True)
        keys = ['labels', 'plateau', 'bigger', 'has12', 'kick_dsize', 'kick_frac_opt', 'kick_ham', 'secs']
        print('MEAN', {k: float(np.mean([r[k] for r in rows])) for k in keys})
        print('graphs with plateau == max_opt', sum(r['plateau'] >= a.max_opt for r in rows), '/', len(rows))
        print('graphs where a larger set than KaMIS was found: plateau walk', sum(r['bigger'] > 0 for r in rows),
              ' ILS', sum(r['ils_best'] > r['kamis'] for r in rows), ' ILS smaller than KaMIS',
              sum(r['ils_best'] < r['kamis'] for r in rows))
        return
    jobs = [(i, p, L['sets'][i], a.walk, a.max_opt, a.kicks, a.kick_from, a.seed) for i, p in enumerate(paths)]
    sets, n_opt, improved = [None] * N, [0] * N, 0
    with Pool(a.procs) as pool:
        for k, (i, X, no, imp) in enumerate(pool.imap_unordered(build_job, jobs, chunksize=4)):
            sets[i], n_opt[i] = X, no
            improved += imp
            if k % 500 == 0:
                print(f'{k}/{N} rows {X.shape[0]} optima {no} elapsed {time.time() - t0:.0f}s', flush=True)
    sizes = [X.sum(1).to(torch.int16) for X in sets]
    meta = dict(rule=f'neighborhood_labels.py build: KaMIS k128 labels + (1,1)-swap plateau walk to {a.max_opt} optima'
                     f' (walk {a.walk}) + {a.kicks} kick-local-search neighbours from each of the first {a.kick_from}'
                     f' optima; distinct; loader node order', labels=a.labels, graphs=TRAIN, seed=a.seed,
                optima_per_graph=n_opt, graphs_improved_over_kamis=improved)
    torch.save(dict(files=files[:N], sets=sets, sizes=sizes, meta=meta), a.out)
    print(f'done {N} graphs, mean rows {np.mean([X.shape[0] for X in sets]):.1f}, mean optima {np.mean(n_opt):.1f},'
          f' improved over KaMIS on {improved}, {time.time() - t0:.0f}s -> {a.out}')


if __name__ == '__main__':
    main()
