"""Minimum dominating set labels by integer programming (local addition, UPSTREAM.md item 64).

For every graph of a gpickle directory: min sum x_i s.t. x_i + sum_{j in N(i)} x_j >= 1, x binary, solved by HiGHS
(scipy.optimize.milp) with a time limit. The first set is the solver's optimum (the supervised baseline label, as
KaMIS / Gurobi for MIS). With --extra R, R more optimal sets: random linear objectives over the optimal face
(sum x = k), so the stored sets are random optima, not the solver's neighbours of the first one.
Node order is networkx insertion order, as from_networkx in datasets/mcut.py (the MDS dataset).
Output (torch.save), in the label-set format of utils/label_sets.py:
    {'files': [basename, ...] sorted, 'sets': [uint8 (M_i, n_i), first = solver optimum], 'k': [...],
     'proven': [bool: solver reported optimality for the first set], 'meta': {...}}

    python cam/scripts/mds/ilp_labels.py <graph dir> <out.pt> [--extra 0] [--time_limit 60] [--procs 48] [--n -1]
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

ARGS = None


def solve(c, cons, n, tl):
    r = milp(c, constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1),
             options={'time_limit': tl, 'disp': False})
    return (None if r.x is None else np.rint(r.x).astype(np.uint8)), r.status == 0


def job(f):
    g = nx.read_gpickle(f)
    g = nx.convert_node_labels_to_integers(g)          # insertion order, as from_networkx
    g.remove_edges_from(nx.selfloop_edges(g))
    n = g.number_of_nodes()
    A = nx.to_scipy_sparse_array(g, nodelist=range(n), format='csr')
    cons = [LinearConstraint(A + sp.eye(n), 1, np.inf)]
    x, proven = solve(np.ones(n), cons, n, ARGS.time_limit)
    assert x is not None, f'no feasible set for {f}'
    k = int(x.sum()); sets = [x]
    if ARGS.extra > 0:
        face = cons + [LinearConstraint(np.ones((1, n)), k, k)]
        rng = np.random.default_rng(int.from_bytes(os.path.basename(f).encode()[-6:], 'little'))
        seen = {x.tobytes()}
        for _ in range(ARGS.extra):
            y, _ = solve(rng.random(n), face, n, ARGS.time_limit)
            if y is not None and y.tobytes() not in seen:
                seen.add(y.tobytes()); sets.append(y)
    return os.path.basename(f), torch.from_numpy(np.stack(sets)), k, proven


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('graph_dir'); ap.add_argument('out')
    ap.add_argument('--extra', type=int, default=0)
    ap.add_argument('--time_limit', type=float, default=60)
    ap.add_argument('--procs', type=int, default=48)
    ap.add_argument('--n', type=int, default=-1)
    ARGS = ap.parse_args()
    files = sorted(glob.glob(ARGS.graph_dir + '/*gpickle'))
    if ARGS.n > 0:
        files = files[:ARGS.n]
    t0, res = time.time(), {}
    with Pool(ARGS.procs) as p:
        for j, (name, s, k, pr) in enumerate(p.imap_unordered(job, files, chunksize=4)):
            res[name] = (s, k, pr)
            if (j + 1) % 250 == 0:
                print(f'# {j + 1}/{len(files)} {(time.time() - t0) / 60:.1f} min', flush=True)
    names = [os.path.basename(f) for f in files]
    out = {'files': names, 'sets': [res[m][0] for m in names], 'k': [res[m][1] for m in names],
           'proven': [res[m][2] for m in names],
           'meta': {'solver': 'HiGHS (scipy.optimize.milp)', 'time_limit': ARGS.time_limit, 'extra': ARGS.extra,
                    'graph_dir': ARGS.graph_dir}}
    torch.save(out, ARGS.out)
    m = np.array([s.shape[0] for s in out['sets']])
    print(f'{len(names)} graphs: mean size {np.mean(out["k"]):.3f}, proven {np.mean(out["proven"]):.3f}, '
          f'sets/graph {m.mean():.2f}, {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
