"""Small random graphs for stage 2B of docs/20261125/plan.md (cross-graph DASBS checked against the exact SB).

Family: n uniform in [n_min, n_max], Erdős–Rényi with edge probability uniform in [p_min, p_max]; train / valid / test
drawn from disjoint seed ranges, and a test graph is redrawn if it is isomorphic to a training graph (so the test set
is held out). Also writes `wl/`: pairs of graphs that 1-WL (and hence a message-passing GNN without node features)
cannot tell apart but whose exact SB controllers differ (math_route2.md §8), e.g. C_n vs two disjoint cycles.
Files are networkx gpickles named idx_<k>.gpickle, the layout of datasets/mis.py.

Usage: python scripts/sb/gen_small_graphs.py --out DIR [--train 2000 --valid 64 --test 64]
"""
import argparse
import os

import networkx as nx
import numpy as np


def er(rng, n_min, n_max, p_min, p_max):
    n = int(rng.integers(n_min, n_max + 1))
    p = float(rng.uniform(p_min, p_max))
    G = nx.gnp_random_graph(n, p, seed=int(rng.integers(2 ** 31)))
    return G


def wl_pairs():
    """(name, G1, G2): 2-regular graphs on the same number of nodes, one cycle vs a union of cycles."""
    out = []
    for n, parts in ((8, (4, 4)), (9, (3, 3, 3)), (10, (4, 6)), (10, (5, 5)), (12, (6, 6)), (12, (4, 4, 4))):
        G2 = nx.disjoint_union_all([nx.cycle_graph(k) for k in parts])
        out.append((f'C{n}_vs_' + '+'.join(f'C{k}' for k in parts), nx.cycle_graph(n), G2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--train', type=int, default=2000)
    ap.add_argument('--valid', type=int, default=64)
    ap.add_argument('--test', type=int, default=64)
    ap.add_argument('--n_min', type=int, default=8)
    ap.add_argument('--n_max', type=int, default=12)
    ap.add_argument('--p_min', type=float, default=0.2)
    ap.add_argument('--p_max', type=float, default=0.45)
    args = ap.parse_args()
    sets = {}
    for split, count, seed in (('train', args.train, 0), ('valid', args.valid, 1), ('test', args.test, 2)):
        rng = np.random.default_rng(seed)
        graphs = []
        while len(graphs) < count:
            G = er(rng, args.n_min, args.n_max, args.p_min, args.p_max)
            if split != 'train' and any(nx.faster_could_be_isomorphic(G, H) and nx.is_isomorphic(G, H)
                                        for H in sets['train'] if H.number_of_nodes() == G.number_of_nodes()):
                continue
            graphs.append(G)
        sets[split] = graphs
        d = os.path.join(args.out, split)
        os.makedirs(d, exist_ok=True)
        for k, G in enumerate(graphs):
            nx.write_gpickle(G, os.path.join(d, f'idx_{k}.gpickle'))
        sizes = [G.number_of_nodes() for G in graphs]
        print(f'{split}: {len(graphs)} graphs, n {min(sizes)}-{max(sizes)}, '
              f'mean edges {np.mean([G.number_of_edges() for G in graphs]):.1f}')
    d = os.path.join(args.out, 'wl')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'pairs.txt'), 'w') as f:
        for k, (name, G1, G2) in enumerate(wl_pairs()):
            nx.write_gpickle(G1, os.path.join(d, f'idx_{2 * k}.gpickle'))
            nx.write_gpickle(G2, os.path.join(d, f'idx_{2 * k + 1}.gpickle'))
            f.write(f'{2 * k} {2 * k + 1} {name}\n')
    print('wl: written', len(wl_pairs()), 'pairs')


if __name__ == '__main__':
    main()
