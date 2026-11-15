"""Generate extra ER-[700-800] training graphs for the larger SL vs UL training pool.

Same sampler as the existing training set (DIMES / Boether et al. MIS benchmark framework,
data_generation/random_graph.py: ErdosRenyi): n uniform in [700, 800], networkx.erdos_renyi_graph(n,
0.15), pickled networkx Graph without node attributes, file name ER_700_800_0.15_<i>.gpickle.
Graph i is generated from its own seed (base seed + i), so any subset can be regenerated exactly.

    python cam/scripts/mis/gen_er700_graphs.py --out DIR --start 16384 --end 163840 --procs 16
"""
import argparse
import os
import pickle
import random
from multiprocessing import Pool

import networkx as nx

MIN_N, MAX_N, P = 700, 800, 0.15


def make(args):
    out, i, base_seed = args
    path = os.path.join(out, f'ER_{MIN_N}_{MAX_N}_{P}_{i}.gpickle')
    if os.path.exists(path):
        return 0
    rng = random.Random(base_seed + i)
    n = rng.randint(MIN_N, MAX_N)
    G = nx.erdos_renyi_graph(n, P, seed=rng.randrange(2**31))
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, required=True, help='first graph index (inclusive)')
    ap.add_argument('--end', type=int, required=True, help='last graph index (exclusive)')
    ap.add_argument('--seed', type=int, default=20261115)
    ap.add_argument('--procs', type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.out, i, a.seed) for i in range(a.start, a.end)]
    done = 0
    with Pool(a.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(make, jobs, chunksize=64)):
            done += r
            if (k + 1) % 5000 == 0:
                print(f'{k + 1}/{len(jobs)} checked, {done} written', flush=True)
    print(f'finished: {len(jobs)} checked, {done} written', flush=True)


if __name__ == '__main__':
    main()
