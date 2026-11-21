"""Many LKH-3 tours per TSP instance (local addition, UPSTREAM.md item 61).

For every line of a DIFUSCO-format instance file, run LKH-3 with K seeds (one run each, MAX_TRIALS trials,
coordinates x 1e6 rounded, EUC_2D) and keep the distinct tours sorted by length, in the tour-set format of
utils/tour_sets.py. Also writes OUT_TXT: the instance file with each line's shortest tour as its label.
With MAX_TRIALS=1 a run takes ~0.3 s on TSP-500 and seeds give different near-optimal tours; with 500 trials
every seed returns the same tour (probe on the TSP-500 test set).

LKH binary: LKH-3.0.8 built from the elkai-2.0.1 sdist with a main reading `LKH <par file> <problem file>`
and printing the 1-based tour (tools/LKH-3.0.8/LKH); the par file sets PROBLEM_FILE = :stdin:.

    python cam/scripts/tsp/lkh_labels.py IN.txt OUT_SETS.pt OUT_TXT --k 32 --trials 1 --procs 64
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils.tour_sets import distinct_sorted, parse_points, points_digest  # noqa: E402

LKH = os.environ.get('LKH_BIN', '/usr0/home/shengyuf/yunyi/tools/LKH-3.0.8/LKH')
TMP = os.environ.get('LKH_TMP', '/usr0/home/shengyuf/yunyi/.cache/lkh_tmp')


def lkh(points, seed, trials, stem):
    prob, par = f'{TMP}/{stem}.tsp', f'{TMP}/{stem}.par'
    with open(prob, 'w') as h:
        h.write(f'TYPE : TSP\nDIMENSION : {len(points)}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n')
        for i, (x, y) in enumerate(points):
            h.write(f'{i + 1} {int(round(x * 1e6))} {int(round(y * 1e6))}\n')
        h.write('EOF\n')
    with open(par, 'w') as h:
        h.write(f'PROBLEM_FILE = :stdin:\nRUNS = 1\nMAX_TRIALS = {trials}\nSEED = {seed}\nTRACE_LEVEL = 0\n')
    out = subprocess.run([LKH, par, prob], capture_output=True, text=True, check=True).stdout.strip().split('\n')
    os.remove(prob); os.remove(par)
    return np.array([int(v) - 1 for v in out[-1].split()])


def solve(job):
    i, line, k, trials = job
    p = parse_points(line)
    tours = [lkh(p, s + 1, trials, f'{os.getpid()}_{i}') for s in range(k)]
    t, l = distinct_sorted(p, tours)
    return i, t, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out_sets'); ap.add_argument('out_txt')
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--trials', type=int, default=1)
    ap.add_argument('--procs', type=int, default=64)
    a = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    lines = open(a.inp).read().splitlines()
    res, t0 = {}, time.time()
    with Pool(a.procs) as pool:
        for n, (i, t, l) in enumerate(pool.imap_unordered(solve, [(i, l, a.k, a.trials) for i, l in enumerate(lines)],
                                                          chunksize=2)):
            res[i] = (t, l)
            if (n + 1) % 500 == 0:
                print(f'# {n + 1}/{len(lines)} {(time.time() - t0) / 60:.1f} min', flush=True)
    tours = [res[i][0] for i in range(len(lines))]; lengths = [res[i][1] for i in range(len(lines))]
    meta = {'source': 'LKH-3', 'rule': f'distinct tours of {a.k} seeds, RUNS=1, MAX_TRIALS={a.trials}', 'inp': a.inp}
    torch.save({'points_digest': points_digest(lines), 'tours': tours, 'lengths': lengths, 'meta': meta}, a.out_sets)
    with open(a.out_txt, 'w') as f:
        for line, t in zip(lines, tours):
            f.write(line.split(' output ')[0].strip() + ' output ' +
                    ' '.join(str(int(v) + 1) for v in t[0].tolist() + [t[0, 0].item()]) + '\n')
    m = np.array([t.shape[0] for t in tours])
    print(f'distinct tours/instance mean {m.mean():.1f} min {m.min()} median {np.median(m):.0f}; '
          f'best length {np.mean([l[0].item() for l in lengths]):.4f}, '
          f'mean length {np.mean([l.mean().item() for l in lengths]):.4f}, {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
