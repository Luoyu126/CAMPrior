"""Paired per-graph comparison of two settings X and Y (each: one or more run directories, several globs joined by commas; per run, every evaluation
file matching TAG, e.g. 'anchor_renoise_t50_seed*'). Per graph: mean MIS size over Y's evaluations minus the mean over
X's; mean difference with a graph-level bootstrap 95% CI (10000 resamples, seed 0), and the fractions of graphs where
Y is better / worse. Graph identity is the file name, so both sides must be evaluated on the same set.

    python experiments/20261125_sbdifusco_rbsmall/paired.py 'X_RUN_GLOB' 'Y_RUN_GLOB' 'TAG' [--json]

Verdict used in README (plan.md 1C): positive if the CI lies above 0, negative if below, neutral otherwise.
"""
import csv
import glob
import json
import sys

import numpy as np


def arm(run_glob, tag):
    vals, files = [], None
    for r in sorted(set(sum((glob.glob(g) for g in run_glob.split(',')), []))):   # comma: several globs
        for f in sorted(glob.glob(f'{r}/{tag}_result.txt.per_graph.csv')):
            rows = sorted(csv.DictReader(open(f)), key=lambda x: x['file'])
            ff = [x['file'] for x in rows]
            assert files is None or ff == files, f
            files = ff
            vals.append(-np.array([float(x['obj']) for x in rows]))
    assert vals, f'no evaluation {tag} under {run_glob}'
    return np.mean(vals, 0), len(vals), files


def main():
    xg, yg, tag = sys.argv[1:4]
    x, nx, fx = arm(xg, tag)
    y, ny, fy = arm(yg, tag)
    assert fx == fy, 'different graph sets'
    d = y - x
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    verdict = 'positive' if lo > 0 else 'negative' if hi < 0 else 'neutral'
    res = dict(tag=tag, X=float(x.mean()), Y=float(y.mean()), n_X=nx, n_Y=ny, diff=float(d.mean()), ci=[float(lo), float(hi)],
               better=float((d > 0).mean()), worse=float((d < 0).mean()), graphs=len(d), verdict=verdict)
    if '--json' in sys.argv:
        print(json.dumps(res))
    else:
        print(f"{tag}: X {res['X']:.4f} ({nx} evals)  Y {res['Y']:.4f} ({ny} evals)  Y-X {res['diff']:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  Y better/worse {res['better']:.3f}/{res['worse']:.3f}  {verdict}")


if __name__ == '__main__':
    main()
