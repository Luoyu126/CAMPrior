"""Paired per-graph comparison of two DIFUSCO settings X and Y (each: several runs x renoise test seeds).
Per graph: mean size over Y's runs minus mean over X's; graph-level bootstrap CI of the mean difference.
    python paired_arms.py 'X_RUN_GLOB' 'Y_RUN_GLOB'
"""
import sys, glob, csv, numpy as np
def arm(g):
    v, files = [], None
    for r in sorted(glob.glob(g)):
        for f in sorted(glob.glob(r + '/test_renoise_seed*_result.txt.per_graph.csv')):
            rows = sorted(csv.DictReader(open(f)), key=lambda x: x['file'])
            ff = [x['file'] for x in rows]; assert files is None or ff == files; files = ff
            v.append(-np.array([float(x['obj']) for x in rows]))
    return np.mean(v, 0), len(v)
x, nx = arm(sys.argv[1]); y, ny = arm(sys.argv[2]); d = y - x
rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
print(f'X {x.mean():.4f} ({nx} tests)  Y {y.mean():.4f} ({ny} tests)  Y-X {d.mean():+.4f}  95% CI '
      f'[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  graphs Y better/worse {(d > 0).mean():.3f}/{(d < 0).mean():.3f}')
