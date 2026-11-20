"""Paired per-graph comparison of a DIFUSCO setting (several runs x test seeds) against DiffUCO (test seeds 0-3).
Per graph: mean size over our runs minus mean size over DiffUCO's test seeds; graph-level bootstrap CI of the mean
difference, and the same for each of our runs separately.
    python paired.py RUN_DIR [RUN_DIR ...]
"""
import sys, glob, csv, numpy as np
DU = '/home/shengyuf/yunyi/Schrödinger Bridge/CAMPrior/runs/scratch/dist_1119/du_t50_s%d.txt.per_graph.csv'
def load(f):
    r = sorted(csv.DictReader(open(f)), key=lambda x: x['file']); return -np.array([float(x['obj']) for x in r]), [x['file'] for x in r]
du = []; files = None
for s in range(4):
    v, f = load(DU % s); du.append(v); files = files or f
du = np.mean(du, 0)
ours = []
for r in sys.argv[1:]:
    runs = []
    for f in sorted(glob.glob(r + '/test_renoise_seed*_result.txt.per_graph.csv')):
        v, ff = load(f); assert ff == files; runs.append(v)
    ours.append(np.mean(runs, 0)); print(f'{r.split("reach_")[-1]}: {np.mean(runs, 1).round(3)} -> diff {ours[-1].mean() - du.mean():+.4f}')
o = np.mean(ours, 0); d = o - du
rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
print(f'ours {o.mean():.4f}  DiffUCO {du.mean():.4f}  diff {d.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  '
      f'graphs better/worse {(d > 0).mean():.3f}/{(d < 0).mean():.3f}')
