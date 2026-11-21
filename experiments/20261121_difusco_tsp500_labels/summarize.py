"""Table of the TSP-500 test results: per run and test setting, mean tour length and mean gap (%) to the
reference tours of tsp500_test_concorde.txt, plus paired per-instance differences (bootstrap 95% CI) between
the arms. Results are the per_graph.csv files of test.sh (idx = line of the test file).
    python experiments/20261121_difusco_tsp500_labels/summarize.py [extra_name=per_graph.csv ...]
"""
import csv, glob, os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'cam'))
from utils.tour_sets import canonical, parse_points, tour_length  # noqa: E402
lines = open('/usr0/home/shengyuf/yunyi/data/CAM_data/TSP/tsp500_test_concorde.txt').read().splitlines()
ref = np.array([tour_length(parse_points(l), canonical(np.array(l.split(' output ')[1].split(), int) - 1)) for l in lines])


def load(f):
    v = np.full(len(ref), np.nan)
    for r in csv.DictReader(open(f)):
        v[int(r['idx'])] = float(r['obj'])
    return v


res = {}
R = os.path.join(REPO, 'runs/experiments/20261121_difusco_tsp500_labels')
for f in sorted(glob.glob(R + '/*/test_*_result.txt.per_graph.csv')):
    run = os.path.basename(os.path.dirname(f)).replace('20261121_difusco_tsp500_labels_', '')
    res[(run, os.path.basename(f).split('_result')[0][5:])] = load(f)
for a in sys.argv[1:]:
    k, f = a.split('=', 1); res[(k, '')] = load(f)
print(f'reference length {ref.mean():.4f}')
for (run, tag), v in res.items():
    m = ~np.isnan(v)
    print(f'{run:18s} {tag:28s} n={m.sum():3d} length {v[m].mean():.4f} gap {100 * (v[m] / ref[m] - 1).mean():+.3f}%')
rng = np.random.default_rng(0)


def paired(a, b):
    m = ~np.isnan(a) & ~np.isnan(b); d = 100 * (b[m] - a[m]) / ref[m]
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    return f'{d.mean():+.3f}% [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] (n={m.sum()})'


tags = sorted({t for _, t in res if t})
for t in tags:
    g = {r: v for (r, tt), v in res.items() if tt == t}
    for a, b in [('lkh_random_s0', 'lkh_soft_s0'), ('cam_random_s0', 'cam_soft_s0'),
                 ('lkh_random_s0', 'cam_random_s0'), ('lkh_soft_s0', 'cam_soft_s0')]:
        if a in g and b in g:
            print(f'[{t}] {b} - {a}: gap diff {paired(g[a], g[b])}')
