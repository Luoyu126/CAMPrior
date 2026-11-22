"""Summarize the in-chain 2-opt sweep: per tag, mean length and gap to Concorde with and without the final 2-opt,
mean 2-opt moves in the chain / in the final 2-opt, and the paired gap difference to the baseline (plain renoise,
final 2-opt 1000, from 20261121_difusco_tsp500_labels) with a bootstrap 95% CI.
    python summarize.py [model run]"""
import csv, glob, json, os, re, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.join(REPO, 'cam'))
from utils.tour_sets import canonical, parse_points, tour_length  # noqa: E402
RUN = sys.argv[1] if len(sys.argv) > 1 else '20261121_difusco_tsp500_labels_lkh_soft_s0'
L = open('/usr0/home/shengyuf/yunyi/data/CAM_data/TSP/tsp500_test_concorde.txt').read().splitlines()[:128]
ref = np.array([tour_length(parse_points(l), canonical(np.array(l.split(' output ')[1].split(), int) - 1)) for l in L])
base = np.full(128, np.nan)
for r in csv.DictReader(open(f'{REPO}/runs/experiments/20261121_difusco_tsp500_labels/{RUN}/test_renoise_2opt1000_seed0_result.txt.per_graph.csv')):
    base[int(r['idx'])] = float(r['obj'])
rng = np.random.default_rng(0); B = rng.integers(0, 128, (10000, 128))
def gap(v): return 100 * (v / ref - 1)
def ci(d): b = d[B].mean(1); return f'{d.mean():+.3f} [{np.percentile(b, 2.5):+.3f},{np.percentile(b, 97.5):+.3f}]'
print(f'reference {ref.mean():.4f}; baseline renoise + final 2-opt: {base.mean():.4f} ({gap(base).mean():.3f}%)')
print(f'{"tag":<16}{"n":>4}{"no final":>17}{"final 2opt":>17}{"chain mv":>9}{"final mv":>9}  final - baseline (gap %, 95% CI)')
rows = []
for f in glob.glob(f'{REPO}/runs/experiments/20261122_difusco_tsp500_chain2opt/{RUN}/*.jsonl'):
    tag = os.path.basename(f)[:-6]; J = [json.loads(l) for l in open(f)]
    if not J: continue
    d = int(re.match(r'd(\d+)', tag).group(1)) if tag.startswith('d') else -1
    rows.append((d, tag, J))
for d, tag, J in sorted(rows):
    i = np.array([j['idx'] for j in J]); nf = np.array([j['nofinal'] for j in J]); fi = np.array([j['final'] for j in J])
    cm = np.mean([j['chain_iters'] for j in J]); fm = np.mean([j['final_iters'] for j in J])
    print(f'{tag:<16}{len(J):>4}{nf.mean():>9.4f} {gap(nf)[0] if False else (100*(nf/ref[i]-1)).mean():>6.3f}%'
          f'{fi.mean():>9.4f} {(100*(fi/ref[i]-1)).mean():>6.3f}%{cm:>9.0f}{fm:>9.1f}  '
          + (ci(100 * (fi - base[i]) / ref[i]) if len(J) == 128 else '(partial)'))
