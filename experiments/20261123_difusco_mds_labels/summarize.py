"""Summarize the MDS SL-vs-UL tests: per run and sampler, mean dominating-set size on the 1000 test graphs, gap to the
integer-programming optimum (data/mds_labels/<ds>_small_test_ilp1.pt), share of graphs solved optimally, and paired
differences (bootstrap 95% CI over graphs) for the comparisons that test the two hypotheses:
  labels:   multi-optimum labels (ilp8 random / soft) vs the single-optimum baseline (ilp1), same sampler
  sampler:  renoise vs D3PM posterior, same model
DiffUCO test results (20261123_diffuco_mds_ul) are included when present.
    python experiments/20261123_difusco_mds_labels/summarize.py <ba|rb>"""
import csv, glob, os, sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DS = sys.argv[1]
il = torch.load(f'/usr0/home/shengyuf/yunyi/data/mds_labels/{DS}_small_test_ilp1.pt', weights_only=False)
opt = dict(zip(il['files'], il['k']))
files = il['files']; k = np.array([opt[f] for f in files], float)
B = np.random.default_rng(0).integers(0, len(files), (10000, len(files)))


def load(path):
    v = {r['file']: abs(float(r['obj'])) for r in csv.DictReader(open(path))}
    if len(v) != len(files):
        return None
    return np.array([v[f] for f in files])


res = {}
for exp in ['20261123_difusco_mds_labels', '20261123_diffuco_mds_ul']:
    for p in sorted(glob.glob(f'{REPO}/runs/experiments/{exp}/*_{DS}small_*/test_*_result.txt.per_graph.csv')):
        run = os.path.basename(os.path.dirname(p)).replace(exp + '_', '')
        tag = os.path.basename(p).replace('_result.txt.per_graph.csv', '').replace('test_', '')
        x = load(p)
        if x is not None:
            res[(run, tag)] = x


def ci(d):
    b = d[B].mean(1)
    return f'{d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]'


print(f'{DS}-small: {len(files)} test graphs, ILP optimum {k.mean():.3f} (proven on {np.mean(il["proven"]):.3f})')
print(f'{"run":<36}{"sampler":<18}{"size":>8}{"gap %":>8}{"optimal":>9}')
for (run, tag), x in res.items():
    print(f'{run:<36}{tag:<18}{x.mean():>8.3f}{100 * np.mean(x / k - 1):>8.3f}{np.mean(x <= k):>9.3f}')

print('\npaired size difference (negative = first is better), 95% bootstrap CI')
runs = sorted({r for r, _ in res})
for r in runs:                                   # sampler: renoise - posterior, same model
    a, b = res.get((r, 'renoise_seed0')), res.get((r, 'posterior_seed0'))
    if a is not None and b is not None:
        print(f'{r}: renoise - posterior  {ci(a - b)}')
base = [r for r in runs if r.endswith('ilp1_post_s0')]
for r in runs:                                   # labels: other label mode - ilp1 baseline (seed 0), same sampler
    if 'ilp1' in r or not base:
        continue
    for s in ['posterior_seed0', 'renoise_seed0']:
        a, b = res.get((r, s)), res.get((base[0], s))
        if a is not None and b is not None:
            print(f'{r} - {base[0]} [{s}]  {ci(a - b)}')
s0, s1 = [res.get((f'{DS}small_ilp1_post_s{i}', 'posterior_seed0')) for i in (0, 1)]
if s0 is not None and s1 is not None:
    print(f'seed noise: ilp1 s1 - s0 [posterior]  {ci(s1 - s0)}')
