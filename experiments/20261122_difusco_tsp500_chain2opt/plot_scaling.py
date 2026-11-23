"""Scaling plot, gap to Concorde (final 2-opt included). Left: vs the per-step 2-opt cap d, one line per number of
sampling steps T. Right: vs T, one line per d. Only tags with all 128 instances. Writes scaling_<model>.png next to this script.
    python plot_scaling.py [model run]"""
import glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.join(REPO, 'cam'))
from utils.tour_sets import canonical, parse_points, tour_length  # noqa: E402
RUN = sys.argv[1] if len(sys.argv) > 1 else '20261121_difusco_tsp500_labels_lkh_soft_s0'
L = open('/usr0/home/shengyuf/yunyi/data/CAM_data/TSP/tsp500_test_concorde.txt').read().splitlines()[:128]
ref = np.array([tour_length(parse_points(l), canonical(np.array(l.split(' output ')[1].split(), int) - 1)) for l in L])
base = np.genfromtxt(f'{REPO}/runs/experiments/20261121_difusco_tsp500_labels/{RUN}/test_renoise_2opt1000_seed0_result.txt.per_graph.csv',
                     delimiter=',', names=True, dtype=None, encoding=None)
bgap = np.mean(100 * (base['obj'][np.argsort(base['idx'])] / ref - 1))
pts = {}
for f in glob.glob(f'{REPO}/runs/experiments/20261122_difusco_tsp500_chain2opt/{RUN}/d*_t*_seed0.jsonl'):
    J = [json.loads(l) for l in open(f)]
    if len(J) != 128: continue
    d, T = map(int, re.match(r'd(\d+)_t(\d+)', os.path.basename(f)).groups())
    i = np.array([j['idx'] for j in J]); fi = np.array([j['final'] for j in J])
    pts.setdefault(T, []).append((np.mean([j['chain_iters'] for j in J]), np.mean(100 * (fi / ref[i] - 1)), d))
RAMP = {10: '#86b6ef', 20: '#5598e7', 50: '#2a78d6', 100: '#1c5cab', 200: '#104281'}  # one-hue ramp: more steps = darker
DCOL = {5: '#eb6834', 20: '#2a78d6', 50: '#1baf7a'}                                           # categorical slots
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, sharey=True, gridspec_kw={'width_ratios': [3, 2]})
for ax in (a1, a2):
    for s_ in ['top', 'right']: ax.spines[s_].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors='#5c5c58', labelsize=8)
    ax.axhline(bgap, color='#8a8a85', lw=1.2, ls=(0, (4, 3)))
a1.text(0.5, bgap + 0.02, f'renoise + final 2-opt only ({bgap:.2f}%)', color='#5c5c58', fontsize=8, va='bottom')
for T in sorted(pts):
    p = sorted((d, g) for _, g, d in pts[T])
    if len(p) < 2: continue
    a1.plot([max(d, 0.5) for d, _ in p], [g for _, g in p], '-o', color=RAMP.get(T, '#2a78d6'), lw=2, ms=6, mec='white', mew=1.5, label=f'T = {T}')
a1.set_xscale('log'); a1.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100]); a1.set_xticklabels(['0', '1', '2', '5', '10', '20', '50', '100'])
a1.set_xlabel('2-opt moves per step, cap d (each step converges by itself at ~36)', fontsize=9, color='#3a3a37')
a1.set_ylabel('gap to Concorde, % (final 2-opt included)', fontsize=9, color='#3a3a37')
a1.legend(title='sampling steps', title_fontsize=8, frameon=False, fontsize=8, loc='lower left')
for d in sorted(DCOL):
    p = sorted((T, g) for T in pts for _, g, dd in pts[T] if dd == d)
    if len(p) < 2: continue
    a2.plot([T for T, _ in p], [g for _, g in p], '-o', color=DCOL[d], lw=2, ms=6, mec='white', mew=1.5, label=f'd = {d}')
a2.set_xscale('log'); a2.set_xticks([10, 20, 50, 100, 200]); a2.set_xticklabels(['10', '20', '50', '100', '200'])
a2.set_xlabel('sampling steps T', fontsize=9, color='#3a3a37')
a2.legend(title='2-opt per step', title_fontsize=8, frameon=False, fontsize=8, loc='lower left')
fig.suptitle(f'TSP-500 in-chain 2-opt ({RUN.split("labels_")[-1].rsplit("_s", 1)[0]} model, 128 test instances, 16 samples)', fontsize=10, x=0.01, ha='left')
fig.tight_layout(); out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'scaling_{RUN.split("labels_")[-1]}.png')
fig.savefig(out); print(out)
