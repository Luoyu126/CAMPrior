"""Per-iteration table of route-1' runs (docs/20261128/plan_route1prime.md §4): the stopping statistic (validation
MIS, mean of the last 10 epochs, first 64 validation graphs), the anchor evaluation of G's last epoch (renoise 50 /
10 / 5, posterior 50, evaluation seed 0; start-dependence ratio at seed 100), and the training dynamics.

Usage: python experiments/20261128_dsbm_rbsmall/summary.py <run name prefix, e.g. d1_> [--markdown]
"""
import glob
import json
import os
import sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runs', 'experiments', '20261128_dsbm_rbsmall')


def rows(run_dir):
    st = json.load(open(os.path.join(run_dir, 'state.json')))
    out = []
    for k in sorted(st['iters'], key=int):
        m = st['iters'][k].get('metrics')
        if not m:
            continue
        ev = {}
        f = os.path.join(run_dir, f'fwd_it{k}', 'eval_done.json')
        if os.path.exists(f):
            ev = json.load(open(f))
        sd = None
        if ev.get('start_dependence'):
            sd = json.loads(ev['start_dependence'][0]).get('ratio')
        fw, bw = m.get('fwd', {}), m.get('bwd', {})
        out.append({
            'it': int(k), 'val': m['val'],
            'r50': ev.get('anchor_renoise_t50_seed0'), 'r10': ev.get('anchor_renoise_t10_seed0'),
            'r5': ev.get('anchor_renoise_t5_seed0'), 'p50': ev.get('anchor_posterior_t50_seed0'), 'sdep': sd,
            'g_size': fw.get('size'), 'g_dnear': fw.get('d_near'), 'g_tvx': (fw.get('tv_near', 0) - fw.get('tv_floor', 0)),
            'g_cost': fw.get('cost'), 'n_dens': bw.get('density'), 'n_conf': bw.get('conflict'),
            'n_agree': bw.get('agree'), 'cf': m.get('cost_fcache'), 'cb': m.get('cost_bcache')})
    return out, st


def fmt(v, nd=3):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ''
    cols = ['it', 'val', 'r50', 'r10', 'r5', 'p50', 'sdep', 'g_size', 'g_dnear', 'g_tvx', 'g_cost', 'n_dens', 'n_conf',
            'n_agree', 'cf', 'cb']
    for d in sorted(glob.glob(os.path.join(R, f'20261128_dsbm_rbsmall_{prefix}*'))):
        if not os.path.exists(os.path.join(d, 'state.json')):
            continue
        rs, st = rows(d)
        print(f"\n### {os.path.basename(d).replace('20261128_dsbm_rbsmall_', '')}  (stopped at {st.get('stopped_at', '—')}, "
              f"{st.get('stop_reason', 'running')})\n")
        print('| ' + ' | '.join(cols) + ' |')
        print('|' + '---|' * len(cols))
        for r in rs:
            print('| ' + ' | '.join(fmt(r[c], 1 if c in ('cf', 'cb', 'g_cost') else 3) for c in cols) + ' |')


if __name__ == '__main__':
    main()
