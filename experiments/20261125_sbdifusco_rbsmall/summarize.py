"""Collect the evaluations of experiments/20261125_sbdifusco_rbsmall into tables (markdown, printed).

For every run directory under runs/experiments/20261125_sbdifusco_rbsmall/ (baseline_* are symlinked checkpoints of
the 11-19 baselines): best-of-20 mean MIS size for every <set>_<sampler>_t<steps>_seed<N>_result.txt, and, for the
pair-traced evaluations (<tag>_pairs.jsonl, seed 100), the single-sample mean size and the start-dependence metrics of
cam/scripts/sb/start_dependence.py (val64 traces use the KaMIS multi-solutions of the first 64 validation graphs).

    python experiments/20261125_sbdifusco_rbsmall/summarize.py [--set anchor] [--runs OTHER_RUN_ROOT]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(REPO, 'runs', 'experiments', os.path.basename(HERE))
LABELS = '/usr0/home/shengyuf/yunyi/data/rb_small_sb/val64_kamis_k128.pt'
PAT = re.compile(r'^(anchor|test|val64)_(\w+?)_t(\d+)_seed(\d+)_result\.txt$')


def short(run):
    return run.replace('20261125_sbdifusco_rbsmall_labelvar_', 'lv_').replace('20261125_sbdifusco_rbsmall_', '')


def sizes(set_name):
    rows = {}
    for d in sorted(glob.glob(os.path.join(RUNS, '*'))):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = PAT.match(f)
            if not m or m.group(1) != set_name:
                continue
            txt = open(os.path.join(d, f)).read()
            v = re.findall(r'Avg\. size: ([0-9.]+)', txt)
            if v:
                rows.setdefault(short(os.path.basename(d)), {})[(m.group(2), int(m.group(3)), int(m.group(4)))] = float(v[-1])
    return rows


def pairs(python):
    out = {}
    for f in sorted(glob.glob(os.path.join(RUNS, '*', '*_pairs.jsonl'))):
        run = short(os.path.basename(os.path.dirname(f)))
        tag = os.path.basename(f)[:-len('_pairs.jsonl')]
        cmd = [python, os.path.join(REPO, 'cam', 'scripts', 'sb', 'start_dependence.py'), f]
        if tag.startswith('val64'):
            cmd += ['--labels', LABELS]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            out[(run, tag)] = json.loads(r.stdout.strip().splitlines()[-1])
        else:
            out[(run, tag)] = {'error': r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no output'}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='anchor')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--runs', default=None, help='another run root, e.g. runs/experiments/20261125_sbdifusco_rbsmall_labelvar')
    args = ap.parse_args()
    global RUNS
    if args.runs:
        RUNS = os.path.abspath(args.runs)
    rows = sizes(args.set)
    cols = sorted({k for r in rows.values() for k in r if k[2] < 100}, key=lambda k: (k[0] != 'renoise', -k[1], k[2]))
    print(f'## best-of-20 mean MIS size, {args.set}\n')
    print('| run | ' + ' | '.join(f'{s} {t} steps (seed {sd})' for s, t, sd in cols) + ' |')
    print('|---|' + '---|' * len(cols))
    for run, r in sorted(rows.items()):
        print(f'| {run} | ' + ' | '.join(f'{r[c]:.3f}' if c in r else '' for c in cols) + ' |')
    P = pairs(args.python)
    if P:
        print('\n## pair-traced evaluations (chains 2j, 2j+1 share the start)\n')
        keys = ['size', 'd_same', 'd_diff', 'ratio', 'hit', 'hit_shuf', 'd_near', 'd_rand']
        print('| run | evaluation | ' + ' | '.join(keys) + ' |')
        print('|---|---|' + '---|' * len(keys))
        for (run, tag), m in sorted(P.items()):
            print(f'| {run} | {tag} | ' + ' | '.join(f'{m[k]:.4f}' if isinstance(m.get(k), float) else str(m.get(k, ''))
                                                     for k in keys) + ' |')


if __name__ == '__main__':
    main()
