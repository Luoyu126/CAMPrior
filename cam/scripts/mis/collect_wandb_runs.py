"""Collect the configs and results of local wandb runs (read-only) into one CSV.

Used to import the DiffUCO runs behind the CAM paper (Shengyu's RiemannCO checkout,
/home/shengyuf/CAM4CO/wandb). For each run directory it reads
  files/wandb-metadata.json  start time, git remote/commit, command-line args (ground truth)
  files/config.yaml          full argparse config, when present (args override it)
  files/output.log           per-epoch validation lines 'Epoch: E eval/object: -S' (training runs)
                             and the final 'Avg. obj: -S' / 'Avg. time: T' (test runs)
Objectives are written as positive MIS sizes. The log file's modification time is kept as the
run's last activity, which helps to tell which training run last wrote a shared save_dir.

    python cam/scripts/mis/collect_wandb_runs.py --wandb DIR --method DiffUCO --out runs.csv
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re

FIELDS = ['run', 'started', 'log_end', 'git_commit', 'method', 'mode', 'train_set', 'test_set',
          'noise_energy', 'lr', 'tau0', 'num_tp', 'beta', 'N_anneal', 'optimizer', 'lr_scheduler',
          'samples_per_epoch', 'batch_size', 'mini_batch_size', 'num_kp', 'epochs', 'patience',
          'num_h', 'num_l', 'seed', 'num_t', 'num_k', 'inference_schedule', 'save_dir',
          'n_eval', 'last_epoch', 'last_val', 'best_val', 'best_epoch', 'test_obj', 'test_time']


def read_config(path):
    """Flat {key: value} from a wandb config.yaml (only the 'value:' lines, no yaml dependency)."""
    cfg, key = {}, None
    for line in open(path):
        m = re.match(r'^([A-Za-z_0-9]+):\s*$', line)
        if m:
            key = m.group(1)
            continue
        m = re.match(r'^\s+value:\s*(.*)$', line)
        if m and key and key not in cfg:
            cfg[key] = m.group(1).strip()
    return cfg


def parse_args(args):
    out, i = {}, 0
    while i < len(args):
        a = args[i]
        if a.startswith('--'):
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                out[a[2:]] = args[i + 1]
                i += 2
                continue
            out[a[2:]] = 'true'
        i += 1
    return out


def collect(run_dir):
    meta_path = os.path.join(run_dir, 'files', 'wandb-metadata.json')
    if not os.path.exists(meta_path):
        return None
    meta = json.load(open(meta_path))
    cfg_path = os.path.join(run_dir, 'files', 'config.yaml')
    cfg = read_config(cfg_path) if os.path.exists(cfg_path) else {}
    cfg.update(parse_args(meta.get('args', [])))
    if 'valid_sample' in cfg:
        cfg.setdefault('valid_samples', cfg['valid_sample'])
    row = {k: cfg.get(k, '') for k in FIELDS}
    row['run'] = os.path.basename(run_dir.rstrip('/'))
    row['started'] = meta.get('startedAt', '')[:19]
    row['git_commit'] = meta.get('git', {}).get('commit', '')[:7]
    row['mode'] = 'train' if cfg.get('do_train') in ('true', 'True') else 'test'
    row['train_set'] = os.path.basename(cfg.get('train_path', '').rstrip('/'))
    row['test_set'] = os.path.basename(cfg.get('test_path', '').rstrip('/'))
    log_path = os.path.join(run_dir, 'files', 'output.log')
    if os.path.exists(log_path):
        row['log_end'] = dt.datetime.utcfromtimestamp(os.path.getmtime(log_path)).isoformat()[:19]
        log = open(log_path, errors='ignore').read().replace('\r', '\n')
        ev = [(int(e), -float(s)) for e, s in re.findall(r'Epoch: (\d+) eval/object: (-?[\d.]+)', log)]
        if ev:
            best = max(ev, key=lambda x: x[1])
            row.update(n_eval=len(ev), last_epoch=ev[-1][0], last_val=f'{ev[-1][1]:.4f}',
                       best_val=f'{best[1]:.4f}', best_epoch=best[0])
        m = re.findall(r'Avg\. obj: (-?[\d.]+)', log)
        if m:
            row['test_obj'] = f'{-float(m[-1]):.4f}'
        m = re.findall(r'Avg\. time: ([\d.]+)', log)
        if m:
            row['test_time'] = f'{float(m[-1]):.4f}'
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wandb', required=True, help='wandb directory with run-* folders')
    ap.add_argument('--method', default=None, help='keep only runs of this --method')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(a.wandb, '*run-*'))):
        r = collect(d)
        if r and (a.method is None or r['method'] == a.method):
            rows.append(r)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} runs -> {a.out}')


if __name__ == '__main__':
    main()
