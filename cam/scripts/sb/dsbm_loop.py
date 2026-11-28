"""Route 1' iteration driver (DSBM, docs/20261128/plan_route1prime.md §2-§5). One run = one directory.

Modes
  dsbm    two networks, cross-fed every iteration (the 1D pipeline of experiments/20261125_sbdifusco_rbsmall_imf/
          run_imf.sh, extended): the generator G (SB-DIFUSCO, start z -> solution) trains on pairs (z from the noising
          network, y ~ pi_B); the noising network N (--sb_reverse, solution -> start) trains on pairs (z ~ uniform,
          y from G, decoded unless --raw_final). Iteration 0 trains on the initial pairs (--init imf: independent,
          ipf: the reference's own end pair).
  frozen  control with the same iteration and epoch schedule but no learned coupling: G alone, every batch drawing
          the closed-form coupling (omega > 0: Sinkhorn potentials --logg) or the independent one (omega = 0).

Iteration k: train (k = 0: --E0 epochs, G from --base, N from scratch; k > 0: --E epochs from iteration k-1's last
epoch, the cosine schedule restarted) -> trace the iteration-k networks on the training graphs (J chains per graph;
dsbm: all graphs, these traces are iteration k+1's caches; frozen: the first --metrics_graphs graphs) -> metrics
(scripts/sb/dsbm_metrics.py on the first --metrics_graphs graphs) -> anchor evaluation of G's last epoch in the
background (experiments/20261125_sbdifusco_rbsmall/evaluate.sh: renoise 50/10/5 and posterior 50 at seed 0, the pair
trace at seed 100 with scripts/sb/start_dependence.py) -> stopping rule.

Stopping rule (plan §3, statistic fixed in docs/20261128/status_route1prime.md before any result): at least
--min_iters iterations; then stop at the first iteration k at which, over the last 3 changes (k-3 -> k), the
validation MIS (mean of the last 10 epochs of G's training log, first 64 validation graphs) moved by at most 0.03
each, and (dsbm) both caches' mean Hamming costs moved by at most 1 % each; at most --max_iters iterations.
--target N instead runs exactly N iterations (to align a group of runs to the latest one). The anchor never enters
the rule.

Restartable: state.json records finished steps; an interrupted training is moved aside (<dir>.killed<time>) and
redone, an incomplete trace is redone, missing evaluations are rerun.

Usage: python scripts/sb/dsbm_loop.py --out DIR --mode dsbm|frozen --omega W --gpus G[,G2] [--init imf|ipf]
       [--label_set LS] [--temp T] [--logg FILE] [--base CKPT] [--seed 0] [--E0 100] [--E 50] [--J 16]
       [--min_iters 10] [--max_iters 30] [--target N] [--raw_final] [--dry]
"""
import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

CAM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
REPO = os.path.dirname(CAM)
D = '/usr1/shengyuf/MIS'
EVAL = os.path.join(REPO, 'experiments/20261125_sbdifusco_rbsmall/evaluate.sh')
EVAL_SPECS = [('renoise', 50, 0), ('renoise', 10, 0), ('renoise', 5, 0), ('posterior', 50, 0)]
PAIR_SPEC = ('renoise', 50, 100)
LS_DEFAULT = '/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/matched_kamis_loaderorder.pt'
BASE_DEFAULT = os.path.join(REPO, 'runs/experiments/20261119_difusco_rbsmall_reach/'
                            '20261119_difusco_rbsmall_reach_q5_mkamisfix_long_soft_s0/best.pt')


def stop_rule(v, c_f, c_b, k, min_iters, max_iters, target=None):
    """Whether to stop after iteration k (0-based). v: validation statistic per iteration; c_f, c_b: cache costs per
    iteration (None entries for the frozen mode). Returns (stop, reason)."""
    if target is not None:
        return (k + 1 >= target), f'target {target}'
    if k + 1 >= max_iters:
        return True, f'max_iters {max_iters}'
    if k + 1 < min_iters or k < 3:
        return False, 'min_iters'
    ok_v = all(abs(v[j] - v[j - 1]) <= 0.03 for j in range(k - 2, k + 1))
    ok_c = True
    for c in (c_f, c_b):
        if c[k] is None:
            continue
        ok_c &= all(abs(c[j] - c[j - 1]) <= 0.01 * abs(c[j - 1]) for j in range(k - 2, k + 1))
    return (ok_v and ok_c), f'converged v {ok_v} cost {ok_c}'


def val_stat(train_log, last=10):
    vals = []
    for line in open(train_log, errors='ignore'):
        if line.startswith('Epoch:') and 'eval/object:' in line:
            vals.append(-float(line.split('eval/object:')[1].split()[0]))
    assert vals, f'{train_log}: no validation lines'
    return float(np.mean(vals[-last:])), len(vals)        # fewer than `last` epochs only in smoke tests


def cache_cost(path):
    d = torch.load(path, weights_only=False)
    return float(np.mean([float((s != l).sum(1).float().mean()) for s, l in zip(d['starts'], d['labels'])]))


class Loop:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(a.out)
        os.makedirs(os.path.join(self.out, 'caches'), exist_ok=True)
        self.gpus = a.gpus.split(',')
        assert len(self.gpus) >= (2 if a.mode == 'dsbm' else 1)
        self.state_file = os.path.join(self.out, 'state.json')
        self.state = json.load(open(self.state_file)) if os.path.exists(self.state_file) else {'iters': {}}
        self.state['args'] = vars(a)
        self.name = os.path.basename(self.out)
        self.evals = []
        self.env = dict(os.environ, CAM_GRAPH_CACHE='/usr0/home/shengyuf/yunyi/.cache/cam/graphs')
        self.py = os.environ['CAM_PYTHON']
        self.common = ['--train_path', f'{D}/RB_small_train', '--valid_path', f'{D}/RB_small_valid', '--problem', 'mis',
                       '--seed', str(a.seed), '--method', 'DIFUSCO', '--model', 'EncodeProcessDecode', '--num_h', '64',
                       '--num_l', '8', '--num_tp', '50', '--diffusion_schedule', 'cosine', '--num_t', '50',
                       '--inference_schedule', 'linear', '--heat_lambda', '0', '--heat_until', '0.5',
                       '--difusco_sampler', 'renoise', '--num_workers', '4', '--sb_coupling', '--sb_omega', str(a.omega),
                       '--label_choice', 'soft', '--wandb_name', 'SLVUL']

    # bookkeeping
    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(os.path.join(self.out, 'loop.log'), 'a') as f:
            f.write(line + '\n')

    def save(self):
        tmp = self.state_file + '.tmp'
        json.dump(self.state, open(tmp, 'w'), indent=1)
        os.replace(tmp, self.state_file)

    def it(self, k):
        return self.state['iters'].setdefault(str(k), {})

    def net_dir(self, net, k):
        return os.path.join(self.out, f'{net}_it{k}')

    def epochs(self, k):
        return self.a.E0 if k == 0 else self.a.E

    def last_ckpt(self, net, k):
        return os.path.join(self.net_dir(net, k), f'epoch_{self.epochs(k) - 1:04d}.pt')

    def run_parallel(self, jobs):
        """jobs: list of (argv, env, logfile, cwd). Waits for all; raises if any failed."""
        if self.a.dry:
            for j in jobs:
                self.log('DRY ' + ' '.join(j[0]))
            return
        procs = []
        for argv, env, logf, cwd in jobs:
            procs.append((subprocess.Popen(argv, env=env, cwd=cwd, stdout=open(logf, 'a'), stderr=subprocess.STDOUT), logf))
        bad = [lf for p, lf in procs if p.wait() != 0]
        if bad:
            raise RuntimeError(f'failed: {bad}')

    # steps
    def initial_caches(self):
        f0, b0 = os.path.join(self.out, 'caches/f0.pt'), os.path.join(self.out, 'caches/b0.pt')
        if os.path.exists(f0) and os.path.exists(b0):
            return
        mode = 'ipf' if self.a.init == 'ipf' else 'independent'
        argv = [self.py, 'scripts/sb/imf_cache.py', mode, '--label_set', self.a.label_set, '--J', str(self.a.J),
                '--seed', str(self.a.seed), '--out_fwd', f0, '--out_bwd', b0]
        if self.a.temp is not None:
            argv += ['--temp', str(self.a.temp)]
        if mode == 'ipf':
            argv += ['--omega', str(self.a.omega)]
        self.run_parallel([(argv, self.env, os.path.join(self.out, 'caches/caches.log'), CAM)])

    def train_job(self, net, k, gpu):
        d = self.net_dir(net, k)
        if os.path.exists(d) and not os.path.exists(self.last_ckpt(net, k)):
            moved = f'{d}.killed{int(time.time())}'
            shutil.move(d, moved)
            self.log(f'iteration {k}: incomplete {net} training moved to {os.path.basename(moved)}')
        os.makedirs(d, exist_ok=True)
        argv = [self.py, 'main.py'] + self.common + ['--test_path', f'{D}/RB_small_test']
        argv += ['--sb_reverse', '--valid_samples', '2'] if net == 'bwd' else ['--valid_samples', '64']
        if self.a.train_samples:
            argv += ['--train_samples', str(self.a.train_samples)]
        if self.a.mode == 'dsbm':
            argv += ['--sb_pair_cache', os.path.join(self.out, f'caches/{"f" if net == "fwd" else "b"}{k}.pt')]
        else:
            argv += ['--label_set', self.a.label_set]
            if self.a.logg:
                argv += ['--sb_logg', self.a.logg]
        ck = (self.a.base if net == 'fwd' else None) if k == 0 else self.last_ckpt(net, k - 1)
        if ck:
            argv += ['--checkpoint', ck]
        argv += ['--batch_size', '40', '--optimizer', 'AdamW', '--lr', '0.0002', '--weight_decay', '0.0001',
                 '--lr_scheduler', 'cosine-decay', '--epochs', str(self.epochs(k)), '--patience', '1000',
                 '--save_every_epoch', '--num_k', '20', '--do_train', '--save_dir', d,
                 '--wandb_run_name', f'{self.name}_{net}_it{k}']
        env = dict(self.env, CUDA_VISIBLE_DEVICES=gpu)
        return argv, env, os.path.join(d, 'train.log'), CAM

    def trace_path(self, net, k):
        return os.path.join(self.out, f'caches/trace_{net}_it{k}.jsonl')

    def trace_job(self, net, k, gpu, graphs):
        t = self.trace_path(net, k)
        if os.path.exists(t):
            os.replace(t, t + '.old')
        argv = [self.py, 'main.py'] + self.common + ['--test_path', f'{D}/RB_small_train', '--batch_size', '1',
                                                     '--num_k', str(self.a.J), '--checkpoint', self.last_ckpt(net, k),
                                                     '--save_dir', self.net_dir(net, k),
                                                     '--log_file', t.replace('.jsonl', '_result.txt'),
                                                     '--wandb_run_name', f'{self.name}_{net}_trace{k}']
        argv += ['--test_samples', str(graphs)]
        env = dict(self.env, CUDA_VISIBLE_DEVICES=gpu, CAM_SB_PAIR_TRACE=t, CAM_SB_TRACE_NOSHARE='1', CAM_WANDB='0')
        if net == 'bwd':
            argv += ['--sb_reverse']
            env['CAM_SB_STARTS'] = self.a.label_set
            env['CAM_SB_STARTS_TEMP'] = 'none' if self.a.temp is None else str(self.a.temp)
        elif self.a.raw_final:
            env['CAM_SB_TRACE_RAW'] = '1'
        return argv, env, t.replace('.jsonl', '.log'), CAM

    def trace_complete(self, net, k, graphs):
        t = self.trace_path(net, k)
        if not os.path.exists(t):
            return False
        n = sum(1 for _ in open(t))
        return n == graphs

    def evaluate_bg(self, k):
        """Anchor evaluation of G's last epoch of iteration k, in the background on the first GPU."""
        d = self.net_dir('fwd', k)
        done = os.path.join(d, 'eval_done.json')
        if os.path.exists(done) or self.a.dry:
            return
        ck = self.last_ckpt('fwd', k)
        q = shlex.quote
        lines = [f'cd {q(REPO)}', 'set -u']
        flags = f'--sb_coupling --sb_omega {self.a.omega} --checkpoint {q(ck)}'
        for s, t, seed in EVAL_SPECS:
            tag = f'anchor_{s}_t{t}_seed{seed}'
            lines.append(f'[ -f {q(os.path.join(d, tag + "_result.txt"))} ] || bash {q(EVAL)} {self.gpus[0]} {q(d)} anchor '
                         f'{s} {t} {seed} {flags}')
        s, t, seed = PAIR_SPEC
        tag = f'anchor_{s}_t{t}_seed{seed}'
        lines.append(f'[ -f {q(os.path.join(d, tag + "_result.txt"))} ] || PAIR_TRACE=1 bash {q(EVAL)} {self.gpus[0]} {q(d)} '
                     f'anchor {s} {t} {seed} {flags}')
        lines.append(f'{q(self.py)} {q(CAM + "/scripts/sb/start_dependence.py")} {q(os.path.join(d, tag + "_pairs.jsonl"))} '
                     f'> {q(os.path.join(d, "start_dependence.txt"))}')
        lines.append(f'{q(self.py)} {q(CAM + "/scripts/sb/dsbm_loop.py")} --collect_eval {q(d)} > {q(done + ".tmp")} && '
                     f'mv {q(done + ".tmp")} {q(done)}')
        sh = os.path.join(d, 'eval.sh')
        open(sh, 'w').write('\n'.join(lines) + '\n')
        p = subprocess.Popen(['bash', sh], env=self.env, stdout=open(os.path.join(d, 'eval.log'), 'a'),
                             stderr=subprocess.STDOUT)
        self.evals.append(p)

    def iteration(self, k):
        a, st = self.a, self.it(k)
        n_train = a.train_samples or len(glob.glob(f'{D}/RB_small_train/*gpickle'))
        if not st.get('trained'):
            self.log(f'iteration {k}: training ({self.epochs(k)} epochs)')
            jobs = [self.train_job('fwd', k, self.gpus[0])]
            if a.mode == 'dsbm':
                jobs.append(self.train_job('bwd', k, self.gpus[1]))
            self.run_parallel(jobs)
            st['trained'] = True; self.save()
        self.evaluate_bg(k)
        graphs_trace = n_train if a.mode == 'dsbm' else a.metrics_graphs
        if not st.get('traced'):
            self.log(f'iteration {k}: tracing')
            jobs = []
            if not self.trace_complete('fwd', k, graphs_trace):
                jobs.append(self.trace_job('fwd', k, self.gpus[0], graphs_trace))
            if a.mode == 'dsbm' and not self.trace_complete('bwd', k, graphs_trace):
                jobs.append(self.trace_job('bwd', k, self.gpus[1], graphs_trace))
            self.run_parallel(jobs)
            st['traced'] = True; self.save()
        if a.mode == 'dsbm' and not st.get('cached'):
            nxt = k + 1
            for net, letter, direction in (('fwd', 'b', 'fwd'), ('bwd', 'f', 'bwd')):
                self.run_parallel([([self.py, 'scripts/sb/imf_cache.py', 'from_trace', '--trace', self.trace_path(net, k),
                                     '--direction', direction, '--out', os.path.join(self.out, f'caches/{letter}{nxt}.pt')],
                                    self.env, os.path.join(self.out, 'caches/caches.log'), CAM)])
            st['cached'] = True; self.save()
        if not st.get('metrics') and not a.dry:
            argv = [self.py, 'scripts/sb/dsbm_metrics.py', '--fwd', self.trace_path('fwd', k), '--label_set', a.label_set,
                    '--graphs', str(a.metrics_graphs)]
            if a.mode == 'dsbm':
                argv += ['--bwd', self.trace_path('bwd', k)]
            if a.temp is not None:
                argv += ['--temp', str(a.temp)]
            out = subprocess.run(argv, env=self.env, cwd=CAM, capture_output=True, text=True, check=True).stdout
            m = json.loads(out.strip().splitlines()[-1])
            m['val'], m['val_epochs'] = val_stat(os.path.join(self.net_dir('fwd', k), 'train.log'))
            if a.mode == 'dsbm':
                m['cost_fcache'] = cache_cost(os.path.join(self.out, f'caches/f{k + 1}.pt'))   # pairs (z from N, y)
                m['cost_bcache'] = cache_cost(os.path.join(self.out, f'caches/b{k + 1}.pt'))   # pairs (y from G, z)
            st['metrics'] = m; self.save()
            self.log(f'iteration {k}: metrics {json.dumps(m)}')

    def run(self):
        a = self.a
        self.log(f'start {self.name}: {json.dumps(vars(a))}')
        if a.mode == 'dsbm':
            self.initial_caches()
        k = 0
        while True:
            self.iteration(k)
            if a.dry:
                stop, why = k + 1 >= (a.target or 2), 'dry'
            else:
                its = [self.it(j)['metrics'] for j in range(k + 1)]
                v = [m['val'] for m in its]
                cf = [m.get('cost_fcache') for m in its]
                cb = [m.get('cost_bcache') for m in its]
                stop, why = stop_rule(v, cf, cb, k, a.min_iters, a.max_iters, a.target)
            self.log(f'iteration {k}: stop={stop} ({why})')
            if stop:
                self.state['stopped_at'] = k + 1; self.state['stop_reason'] = why; self.save()
                break
            k += 1
        for p in self.evals:
            p.wait()
        self.log('done')


def collect_eval(d):
    """Gather the anchor results of one iteration directory into one JSON object (stdout)."""
    res = {}
    for f in glob.glob(os.path.join(d, 'anchor_*_result.txt')):
        tag = os.path.basename(f)[:-len('_result.txt')]
        for line in open(f):
            if 'Avg. size' in line:
                res[tag] = float(line.split()[2])
    sd = os.path.join(d, 'start_dependence.txt')
    if os.path.exists(sd):
        res['start_dependence'] = open(sd).read().strip().splitlines()[-1:]
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect_eval', default=None)
    ap.add_argument('--out')
    ap.add_argument('--mode', choices=['dsbm', 'frozen'])
    ap.add_argument('--omega', type=float)
    ap.add_argument('--gpus')
    ap.add_argument('--init', choices=['imf', 'ipf'], default='imf')
    ap.add_argument('--label_set', default=LS_DEFAULT)
    ap.add_argument('--temp', type=float, default=None)
    ap.add_argument('--logg', default=None, help='frozen mode: Sinkhorn potentials (omega > 0, or weighted pi1)')
    ap.add_argument('--base', default=BASE_DEFAULT)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--E0', type=int, default=100)
    ap.add_argument('--E', type=int, default=50)
    ap.add_argument('--J', type=int, default=16)
    ap.add_argument('--metrics_graphs', type=int, default=256)
    ap.add_argument('--min_iters', type=int, default=10)
    ap.add_argument('--max_iters', type=int, default=30)
    ap.add_argument('--target', type=int, default=None)
    ap.add_argument('--raw_final', action='store_true', help="dsbm: give N the generator's raw samples, not decoded")
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--train_samples', type=int, default=None, help='smoke tests only: first N training graphs')
    a = ap.parse_args()
    if a.collect_eval:
        return collect_eval(a.collect_eval)
    if a.mode == 'frozen' and (a.omega > 0 or a.temp is not None):
        assert a.logg, 'frozen mode with omega > 0 or a weighted pi1 needs --logg'
    Loop(a).run()


if __name__ == '__main__':
    main()
