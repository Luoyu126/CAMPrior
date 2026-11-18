"""Sets of distinct maximum independent sets per graph, for training with many labels (local addition,
see UPSTREAM.md item 44).

An MIS instance usually has many optimal solutions (RB-[200-300]: 22-192 distinct proven-optimal ones
found by 200 RLSA chains on graphs whose optimum was certified), and the KaMIS label is one of them
picked by the solver's own randomness. For every graph, K long RLSA chains (scripts/mis/
build_start_bank.py: random start, tau0 -> 0 linear, top-d flips) are greedy-decoded; the distinct
decoded sets of the best size found are kept, together with the original label when the label is as
large. If RLSA finds a larger set than the label, only the larger ones are kept.

Graphs are read with datasets.mis.Dataset (node and file order as the solvers see them). Shards
(--shard i --num_shards n) let several GPUs fill one file; merge with --merge.

Output (torch.save): {'files': [basename, ...] in the dataset's sorted order,
                      'sets': [uint8 tensor (M_i, num_nodes) of distinct best sets, label first if kept],
                      'label_size': [...], 'best_size': [...], 'label_kept': [...], 'meta': {...}}
    python scripts/mis/build_solution_sets.py --graphs DIR --out FILE --k 200 --steps 3000 --tau0 0.01 --d 5
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets.mis as M  # noqa: E402
from envs.mis import Env  # noqa: E402
from build_start_bank import rlsa_final  # noqa: E402

KEYS = ['index', 'files', 'sets', 'label_size', 'best_size', 'label_kept']


def best_distinct(env, graph, x):
    """Distinct greedy-decoded sets of the largest size among the chains and the label."""
    _, sol = env.evaluate(graph, x)
    sol = sol.float()
    if sol.shape[0] == graph.num_nodes:   # [N, K] -> [K, N]
        sol = sol.T
    y = graph.x.view(-1).float()
    size = sol.sum(1)
    best = max(size.max().item(), y.sum().item())
    top = torch.unique(sol[size == best], dim=0)
    kept = y.sum().item() == best
    if kept:
        top = top[~(top == y).all(1)]
        top = torch.cat([y[None], top], 0)
    return top.to(torch.uint8).cpu(), y.sum().item(), best, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=200)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--tau0', type=float, default=0.01)
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--merge', nargs='*', help='shard files to merge into --out')
    a = ap.parse_args()

    if a.merge:
        parts = [torch.load(f, weights_only=False) for f in a.merge]
        rows = sorted((r for p in parts for r in zip(*[p[k] for k in KEYS])), key=lambda r: r[0])
        assert [r[0] for r in rows] == list(range(len(rows))), 'shards do not cover the dataset'
        out = {k: [r[j] for r in rows] for j, k in enumerate(KEYS) if k != 'index'}
        out['meta'] = parts[0]['meta']
        torch.save(out, a.out)
        m = torch.tensor([s.shape[0] for s in out['sets']]).float()
        print(f"merged {len(rows)} graphs -> {a.out}; sets per graph mean {m.mean():.1f} median "
              f"{m.median():.0f}; label kept {sum(out['label_kept'])}; "
              f"larger than label {sum(b > l for b, l in zip(out['best_size'], out['label_size']))}", flush=True)
        return

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = M.Dataset(a.graphs, torch.float32, a.limit)
    env = Env(SimpleNamespace(beta=1.02, mis_decoding='continuous'))
    gen = torch.Generator(device=dev)
    res = {k: [] for k in KEYS}
    res['meta'] = dict(k=a.k, steps=a.steps, tau0=a.tau0, d=a.d, seed=a.seed, graphs=os.path.abspath(a.graphs))
    t0 = time.time()
    idxs = list(range(a.shard, len(ds), a.num_shards))
    for j, i in enumerate(idxs):
        gen.manual_seed(a.seed * 1_000_003 + i)
        g = ds[i].to(dev)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=dev)
        x, _ = rlsa_final(env, g, a.k, a.steps, a.tau0, a.d, gen)
        sets, ls, bs, kept = best_distinct(env, g, x)
        for k, v in zip(KEYS, [i, os.path.basename(ds.files[i]), sets, ls, bs, kept]):
            res[k].append(v)
        if j % 100 == 0:
            print(f'{j}/{len(idxs)} {time.time() - t0:.0f}s sets {sets.shape[0]}', flush=True)
    torch.save(res, a.out)
    print(f'done {len(idxs)} graphs in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
