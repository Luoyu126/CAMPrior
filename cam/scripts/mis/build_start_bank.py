"""Bank of MCMC start states for flow / bridge matching from an energy-based start (local addition,
see UPSTREAM.md item 43).

For every graph of a directory, K independent RLSA chains (solvers/rlsa.py: random start, num_steps
annealed steps, tau0 -> 0 linear, top-d flip rule) are run and the FINAL state of each chain is kept
(not the best state along the chain), as a sample of the start distribution. Graphs are read with
datasets.mis.Dataset, so node order and file order are the ones the solvers see (CAM_GRAPH_CACHE is
honoured). Shards (--shard i --num_shards n) let several GPUs fill one bank; merge with --merge.

Output (torch.save): {'files': [basename, ...] in the dataset's sorted order,
                      'x': [uint8 tensor (num_nodes, K), ...],
                      'size_best': [greedy-decoded best-of-K set size per graph],
                      'size_mean': [mean over the K chains], 'meta': {...}}
    python scripts/mis/build_start_bank.py --graphs DIR --out FILE --k 20 --steps 1000 --tau0 0.01 --d 5
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import datasets.mis as M  # noqa: E402
from envs.mis import Env  # noqa: E402
from utils.temp_schedulers import TempScheduler  # noqa: E402


@torch.no_grad()
def rlsa_final(env, graph, k, steps, tau0, d, gen):
    n = graph.num_nodes
    graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, (n, n)).to_sparse_csr()
    sched = TempScheduler('linear', tau0, 0.0, 0, steps, 0)
    x = torch.randint(0, 2, (n, k), generator=gen, device=graph.x.device).float()
    _, grad = env.energy_func(graph, x, True)
    for s in range(steps):
        tau = sched(s)
        delta = grad * (2 * x - 1) / 2
        term2 = -torch.kthvalue(-delta, d, dim=0, keepdim=True).values
        p = torch.sigmoid((delta - term2) / tau)
        x = torch.where(torch.rand(x.shape, generator=gen, device=x.device) < p, 1 - x, x)
        _, grad = env.energy_func(graph, x, True)
    sizes, _ = env.evaluate(graph, x)          # greedy decode of each chain's final state
    return x, -sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--tau0', type=float, default=0.01)
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--merge', nargs='*', help='shard files to merge into --out (no sampling)')
    a = ap.parse_args()

    if a.merge:
        parts = [torch.load(f, weights_only=False) for f in a.merge]
        rows = [r for p in parts for r in zip(p['index'], p['files'], p['x'], p['size_best'], p['size_mean'])]
        rows.sort(key=lambda r: r[0])
        assert [r[0] for r in rows] == list(range(len(rows))), 'shards do not cover the dataset'
        out = {'files': [r[1] for r in rows], 'x': [r[2] for r in rows], 'size_best': [r[3] for r in rows],
               'size_mean': [r[4] for r in rows], 'meta': parts[0]['meta']}
        torch.save(out, a.out)
        sb = torch.tensor(out['size_best']).float()
        print(f'merged {len(rows)} graphs -> {a.out}; best-of-K size {sb.mean():.3f}', flush=True)
        return

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = M.Dataset(a.graphs, torch.float32, a.limit)
    env = Env(SimpleNamespace(beta=1.02, mis_decoding='continuous'))
    gen = torch.Generator(device=dev)
    idxs = list(range(a.shard, len(ds), a.num_shards))
    res = {'index': [], 'files': [], 'x': [], 'size_best': [], 'size_mean': [],
           'meta': dict(k=a.k, steps=a.steps, tau0=a.tau0, d=a.d, seed=a.seed, graphs=os.path.abspath(a.graphs))}
    t0 = time.time()
    for j, i in enumerate(idxs):
        gen.manual_seed(a.seed * 1_000_003 + i)
        g = ds[i].to(dev)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=dev)
        x, sizes = rlsa_final(env, g, a.k, a.steps, a.tau0, a.d, gen)
        res['index'].append(i)
        res['files'].append(os.path.basename(ds.files[i]))
        res['x'].append(x.to(torch.uint8).cpu())
        res['size_best'].append(sizes.max().item())
        res['size_mean'].append(sizes.float().mean().item())
        if j % 100 == 0:
            print(f'{j}/{len(idxs)} {time.time() - t0:.0f}s best {res["size_best"][-1]:.0f}', flush=True)
    torch.save(res, a.out)
    sb = torch.tensor(res['size_best']).float()
    print(f'done {len(idxs)} graphs in {time.time() - t0:.0f}s; best-of-K {sb.mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
