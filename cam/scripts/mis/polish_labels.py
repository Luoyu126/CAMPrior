"""Bring sampled labels up to the reference size by local search from them (local addition, UPSTREAM.md item 46).

For every graph whose label in --labels (e.g. scripts/mis/ul_labels.py output) is smaller than the
reference label (graph.x, KaMIS), run --k RLSA chains started at that label (low temperature, the
update of solvers/rlsa.py) and take, among the greedy-decoded results of at least the reference size,
the one closest in Hamming distance to the starting label. If no chain gets there (or --k 0), take the
set of --pool (optimal sets, scripts/mis/build_solution_sets.py) closest to the starting label, else the
reference label. Writes a one-set-per-graph label file with the source of each label.
    python scripts/mis/polish_labels.py --labels IN --graphs DIR --out OUT
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import datasets.mis as M  # noqa: E402
from envs.mis import Env  # noqa: E402
from utils.label_sets import load_label_sets  # noqa: E402
from utils.temp_schedulers import TempScheduler  # noqa: E402


def rlsa_from(env, graph, x0, k, steps, tau0, d, gen):
    n = graph.num_nodes
    graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, (n, n)).to_sparse_csr()
    sched = TempScheduler('linear', tau0, 0.0, 0, steps, 0)
    x = x0.view(n, 1).float().repeat(1, k)
    _, grad = env.energy_func(graph, x, True)
    for s in range(steps):
        delta = grad * (2 * x - 1) / 2
        term2 = -torch.kthvalue(-delta, d, dim=0, keepdim=True).values
        p = torch.sigmoid((delta - term2) / sched(s))
        x = torch.where(torch.rand(x.shape, generator=gen, device=x.device) < p, 1 - x, x)
        _, grad = env.energy_func(graph, x, True)
    _, sol = env.evaluate(graph, x)
    sol = sol.float()
    return sol.T if sol.shape[0] == n else sol   # [k, n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--graphs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=200)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--tau0', type=float, default=0.01)
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pool', default=None, help='optimal sets to fall back on (nearest to the label)')
    a = ap.parse_args()
    dev = 'cuda'
    sets = load_label_sets(a.labels, a.graphs)
    src = torch.load(a.labels, weights_only=False)
    pool = load_label_sets(a.pool, a.graphs) if a.pool else None
    ds = M.Dataset(a.graphs, torch.float32, len(sets))
    env = Env(argparse.Namespace(beta=1.02, mis_decoding='continuous'))
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    out, source = [], []
    for i in range(len(sets)):
        x0 = sets[i][0].float().to(dev)
        g = ds[i].to(dev)
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=dev)
        ref = g.x.view(-1).float()
        if x0.sum() >= ref.sum():
            out.append(sets[i]); source.append('sample'); continue
        sol = rlsa_from(env, g, x0, a.k, a.steps, a.tau0, a.d, gen) if a.k > 0 else x0[None][:0]
        ok = sol[sol.sum(1) >= ref.sum()]
        cand = pool[i].float().to(dev) if pool is not None else x0[None][:0]
        cand = cand[cand.sum(1) >= ref.sum()]
        if len(ok):
            j = (ok - x0).abs().sum(1).argmin()
            out.append(ok[j][None].to(torch.uint8).cpu()); source.append('polished')
        elif len(cand):
            j = (cand - x0).abs().sum(1).argmin()
            out.append(cand[j][None].to(torch.uint8).cpu()); source.append('pool')
        else:
            out.append(ref[None].to(torch.uint8).cpu()); source.append('reference')
    torch.save({'files': src['files'], 'sets': out, 'source': source,
                'meta': {'from': os.path.abspath(a.labels), 'args': vars(a)}}, a.out)
    sizes = torch.tensor([float(s.sum()) for s in out])
    print(f"{len(out)} graphs: mean size {sizes.mean():.3f}; sample {source.count('sample')}, "
          f"polished {source.count('polished')}, pool {source.count('pool')}, reference {source.count('reference')}")


if __name__ == '__main__':
    main()
