"""Diagnostic for the energy-heated DIFUSCO forward (UPSTREAM.md item 42).

For a few training graphs, starting from the label x*, prints at several t:
  - the fraction of nodes that differ from x*,
  - the fraction of nodes set to 1,
  - the number of conflicting edges per graph,
for the uniform forward (lambda = 0) and the heated forward at each given lambda.

  python heated_forward_curve.py --data /usr1/shengyuf/MIS/RB_small_train --graphs 40 --lambdas 0 1 3
"""
import argparse
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from datasets.mis import Dataset  # noqa: E402
from utils.diffusion_schedulers import CategoricalDiffusion  # noqa: E402
from utils.heated_forward import HeatedForward  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--graphs', type=int, default=40)
    p.add_argument('--chains', type=int, default=20)
    p.add_argument('--lambdas', type=float, nargs='+', default=[0, 1, 3])
    p.add_argument('--heat_until', type=float, default=0.5)
    p.add_argument('--beta', type=float, default=1.02)
    p.add_argument('--T', type=int, default=1000)
    p.add_argument('--ts', type=int, nargs='+', default=[1, 10, 30, 100, 200, 300, 400, 500, 600, 800, 1000])
    a = p.parse_args()
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = Dataset(a.data, num_samples=a.graphs, args=types.SimpleNamespace(problem='mis'))
    g = Batch.from_data_list([ds[i] for i in range(a.graphs)]).to(dev)
    n = g.num_nodes
    A = torch.sparse_coo_tensor(g.edge_index, g.edge_weight, (n, n)).to_sparse_csr()
    x0 = g.x.float().view(-1, 1).expand(n, a.chains)
    beta = CategoricalDiffusion(T=a.T, schedule='linear').beta
    print(f'{a.graphs} graphs, {n} nodes, label density {g.x.float().mean().item():.3f}, {a.chains} chains')
    print('lambda      t   diff_from_label   density   conflicts/graph')
    for lam in a.lambdas:
        hf = HeatedForward(beta, lam, a.heat_until, a.beta)
        for t in a.ts:
            xt = hf.sample(A, g.b, x0, torch.full((n,), t, dtype=torch.long, device=dev))
            diff = (xt != x0).float().mean().item()
            dens = xt.mean().item()
            conf = ((A @ xt) * xt).sum().item() / 2 / a.graphs / a.chains
            print(f'{lam:6.1f} {t:6d}   {diff:15.4f}   {dens:7.4f}   {conf:15.1f}')


if __name__ == '__main__':
    main()
