"""Labels from an unsupervised solver's own samples (local addition, UPSTREAM.md item 46).

Runs the usual test of a trained DiffUCO checkpoint (main.py arguments, --test_path = the training
graphs) and keeps, for every graph, the largest greedy-decoded set among its num_k samples (the first
one on ties). Writes a one-set-per-graph label file for DIFUSCO training with
--label_set FILE --label_choice first (the format of scripts/mis/build_solution_sets.py):
    UL_LABELS_OUT=FILE accelerate launch scripts/mis/ul_labels.py <main.py test arguments>
The KaMIS label size of each graph is stored next to the UL label size for comparison.
"""
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import main as M  # noqa: E402
from solvers.diffuco import DiffUCO  # noqa: E402

OUT = os.environ['UL_LABELS_OUT']
KEPT = {}


def evaluate_single(self, graph):
    A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight,
                                torch.Size((graph.num_nodes, graph.num_nodes))).to_sparse_csr()
    graph.A = A
    solution = torch.randint(0, 2, size=(graph.num_nodes, self.num_k), device=self.accelerator.device,
                             dtype=torch.float)
    for t in range(self.num_t):   # the DiffUCO sampler of ULSolver.evaluate_single
        time = torch.ones_like(solution) * int(t * self.num_tp / self.num_t) / self.num_tp
        flip_prob = torch.sigmoid(self.model(graph, solution.unsqueeze(-1), time).squeeze(-1))
        if t == self.num_t - 1:
            solution = solution * (1 - flip_prob) + flip_prob * (1 - solution)
        else:
            solution = torch.where(torch.rand_like(solution) < flip_prob, 1 - solution, solution)
    energy, sol = self.problem.evaluate(graph, solution)
    sol = sol.float()
    if sol.shape[0] == graph.num_nodes:   # [N, K] -> [K, N]
        sol = sol.T
    k = int(sol.sum(1).argmax())
    i = int(torch.as_tensor(graph.idx).reshape(-1)[0])
    KEPT[i] = (sol[k].to(torch.uint8).cpu(), int(graph.x.sum().item()))
    return energy.min()


def main():
    DiffUCO.evaluate_single = torch.no_grad()(evaluate_single)
    try:
        M.main()
    finally:
        rank = int(os.environ.get('RANK', 0))
        torch.save(KEPT, f'{OUT}.part{rank}')
    if rank != 0:
        return
    world = int(os.environ.get('WORLD_SIZE', 1))
    for r in range(world):   # wait for the other ranks' parts
        while not os.path.exists(f'{OUT}.part{r}'):
            time.sleep(1)
    time.sleep(2)
    kept = {}
    for r in range(world):
        kept.update(torch.load(f'{OUT}.part{r}', weights_only=False))
    test_path = sys.argv[sys.argv.index('--test_path') + 1]
    files = [os.path.basename(f) for f in sorted(glob.glob(test_path + '/*gpickle'))][:len(kept)]
    assert sorted(kept) == list(range(len(files))), 'some graphs are missing'
    sets = [kept[i][0][None] for i in range(len(files))]
    ul = torch.tensor([float(s.sum()) for s in sets]); kamis = torch.tensor([float(kept[i][1]) for i in range(len(files))])
    torch.save({'files': files, 'sets': sets, 'ul_size': ul, 'label_size': kamis,
                'meta': {'checkpoint_args': sys.argv[1:]}}, OUT)
    for r in range(world):
        os.remove(f'{OUT}.part{r}')
    print(f'{len(files)} graphs: UL label {ul.mean():.3f}, KaMIS label {kamis.mean():.3f}, '
          f'UL >= KaMIS on {(ul >= kamis).float().mean():.1%}')


if __name__ == '__main__':
    main()
