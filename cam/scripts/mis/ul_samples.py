"""All samples of an unsupervised solver as labels (local addition, UPSTREAM.md item 47).

Runs the usual test of a trained DiffUCO checkpoint (main.py arguments, --test_path = the graphs) and keeps,
for every graph, all --num_k greedy-decoded samples as unique sets with their counts and sizes
(KEEP_SETS=0: sizes only). Output {'files', 'samples': [{'sizes', 'sets', 'counts', 'kamis'}]}; expand
with sets.repeat_interleave(counts, 0) for a label set file of --label_choice random / tilted.
    UL_SAMPLES_OUT=FILE accelerate launch scripts/mis/ul_samples.py <main.py test arguments>
"""
import glob, os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import main as M
from solvers.diffuco import DiffUCO
OUT = os.environ['UL_SAMPLES_OUT']; KEEP_SETS = os.environ.get('KEEP_SETS', '1') == '1'; KEPT = {}
def evaluate_single(self, graph):
    graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, torch.Size((graph.num_nodes, graph.num_nodes))).to_sparse_csr()
    sol = torch.randint(0, 2, size=(graph.num_nodes, self.num_k), device=self.accelerator.device, dtype=torch.float)
    for t in range(self.num_t):
        time_ = torch.ones_like(sol) * int(t * self.num_tp / self.num_t) / self.num_tp
        fp = torch.sigmoid(self.model(graph, sol.unsqueeze(-1), time_).squeeze(-1))
        sol = sol * (1 - fp) + fp * (1 - sol) if t == self.num_t - 1 else torch.where(torch.rand_like(sol) < fp, 1 - sol, sol)
    energy, s = self.problem.evaluate(graph, sol)
    s = s.float(); s = s.T if s.shape[0] == graph.num_nodes else s
    i = int(torch.as_tensor(graph.idx).reshape(-1)[0])
    u, c = torch.unique(s.to(torch.uint8), dim=0, return_counts=True)
    KEPT[i] = dict(sizes=s.sum(1).cpu(), sets=u.cpu() if KEEP_SETS else None, counts=c.cpu(), kamis=int(graph.x.sum()))
    return energy.min()
DiffUCO.evaluate_single = torch.no_grad()(evaluate_single)
try: M.main()
finally:
    rank = int(os.environ.get('RANK', 0)); torch.save(KEPT, f'{OUT}.part{rank}')
if rank == 0:
    world = int(os.environ.get('WORLD_SIZE', 1))
    while not all(os.path.exists(f'{OUT}.part{r}') for r in range(world)): time.sleep(1)
    time.sleep(3); kept = {}
    for r in range(world): kept.update(torch.load(f'{OUT}.part{r}', weights_only=False))
    tp = sys.argv[sys.argv.index('--test_path') + 1]
    files = [os.path.basename(f) for f in sorted(glob.glob(tp + '/*gpickle'))][:len(kept)]
    assert sorted(kept) == list(range(len(files)))
    torch.save({'files': files, 'samples': [kept[i] for i in range(len(files))]}, OUT)
    for r in range(world): os.remove(f'{OUT}.part{r}')
    print('saved', len(files))
