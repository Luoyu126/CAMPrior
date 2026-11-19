"""All samples of a trained DIFUSCO checkpoint as labels (local addition, UPSTREAM.md item 48).

Runs the usual DIFUSCO test (main.py arguments, --test_path = the training graphs, any --num_t) and keeps, for
every graph, all --num_k greedy-decoded samples (sets and sizes). Used to distil many-step sampling into a
model tested with fewer steps (sequence-level KD, Kim & Rush 2016).
    SL_SAMPLES_OUT=FILE accelerate launch scripts/mis/sl_samples.py <main.py test arguments>
"""
import glob, os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import main as M
import envs.mis as E
OUT = os.environ['SL_SAMPLES_OUT']; KEPT = {}
orig = E.Env.evaluate
def ev(self, graph, solution):
    energy, sol = orig(self, graph, solution)
    s = sol.float(); s = s.T if s.shape[0] == graph.num_nodes else s
    KEPT[int(torch.as_tensor(graph.idx).reshape(-1)[0])] = dict(sizes=s.sum(1).cpu(), sets=s.to(torch.uint8).cpu(), kamis=int(graph.x.sum()))
    return energy, sol
E.Env.evaluate = ev
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
    assert sorted(kept) == list(range(len(files))), (len(kept), len(files))
    torch.save({'files': files, 'samples': [kept[i] for i in range(len(files))]}, OUT)
    for r in range(world): os.remove(f'{OUT}.part{r}')
    print('saved', len(files))
