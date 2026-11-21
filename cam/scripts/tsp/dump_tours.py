"""All tours a trained solver samples on a TSP instance file, as a tour-set label file (local addition,
UPSTREAM.md item 61).

Runs main.py's test (any --method; --test_path = the instance file) with the TSP env's evaluate wrapped so
that every decoded tour (after merge and --two_opt_iterations of 2-opt) is kept per instance; then writes the
distinct tours sorted by length in the format of utils/tour_sets.py.
    TOURS_OUT=FILE accelerate launch cam/scripts/tsp/dump_tours.py <main.py test arguments>
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import main as M  # noqa: E402
from envs import tsp as tsp_env  # noqa: E402
from utils.tour_sets import distinct_sorted, parse_points, points_digest  # noqa: E402

OUT = os.environ['TOURS_OUT']; KEPT = {}
_evaluate = tsp_env.Env.evaluate


def evaluate(self, graph, solution):
    costs, tours = _evaluate(self, graph, solution)
    i = int(torch.as_tensor(graph.idx).reshape(-1)[0])
    KEPT.setdefault(i, []).extend(tours.T.cpu().numpy().tolist())
    return costs, tours


tsp_env.Env.evaluate = evaluate
try:
    M.main()
finally:
    rank = int(os.environ.get('RANK', 0)); torch.save(KEPT, f'{OUT}.part{rank}')
if rank == 0:
    world = int(os.environ.get('WORLD_SIZE', 1))
    while not all(os.path.exists(f'{OUT}.part{r}') for r in range(world)):
        time.sleep(1)
    time.sleep(3); kept = {}
    for r in range(world):
        for i, t in torch.load(f'{OUT}.part{r}', weights_only=False).items():
            kept.setdefault(i, []).extend(t)
    lines = open(sys.argv[sys.argv.index('--test_path') + 1]).read().splitlines()
    assert sorted(kept) == list(range(len(lines))), 'not every instance was sampled'
    tours, lengths = [], []
    for i, l in enumerate(lines):
        t, ln = distinct_sorted(parse_points(l), kept[i]); tours.append(t); lengths.append(ln)
    meta = {'source': 'dump_tours', 'argv': sys.argv[1:]}
    torch.save({'points_digest': points_digest(lines), 'tours': tours, 'lengths': lengths, 'meta': meta}, OUT)
    for r in range(world):
        os.remove(f'{OUT}.part{r}')
    m = np.array([t.shape[0] for t in tours])
    print(f'saved {len(lines)}: distinct tours/instance mean {m.mean():.1f} min {m.min()}, '
          f'best length {np.mean([x[0].item() for x in lengths]):.4f}')
