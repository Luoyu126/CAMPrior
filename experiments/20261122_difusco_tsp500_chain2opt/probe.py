"""main.py test with per-instance logging of the in-chain 2-opt samplers on TSP (DIFUSCO renoise_decode,
CAM / DiffUCO --ul_chain_decode).

For every test instance one JSON line goes to $PROBE_OUT:
  idx, nofinal (best of the samples' tours without the final 2-opt), final (with --two_opt_iterations of 2-opt),
  chain_iters (2-opt moves summed over the decode calls of the chain), final_iters (moves of the final 2-opt).
Moves are counted as in envs/tsp.batched_two_opt_torch: one move = one batched step over all samples.
    PROBE_OUT=FILE python probe.py <main.py test arguments>
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cam'))
import main as M  # noqa: E402
from envs import tsp as E  # noqa: E402
from solvers import difusco as D  # noqa: E402
from solvers import meta_solver as MS  # noqa: E402

OUT = os.environ['PROBE_OUT']; S = {'in_decode': False, 'iters': 0, 'chain': 0}

_two = E.batched_two_opt_torch
def two(*a, **k):
    tour, n = _two(*a, **k); S['iters'] += n; return tour, n
E.batched_two_opt_torch = two

def in_decode(f):
    def g(self, *a, **k):
        S['in_decode'] = True
        try:
            return f(self, *a, **k)
        finally:
            S['in_decode'] = False; S['chain'] += S['iters']; S['iters'] = 0
    return g
D.DIFUSCO.decode_tour = in_decode(D.DIFUSCO.decode_tour)            # DIFUSCO renoise_decode
MS.ULSolver.chain_decode_tour = in_decode(MS.ULSolver.chain_decode_tour)  # CAM / DiffUCO --ul_chain_decode

_evaluate = E.Env.evaluate
def evaluate(self, graph, solution):
    if S['in_decode']:
        return _evaluate(self, graph, solution)
    two_opt = self.two_opt_iterations
    self.two_opt_iterations = 0
    try:
        c0, _ = _evaluate(self, graph, solution)
    finally:
        self.two_opt_iterations = two_opt
    S['iters'] = 0
    costs, tours = _evaluate(self, graph, solution)
    with open(OUT, 'a') as f:
        f.write(json.dumps({'idx': int(torch.as_tensor(graph.idx).reshape(-1)[0]), 'nofinal': c0.min().item(),
                            'final': costs.min().item(), 'chain_iters': S['chain'], 'final_iters': S['iters']}) + '\n')
    S['chain'] = 0; S['iters'] = 0
    return costs, tours
E.Env.evaluate = evaluate

M.main()
