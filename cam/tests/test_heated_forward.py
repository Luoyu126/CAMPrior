"""Tests for utils/heated_forward.py (energy-heated DIFUSCO forward). CPU only.

    source cam/scripts/env.sh
    $CAM_PYTHON cam/tests/test_heated_forward.py
"""
import os
import sys
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''
CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAM_ROOT)

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from envs.mis import Env as MISEnv  # noqa: E402
from utils.diffusion_schedulers import CategoricalDiffusion  # noqa: E402
from utils.heated_forward import HeatedForward  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(0)
PEN = 1.02
T = 1000
BETA = CategoricalDiffusion(T=T, schedule='linear').beta


def graph(n=40, p=0.15, seed=0):
    G = nx.gnp_random_graph(n, p, seed=seed)
    ei = torch.tensor([[u, v] for u, v in G.edges()] + [[v, u] for u, v in G.edges()], dtype=torch.long).t()
    A = torch.sparse_coo_tensor(ei, torch.ones(ei.shape[1]), (n, n)).to_sparse_csr()
    mis = nx.maximal_independent_set(G, seed=seed)
    x0 = torch.zeros(n, 1)
    x0[mis] = 1
    return G, A, -torch.ones(n, 1), x0


def flip_rate(hf, A, b, x0, t, K):
    n = x0.shape[0]
    xt = hf.sample(A, b, x0.expand(n, K), torch.full((n,), t, dtype=torch.long))
    return (xt != x0).float().mean().item()


def test_delta_energy_matches_env():
    G, A, b, x0 = graph()
    env = MISEnv(types.SimpleNamespace(beta=PEN, mis_decoding='continuous'))
    g = types.SimpleNamespace(A=A, b=b, batch=torch.zeros(x0.shape[0], dtype=torch.long))
    x = torch.bernoulli(torch.full((x0.shape[0], 1), 0.4))
    dE = HeatedForward.delta_energy(A, b, x, PEN)
    E0 = env.energy_func(g, x)[0].item()
    for i in range(x.shape[0]):
        y = x.clone()
        y[i] = 1 - y[i]
        assert abs(env.energy_func(g, y)[0].item() - E0 - dE[i, 0].item()) < 1e-4
    return 'dE equals the brute-force energy difference on all nodes'


def test_lambda0_is_uniform():
    _, A, b, x0 = graph()
    hf = HeatedForward(BETA, 0.0, 0.5, PEN)
    assert hf.H == 0
    out = []
    for t in (10, 300, 800):
        K = 4000
        want = (1 - np.prod(1 - BETA[:t])) / 2
        got = flip_rate(hf, A, b, x0, t, K)
        se = np.sqrt(want * (1 - want) / (K * x0.shape[0]))
        assert abs(got - want) < 5 * se + 1e-4, (t, got, want)
        out.append(f't={t}: {got:.4f} vs {want:.4f}')
    return 'lambda=0 matches the uniform closed form: ' + ', '.join(out)


def test_jump_equals_stepwise():
    """lambda>0: the closed-form uniform jump after H equals running the uniform steps one by one."""
    _, A, b, x0 = graph()
    hf = HeatedForward(BETA, 2.0, 0.3, PEN)
    t, K = 700, 4000
    n = x0.shape[0]
    got = flip_rate(hf, A, b, x0, t, K)
    x = x0.expand(n, K).clone()
    for s in range(1, t + 1):
        x = hf.step(A, b, x, s)  # lam(s) = 0 for s >= heat_T, i.e. beta_s / 2
    ref = (x != x0).float().mean().item()
    se = np.sqrt(2 * 0.25 / (K * n))
    assert abs(got - ref) < 5 * se, (got, ref)
    return f'jump {got:.4f} vs step-by-step {ref:.4f}'


def test_per_graph_times():
    """Two disjoint graphs in one call with t = 50 and t = 600 get the same law as separate calls."""
    _, A, b, x0 = graph()
    n, K = x0.shape[0], 3000
    hf = HeatedForward(BETA, 2.0, 0.5, PEN)
    ei = A.to_sparse_coo().coalesce().indices()
    ei2 = torch.cat([ei, ei + n], 1)
    A2 = torch.sparse_coo_tensor(ei2, torch.ones(ei2.shape[1]), (2 * n, 2 * n)).to_sparse_csr()
    x02 = torch.cat([x0, x0])
    t = torch.cat([torch.full((n,), 50), torch.full((n,), 600)]).long()
    out = hf.sample(A2, torch.cat([b, b]), x02.expand(2 * n, K), t)
    res = []
    for half, tt in ((slice(0, n), 50), (slice(n, 2 * n), 600)):
        rate = (out[half] != x0).float().mean().item()
        ref = flip_rate(hf, A, b, x0, tt, K)
        assert abs(rate - ref) < 0.01, (tt, rate, ref)
        res.append(f't={tt}: {rate:.4f} vs {ref:.4f}')
    return 'batched vs separate: ' + ', '.join(res)


def test_heated_leaves_label_faster():
    _, A, b, x0 = graph()
    uni = HeatedForward(BETA, 0.0, 0.5, PEN)
    hot = HeatedForward(BETA, 3.0, 0.5, PEN)
    r_u = flip_rate(uni, A, b, x0, 100, 3000)
    r_h = flip_rate(hot, A, b, x0, 100, 3000)
    assert r_h > r_u
    r_end = flip_rate(hot, A, b, x0, T, 3000)
    assert abs(r_end - 0.5) < 0.01
    return f't=100 distance from label: heated {r_h:.4f} > uniform {r_u:.4f}; t=T heated {r_end:.4f}'


if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                print(f'PASS {name}: {fn()}')
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f'FAIL {name}: {e!r}')
    sys.exit(fails)
