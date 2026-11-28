"""Route 1' (DSBM, docs/20261128/plan_route1prime.md) pieces. CPU only, cam environment:

    $CAM_PYTHON cam/tests/test_dsbm.py

1. Weighted pi1 for the backward network's starts and the independent cache: sb_coupling.label_log_weights equals
   the weighting of precompute_logg.py, and DIFUSCO.reverse_starts / imf_cache.py independent draw the labels with
   these weights (chi-square). Negative control: the uniform draw is rejected against the weighted target.
2. IPF initial cache (imf_cache.py ipf): y ~ pi1, z = y through the reference to the noise end, i.e. every bit kept with
   probability (1 + a_T) / 2, a_T = tanh(omega / 2); omega = 0 gives z independent of y.
3. Training-dynamics metrics (scripts/sb/dsbm_metrics.py) on synthetic traces with known answers: exact draws from
   pi_B give in_labels 1, d_near 0 and tv_near at the i.i.d. floor; a collapsed sampler is far above it; uniform
   independent starts give the Bernoulli statistics and ber_gap at its floor, a copy of y gives agree 1. Node order:
   the KaMIS labels of real training graphs have no conflicting edge under graph_edges.
4. Iteration driver (scripts/sb/dsbm_loop.py): the stopping rule (at least min_iters; 3 changes within 0.03 for the
   validation statistic and 1 % for both cache costs; max_iters; target), the validation statistic read from a
   training log (mean of the last 10 epochs), and the cache cost.
5. MCMC end distributions (scripts/sb/mcmc_labels.py): without the radius constraint the chain's long-run
   frequencies match pi_B(s) ∝ exp(|s| / T) over all independent sets of a small graph (exact enumeration, TV);
   with a small radius every sample is an independent set within the radius of a KaMIS solution. Negative
   control: the frequencies are far from the uniform law over independent sets.
"""
import os
import subprocess
import sys
import time
import traceback
import types
import json

os.environ['CUDA_VISIBLE_DEVICES'] = ''
CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(CAM_ROOT)
sys.path.insert(0, CAM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from solvers.difusco import DIFUSCO  # noqa: E402
from utils import sb_coupling as sb  # noqa: E402

torch.set_num_threads(4)
SCRATCH = os.path.join(REPO, 'runs', 'scratch', 'dsbm', 'tests')
os.makedirs(SCRATCH, exist_ok=True)


def chisq_p(counts, p):
    """p-value of a chi-square goodness-of-fit test of integer counts against probabilities p."""
    counts = np.asarray(counts, dtype=float); p = np.asarray(p, dtype=float)
    e = counts.sum() * p
    return 1 - chi2.cdf(((counts - e) ** 2 / e).sum(), len(p) - 1)


def toy_sets(seed=0, n_graphs=3):
    g = torch.Generator().manual_seed(seed)
    sets = []
    for k in range(n_graphs):
        n = 12 + k
        S = torch.unique((torch.rand(6, n, generator=g) < 0.3 + 0.1 * torch.rand(6, 1, generator=g)).to(torch.uint8), dim=0)
        sets.append(S)
    return sets


def row_index(S, X):
    """For every row of X (K, n) the index of the equal row of S (M, n)."""
    eq = (X[:, None, :] == S[None].to(X.dtype)).all(-1)
    assert bool(eq.any(1).all())
    return eq.float().argmax(1)


def test_label_log_weights():
    S = toy_sets()[0]
    u = sb.label_log_weights(S).exp()
    assert torch.allclose(u, torch.full_like(u, 1 / S.shape[0]))
    w = sb.label_log_weights(S, 0.5).exp()
    ref = torch.softmax(S.double().sum(1) / 0.5, 0)            # precompute_logg.label_target
    assert torch.allclose(w, ref, atol=1e-12) and abs(w.sum().item() - 1) < 1e-12
    assert sb.parse_temp('none') is None and sb.parse_temp(None) is None and sb.parse_temp('0.5') == 0.5


def test_reverse_starts_weighted():
    sets = toy_sets()
    path = os.path.join(SCRATCH, 'starts.pt')
    torch.save({'sets': sets}, path)
    for temp in ('0.5', 'none'):
        os.environ['CAM_SB_STARTS'] = path
        os.environ['CAM_SB_STARTS_TEMP'] = temp
        dummy = types.SimpleNamespace()
        for gi, S in enumerate(sets):
            graph = types.SimpleNamespace(idx=torch.tensor([gi]))
            sol = torch.zeros(S.shape[1], 20000)
            X = DIFUSCO.reverse_starts(dummy, graph, sol).T
            c = np.bincount(row_index(S, X).numpy(), minlength=S.shape[0])
            p = sb.label_log_weights(S, sb.parse_temp(temp)).exp().numpy()
            assert chisq_p(c, p) > 1e-3, (temp, gi, c, p)
            if temp == '0.5':   # negative control: the weighted draw is not uniform
                assert chisq_p(c, np.full(S.shape[0], 1 / S.shape[0])) < 1e-6
    del os.environ['CAM_SB_STARTS'], os.environ['CAM_SB_STARTS_TEMP']


def test_imf_cache_independent_weighted():
    sets = toy_sets(1)
    ls = os.path.join(SCRATCH, 'ls.pt')
    torch.save({'sets': sets}, ls)
    fwd, bwd = os.path.join(SCRATCH, 'f0.pt'), os.path.join(SCRATCH, 'b0.pt')
    subprocess.run([sys.executable, os.path.join(CAM_ROOT, 'scripts/sb/imf_cache.py'), 'independent', '--label_set', ls,
                    '--J', '20000', '--temp', '0.5', '--out_fwd', fwd, '--out_bwd', bwd], check=True,
                   stdout=subprocess.DEVNULL)
    F, B = torch.load(fwd, weights_only=False), torch.load(bwd, weights_only=False)
    for gi, S in enumerate(sets):
        Z, Y = F['starts'][gi], F['labels'][gi]
        assert torch.equal(B['starts'][gi], Y) and torch.equal(B['labels'][gi], Z)
        c = np.bincount(row_index(S, Y).numpy(), minlength=S.shape[0])
        assert chisq_p(c, sb.label_log_weights(S, 0.5).exp().numpy()) > 1e-3
        assert abs(Z.float().mean().item() - 0.5) < 0.01          # z ~ Bernoulli(0.5), independent of y
        assert abs(((Z == Y).float().mean().item()) - 0.5) < 0.01


def test_imf_cache_ipf():
    sets = toy_sets(2)
    ls = os.path.join(SCRATCH, 'ls_ipf.pt')
    torch.save({'sets': sets}, ls)
    for omega in (0.35, 1.0, 0.0):
        fwd, bwd = os.path.join(SCRATCH, 'fi.pt'), os.path.join(SCRATCH, 'bi.pt')
        subprocess.run([sys.executable, os.path.join(CAM_ROOT, 'scripts/sb/imf_cache.py'), 'ipf', '--omega', str(omega),
                        '--label_set', ls, '--J', '20000', '--temp', '1', '--out_fwd', fwd, '--out_bwd', bwd],
                       check=True, stdout=subprocess.DEVNULL)
        F, B = torch.load(fwd, weights_only=False), torch.load(bwd, weights_only=False)
        a_T = np.tanh(omega / 2)
        assert abs(F['meta']['a_T'] - a_T) < 1e-5, (F['meta']['a_T'], a_T)
        for gi, S in enumerate(sets):
            Z, Y = F['starts'][gi], F['labels'][gi]
            assert torch.equal(B['starts'][gi], Y) and torch.equal(B['labels'][gi], Z)
            c = np.bincount(row_index(S, Y).numpy(), minlength=S.shape[0])
            assert chisq_p(c, sb.label_log_weights(S, 1.0).exp().numpy()) > 1e-3
            keep = (Z == Y).double().mean().item()
            assert abs(keep - (1 + a_T) / 2) < 0.005, (omega, keep, (1 + a_T) / 2)


def _hex(m):
    return [format(int(''.join(map(str, r.tolist())), 2), 'x') for r in m]


def test_dsbm_metrics_synthetic():
    sys.path.insert(0, os.path.join(CAM_ROOT, 'scripts', 'sb'))
    import dsbm_metrics as dm
    sets = toy_sets(3, n_graphs=40)
    rng = np.random.default_rng(0)
    exact, collapsed = {}, {}
    for i, S in enumerate(sets):
        p = sb.label_log_weights(S, 1.0).exp().numpy()
        Y = S.numpy()[rng.choice(len(p), 64, p=p)].astype(np.int8)
        Z = rng.integers(0, 2, Y.shape).astype(np.int8)
        exact[i] = (Z, Y)
        collapsed[i] = (Z, np.repeat(S.numpy()[:1].astype(np.int8), 64, 0))
    m = dm.fwd_metrics(exact, sets, 1.0)
    assert m['in_labels'] == 1.0 and m['d_near'] == 0.0
    assert abs(m['tv_near'] - m['tv_floor']) < 0.03, m
    c = dm.fwd_metrics(collapsed, sets, 1.0)
    assert c['tv_near'] > m['tv_floor'] + 0.2, (c, m)
    # backward: z uniform and independent of the starting label y, or z = y
    edges, uni, copy = {}, {}, {}
    for i, S in enumerate(sets):
        n = S.shape[1]
        e = np.array([(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.3]).T
        edges[i] = (e[0], e[1])
        Y = S.numpy()[rng.integers(0, len(S), 64)].astype(np.int8)
        uni[i] = (Y, rng.integers(0, 2, Y.shape).astype(np.int8))
        copy[i] = (Y, Y.copy())
    b = dm.bwd_metrics(uni, edges)
    assert abs(b['density'] - 0.5) < 0.01 and abs(b['conflict'] - 0.25) < 0.01 and abs(b['agree'] - 0.5) < 0.01, b
    assert abs(b['ber_gap'] - b['bgap_floor']) < 0.01, b
    assert dm.bwd_metrics(copy, edges)['agree'] == 1.0
    # trace round trip through the hex format of solvers/difusco.py write_pair_trace
    path = os.path.join(SCRATCH, 'trace.jsonl')
    with open(path, 'w') as f:
        for i, (Z, Y) in exact.items():
            f.write(json.dumps({'idx': i, 'n': Z.shape[1], 'start': _hex(Z), 'final': _hex(Y)}) + '\n')
    tr = dm.read_trace(path, 1000)
    assert all(np.array_equal(tr[i][0], exact[i][0]) and np.array_equal(tr[i][1], exact[i][1]) for i in exact)


def test_graph_edges_node_order():
    sys.path.insert(0, os.path.join(CAM_ROOT, 'scripts', 'sb'))
    import dsbm_metrics as dm
    sets = torch.load('/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/matched_kamis_loaderorder.pt',
                      weights_only=False)['sets']
    E = dm.graph_edges(dm.TRAIN, range(5))
    for i in range(5):
        S = sets[i].numpy()
        e0, e1 = E[i]
        assert len(e0) > 0 and int(max(e0.max(), e1.max())) < S.shape[1]
        assert (S[:, e0] * S[:, e1]).sum() == 0, f'graph {i}: a KaMIS label has a conflicting edge'
        Z = (np.random.default_rng(i).random(S.shape) < 0.5).astype(np.int8)   # a random state does conflict
        assert (Z[:, e0] * Z[:, e1]).sum() > 0


def test_stop_rule_and_logs():
    sys.path.insert(0, os.path.join(CAM_ROOT, 'scripts', 'sb'))
    import dsbm_loop as dl
    flat = [20.0] * 12
    cost = [100.0] * 12
    # never before min_iters, even if flat
    assert not dl.stop_rule(flat, cost, cost, 8, 10, 30)[0]
    assert dl.stop_rule(flat, cost, cost, 9, 10, 30)[0]
    # a 0.04 move inside the last 3 changes blocks; outside them it does not
    v = [20.0] * 12; v[8] = 20.04
    assert not dl.stop_rule(v, cost, cost, 9, 10, 30)[0]
    v = [20.0] * 12; v[5] = 20.04
    assert dl.stop_rule(v, cost, cost, 9, 10, 30)[0]
    # cost moving 2 % blocks (either cache); None costs (frozen mode) are ignored
    c = list(cost); c[9] = 102.0
    assert not dl.stop_rule(flat, cost, c, 9, 10, 30)[0]
    assert dl.stop_rule(flat, [None] * 12, [None] * 12, 9, 10, 30)[0]
    # drifting validation keeps going until max_iters
    drift = [20 + 0.05 * j for j in range(30)]
    assert not dl.stop_rule(drift, cost * 3, cost * 3, 20, 10, 30)[0]
    assert dl.stop_rule(drift, cost * 3, cost * 3, 29, 10, 30)[0]
    # target overrides everything
    assert not dl.stop_rule(flat, cost, cost, 9, 10, 30, target=12)[0]
    assert dl.stop_rule(drift, cost * 3, cost * 3, 11, 10, 30, target=12)[0]
    # validation statistic from a training log
    log = os.path.join(SCRATCH, 'train.log')
    with open(log, 'w') as f:
        for e in range(15):
            f.write(f'Epoch: {e} train/loss: 0.1\n  1%| progress\n')
            f.write(f'Epoch: {e} eval/object: {-(20 + 0.01 * e):.6f} eval/time: 0.7 eval/mis_size: {20 + 0.01 * e}\n')
    m, n = dl.val_stat(log)
    assert n == 15 and abs(m - (20 + 0.01 * np.mean(range(5, 15)))) < 1e-9
    # cache cost
    path = os.path.join(SCRATCH, 'cache.pt')
    torch.save({'starts': [torch.zeros(4, 10, dtype=torch.uint8)], 'labels': [torch.ones(4, 10, dtype=torch.uint8)]}, path)
    assert dl.cache_cost(path) == 10.0


def test_mcmc_labels():
    sys.path.insert(0, os.path.join(CAM_ROOT, 'scripts', 'sb'))
    import itertools
    import mcmc_labels as mc
    rng = np.random.default_rng(0)
    n = 8
    edges = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.35]
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    states = [np.array(x, dtype=np.int8) for x in itertools.product((0, 1), repeat=n)
              if all(not (x[a] and x[b]) for a, b in edges)]
    sizes = np.array([x.sum() for x in states])
    big = states[int(np.argmax(sizes))]
    K = np.stack([big, states[len(states) // 2]])
    key = {x.tobytes(): j for j, x in enumerate(states)}
    for T in (1.0, 0.5):
        S, acc = mc.run_chain(adj, K, K[0], n, T, 2000, 3, 60000, rng)
        freq = np.bincount([key[x.tobytes()] for x in S], minlength=len(states)) / len(S)
        exact = np.exp(sizes / T); exact /= exact.sum()
        tv = 0.5 * np.abs(freq - exact).sum()
        assert tv < 0.03, (T, tv, acc)
        unif = np.full(len(states), 1 / len(states))
        assert 0.5 * np.abs(freq - unif).sum() > 0.1
    S, _ = mc.run_chain(adj, K, K[0], 2, 0.5, 500, 2, 5000, rng)
    e = np.array(edges).T
    assert (S[:, e[0]] * S[:, e[1]]).sum() == 0
    assert ((S[:, None, :] != K[None]).sum(-1).min(1) <= 2).all()


tests = [(n, f) for n, f in list(globals().items()) if n.startswith('test_') and callable(f)]


def main():
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f'PASS {name} ({time.time()-t0:.1f}s)', flush=True)
        except Exception:
            failed += 1
            print(f'FAIL {name}\n{traceback.format_exc()}', flush=True)
    print(f'{len(tests)-failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
