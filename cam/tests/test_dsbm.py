"""Route 1' (DSBM, docs/20261128/plan_route1prime.md) pieces. CPU only, cam environment:

    $CAM_PYTHON cam/tests/test_dsbm.py

1. Weighted pi1 for the backward network's starts and the independent cache: sb_coupling.label_log_weights equals
   the weighting of precompute_logg.py, and DIFUSCO.reverse_starts / imf_cache.py independent draw the labels with
   these weights (chi-square). Negative control: the uniform draw is rejected against the weighted target.
"""
import os
import subprocess
import sys
import time
import traceback
import types

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
