"""Tests for solvers/block_labels.py and --label_mode block of solvers/cam_prior.py. CPU only.

    source cam/scripts/env.sh
    $CAM_PYTHON cam/tests/test_block_labels.py

(Own runner like test_sim_labels.py; the test_* functions are also pytest-compatible.)
"""
import os
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''          # CPU only, before torch is imported
os.environ.setdefault('WANDB_MODE', 'disabled')
CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAM_ROOT)
sys.path.insert(0, os.path.join(CAM_ROOT, 'tests'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from solvers.block_labels import block_labels, block_moves  # noqa: E402
from solvers.sim_labels import (SimLabelPool, complement_adjacency, compute_task,  # noqa: E402
                                greedy_decode_size)

torch.set_num_threads(2)
SETTINGS = dict(d=3, eps=0.0, beta_prior=1.02)


def er_graph(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    up = torch.triu((torch.rand(n, n, generator=g) < p).float(), 1)
    A = up + up.T
    return A, A.nonzero().T.contiguous(), -torch.ones(n, 1)


def naive_block(conf, x, i):
    """B(x, i) with Python loops (module docstring of solvers/block_labels.py)."""
    n = len(x)
    y = [int(v) for v in x]
    y[i] = 1
    for j in range(n):
        if conf[i, j]:
            y[j] = 0
    while True:
        cand = [j for j in range(n) if y[j] == 0 and not any(conf[j, k] and y[k] for k in range(n))]
        if not cand:
            break
        score = [sum(1 for k in cand if not conf[j, k]) for j in cand]
        y[cand[int(np.argmax(score))]] = 1               # first maximum: lowest index
    return torch.tensor(y, dtype=torch.float32)


def test_1_block_moves_equal_naive_and_are_maximal():
    g = torch.Generator().manual_seed(0)
    for n, p, comp in ((14, 0.3, False), (16, 0.7, True)):
        A, _, _ = er_graph(n, p, n)
        Aconf = complement_adjacency(0.5*A) if comp else 0.5*A
        conf = Aconf > 0
        for _ in range(5):
            x = torch.randint(0, 2, (n,), generator=g).float()
            seeds = torch.nonzero(x == 0).reshape(-1)
            if len(seeds) == 0:
                continue
            B = block_moves(conf, x, seeds)
            for s, i in enumerate(seeds.tolist()):
                want = naive_block(conf, x, i)
                assert torch.equal(B[:, s], want), (i, B[:, s], want)
                assert B[i, s] == 1
        # from a conflict-free state: the block move is conflict-free and maximal
        x = torch.zeros(n)
        x[int(torch.randint(n, (1,), generator=g))] = 1
        seeds = torch.nonzero(x == 0).reshape(-1)
        B = block_moves(conf, x, seeds)
        for s in range(B.shape[1]):
            y = B[:, s]
            assert (y @ conf.float() @ y).item() == 0
            free = (y == 0) & ((conf.float() @ y) == 0)
            assert not free.any()


def naive_labels(A, b, x_t, starts, seeds, temps, d, eps, beta_prior, S, R_screen, R_eval, max_conf):
    """block_labels with explicit per-column loops and the same draws of the item generator."""
    n, C = x_t.shape
    K = len(temps)
    conf = A > 0
    out = torch.zeros(C, n)

    def step(x, j):
        grad = 2*beta_prior*(A@x) + b.reshape(-1)
        delta = grad*(2*x - 1)/2
        thr = torch.sort(delta, descending=True).values[min(d, n) - 1]
        return torch.sigmoid((delta - thr + eps)/temps[j])

    def gains(gen, x, moves, t, R):
        M = moves.shape[1]
        U = [torch.rand(M*R, n, generator=gen) for _ in range(t, K)]
        res = torch.zeros(M, R)
        for mi in range(M):
            for r in range(R):
                xf, xm = x.clone(), moves[:, mi].clone()
                for k, j in enumerate(range(t, K)):
                    u = U[k][mi*R + r]
                    xf = torch.where(u < step(xf, j), 1 - xf, xf)
                    if j > t:
                        xm = torch.where(u < step(xm, j), 1 - xm, xm)
                res[mi, r] = greedy_decode_size(A, xm.unsqueeze(1))[0] - greedy_decode_size(A, xf.unsqueeze(1))[0]
        return res

    for c in range(C):
        gen = torch.Generator().manual_seed(int(seeds[c]))
        x, t = x_t[:, c], int(starts[c])
        cand = torch.nonzero((x == 0) & ((conf.float() @ x) <= max_conf)).reshape(-1)
        if len(cand) == 0:
            continue
        pick = cand[torch.randperm(len(cand), generator=gen)[:S]]
        moves = torch.stack([naive_block(conf, x, int(i)) for i in pick], 1)
        F = moves[:, int(torch.argmax(gains(gen, x, moves, t, R_screen).mean(1)))]
        G = gains(gen, x, F.unsqueeze(1), t, R_eval).mean()
        out[c] = G*(F != x).float()
    return out


def test_2_labels_equal_naive():
    n, K, C = 10, 5, 6
    A, _, b = er_graph(n, 0.6, 3)
    Ac = complement_adjacency(0.5*A)
    temps = [0.5*(1 - j/K) + 0.05 for j in range(K)]
    g = torch.Generator().manual_seed(4)
    x_t = torch.randint(0, 2, (n, C), generator=g).float()
    starts = torch.randint(0, K, (C,), generator=g)
    seeds = torch.randint(0, 2**40, (C,), generator=g)
    kw = dict(S=3, R_screen=2, R_eval=3, max_conf=2)
    fast = block_labels(Ac, b, x_t, starts, seeds, temps, **SETTINGS, **kw)
    slow = naive_labels(Ac, b, x_t, starts, seeds, temps, **SETTINGS, **kw)
    assert torch.allclose(fast, slow, atol=1e-6), (fast, slow)
    # label structure: one value G per item, on the flipped nodes only
    for c in range(C):
        nz = fast[c][fast[c] != 0]
        assert len(nz) == 0 or torch.allclose(nz, nz[0].expand_as(nz))


def test_3_last_step_and_no_seed():
    """t = K - 1: the move is not followed by any step, the factual state takes one RLSA step.
    A state without an admissible seed gets zero labels."""
    n, K = 8, 4
    A = torch.ones(n, n) - torch.eye(n)                  # complete graph: MIS = one node
    b = -torch.ones(n, 1)
    x_full = torch.ones(n, 1)                            # every node selected: no unselected seed
    lab = block_labels(0.5*A, b, x_full, torch.tensor([1]), torch.tensor([5]), [0.3]*K, **SETTINGS)
    assert (lab == 0).all()
    x_empty = torch.zeros(n, 1)                          # empty set: every node is a seed, its block = {seed}
    lab = block_labels(0.5*A, b, x_empty, torch.tensor([K - 1]), torch.tensor([7]), [1e-4]*K,
                       **SETTINGS, S=4, R_screen=2, R_eval=2)
    # the move ends at a single node (size 1); whatever the factual step flips, the greedy decoding
    # on the complete graph keeps one node: gain 0
    assert torch.allclose(lab, torch.zeros_like(lab))


def _task(n=12, K=5, C=10, seed=9, complement=True):
    A, ei, b = er_graph(n, 0.6, seed)
    g = torch.Generator().manual_seed(seed + 1)
    temps = [0.4*(1 - j/K) + 0.05 for j in range(K)]
    common = dict(kind='block', edge_index=ei.numpy(), edge_weight=np.full(ei.shape[1], 0.5, dtype=np.float32),
                  n=n, b=b.numpy(), temps=temps, graph_key=1, complement=complement, beta=1.02,
                  block_seeds=4, block_screen_R=3, block_eval_R=4, block_max_conf=2, **SETTINGS)
    x_t = torch.randint(0, 2, (n, C), generator=g).to(torch.uint8).numpy()
    starts = torch.randint(0, K, (C,), generator=g).numpy()
    seeds = torch.randint(0, 2**40, (C,), generator=g).numpy()
    arrs = (x_t, np.zeros((C, n), dtype=np.int64), np.ones((C, n), dtype=np.float32), starts, seeds)
    return common, arrs, A


def test_4_task_complement_grouping_and_pool():
    common, arrs, A = _task()
    full = compute_task(dict(common, x_next=arrs[0], sel=arrs[1], sign=arrs[2], starts=arrs[3],
                             seeds=arrs[4]), torch.device('cpu'))
    direct = block_labels(complement_adjacency(0.5*A), -torch.ones(len(A), 1), torch.as_tensor(arrs[0]).float(),
                          torch.as_tensor(arrs[3]), torch.as_tensor(arrs[4]), common['temps'], **SETTINGS,
                          S=4, R_screen=3, R_eval=4, max_conf=2)
    assert np.allclose(full, direct.numpy(), atol=1e-6)
    h = len(arrs[3])//2
    parts = [compute_task(dict(common, x_next=arrs[0][:, sl], sel=arrs[1][sl], sign=arrs[2][sl],
                               starts=arrs[3][sl], seeds=arrs[4][sl]), torch.device('cpu'))
             for sl in (slice(0, h), slice(h, None))]
    assert np.allclose(np.concatenate(parts), full, atol=1e-6)
    pool = SimLabelPool(['cpu', 'cpu', 'cpu'], fallback=None, timeout=60, nthreads=1, log=lambda *_: None)
    try:
        assert np.allclose(pool.compute(common, *arrs), full, atol=1e-6)
    finally:
        pool.close()


def test_5_camprior_block_path():
    import test_cam_prior as tcp
    from solvers.cam_prior import CAMPrior
    with tcp.TmpDir() as tmp:
        data = tcp.write_dataset(os.path.join(tmp, 'g'), [13], 0.6, 2)
        graph = tcp.load_batch(data)
        for problem in ('mcl', 'mis'):
            solver = tcp.build(CAMPrior, data, problem=problem, label_mode='block', label_states=0.25,
                               block_seeds=3, block_screen_R=2, block_eval_R=2, prior_recenter='on',
                               prior_d=3, save_dir=os.path.join(tmp, 'ck'))
            torch.manual_seed(1)
            out = solver.generate_data(graph)
            T, Kc, n = solver.num_tp, solver.num_kp, 13
            tgt, mask = torch.stack([o[3] for o in out]), torch.stack([o[5] for o in out])
            assert torch.isfinite(tgt).all() and (tgt[mask == 0] == 0).all()
            labelled = round(0.25*T*Kc)
            assert int((mask.sum(1) > 0).sum()) == labelled and (mask.sum(1)[mask.sum(1) > 0] == n).all()
            assert solver.label_scale == T*Kc/labelled
            loss = solver.train_batch(graph, [torch.stack(c) for c in zip(*[out[i] for i in range(6)])])
            loss.backward()
            assert torch.isfinite(loss)
        # a batch of two graphs: each graph's labels equal block_labels on its own complement graph,
        # and land on its rows of the buffer
        data2 = tcp.write_dataset(os.path.join(tmp, 'g2'), [13, 9], 0.6, 5)
        graph2 = tcp.load_batch(data2)
        import solvers.cam_prior as cp
        seen = []
        orig = cp.simulated_labels

        def spy(pool, device, common, x_next, sel, sign, starts, seeds):
            lab = orig(pool, device, common, x_next, sel, sign, starts, seeds)
            seen.append(dict(common=common, x=x_next, starts=starts, seeds=seeds, lab=lab))
            return lab
        cp.simulated_labels = spy
        try:
            solver = tcp.build(CAMPrior, data2, problem='mcl', label_mode='block', label_states=0.25,
                               block_seeds=3, block_screen_R=2, block_eval_R=2, prior_d=3,
                               save_dir=os.path.join(tmp, 'ck2'))
            torch.manual_seed(2)
            out = solver.generate_data(graph2)
        finally:
            cp.simulated_labels = orig
        assert len(seen) == 2
        A = graph2.A.to_dense()
        ptr = graph2.ptr.tolist()
        T, Kc = solver.num_tp, solver.num_kp
        for gi, rec in enumerate(seen):
            lo, hi = ptr[gi], ptr[gi + 1]
            assert rec['common']['n'] == hi - lo
            want = block_labels(complement_adjacency(A[lo:hi, lo:hi]), graph2.b[lo:hi],
                                torch.as_tensor(rec['x']).float(), torch.as_tensor(rec['starts']),
                                torch.as_tensor(rec['seeds']), rec['common']['temps'], d=3, eps=solver.prior_eps,
                                beta_prior=solver.prior_beta, S=3, R_screen=2, R_eval=2, max_conf=2)
            assert np.allclose(rec['lab'], want.numpy(), atol=1e-6)
            assert list(rec['starts']) == sorted(rec['starts'])
            for q, t_ in enumerate(rec['starts']):
                # stored state of item q: buffer row c*T + t (chain-major, one clip); find the chain
                rows = [out[c*T + int(t_)] for c in range(Kc)]
                match = [r for r in rows if np.array_equal(r[0][lo:hi].numpy().astype(np.uint8), rec['x'][:, q])
                         and r[5].sum() > 0]
                assert match and any(np.allclose(r[3][lo:hi].numpy(), rec['lab'][q], atol=1e-6) for r in match)
        assert seen[0]['seeds'][0] != seen[1]['seeds'][0]
        # refused settings
        for over in (dict(label_frac=0.5), dict(problem='mcut')):
            try:
                tcp.build(CAMPrior, data, label_mode='block', **over)
            except (ValueError, NotImplementedError):
                continue
            raise AssertionError(f'block labels accepted {over}')


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
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
