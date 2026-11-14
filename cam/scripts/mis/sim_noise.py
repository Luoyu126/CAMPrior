"""Noise of the simulated labels (--label_mode sim) and how it falls with R rollouts per label
(local addition, see cam/UPSTREAM.md).

Question (docs/status.md 2.11): CAM's label pushes CAMPrior on maximum clique towards "don't add";
the simulated label (solvers/sim_labels.py; DAM Prop. 2.3 / Eq. 12) sees "add, then repair", but one
rollout per label was too noisy on MIS (docs/status.md 2.8). How many rollouts per label does it
take, and does the averaged label still say "don't add"?

On a checkpoint, run one training rollout (current sampler, the epoch's temperature) per graph,
pick C stored states X_t spread over the steps and m nodes per state (uniform, as in training), and
for every (state, node) run R_max pairs of RLSA rollouts from the two candidate next states
(X_{t+1} as sampled, and X_{t+1} with the node's step-t decision reversed), over steps t+1 .. T-1
with the prior of CAMPrior (solver.prior_logits: same energy incl. the complement graph of maximum
clique, d, eps and temperatures). The two rollouts of a pair share their uniforms; pairs are
independent. One pair gives label = s * (g(Z_factual) - g(Z_toggled)), s = +1 if the node was not
flipped at step t, -1 if it was (a positive label favours flipping, as CAM's target), g = the
training energy (--objective decoded: -size of the greedily decoded clique, the validation
objective). Per (state, node):
    signal = mean over the R_max pairs,   noise = one pair's label - signal.
Reported per stage of t and per node class (unselected / selected at X_t): RMS signal, RMS noise of
one pair, the SNR of the mean of R pairs (signal power / (noise power / R)) for R = 1, 4, 8, 16, the
implied correlation of an R-pair label with the signal, sqrt(SNR / (1 + SNR)), the mean signal and
the fraction of positive signals (for unselected nodes: "adding helps"), and the same for CAM's
label grad_H(X_{T-1}) * (2 x_t - 1) with its correlation to the simulated signal.

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=7 $CAM_PYTHON cam/scripts/mis/sim_noise.py \
        --checkpoint runs/cam/mcl_rb_small_pilot/<exp>/epoch_0000.pt --epoch 1 --recenter on
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CAM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
STAGES = {'early (t < 17)': (0, 17), 'mid (17-33)': (17, 34), 'late (t >= 34)': (34, 50), 'all': (0, 50)}
RS = (1, 4, 8, 16)


class _Parser(Exception):
    pass


def main_parser():
    """The argument parser built inside main.main(), captured at its parse_args call."""
    import main as M

    def grab(self, *a, **k):
        raise _Parser(self)
    orig = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = grab
    try:
        M.main()
    except _Parser as e:
        return e.args[0]
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise RuntimeError('main.main() did not call parse_args')


@torch.no_grad()
def prior_rollout(solver, graph, X, j0, ratio, U, objective='energy', horizons=()):
    """Run the RLSA prior from X (n x B) over steps j0 .. T-1 with uniforms U (T x n x B); returns
    {h: g} with g (B,) of the state after h steps, for each h in horizons below the remaining steps,
    and h = -1 for the end state (the full rollout). g is the training energy, or with objective
    'decoded' the validation objective (problem.evaluate: greedy decoding to a clique, -size)."""
    def g(X):
        if objective == 'decoded':
            return solver.problem.evaluate(graph, X)[0].reshape(-1).float()
        return solver.problem.energy_func(graph, X)[0].reshape(-1)
    out = {}
    if 0 in horizons:
        out[0] = g(X)
    for j in range(j0, solver.num_tp):
        p = torch.sigmoid(solver.prior_logits(graph, X, j, ratio))
        X = torch.where(U[j] < p, 1 - X, X)
        if j - j0 + 1 in horizons and j < solver.num_tp - 1:
            out[j - j0 + 1] = g(X)
    out[-1] = g(X)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--epoch', type=int, required=True, help='epoch whose tau_train is used (0 = before training)')
    ap.add_argument('--recenter', default='off', choices=['on', 'off'])
    ap.add_argument('--prior_tau0', default='1')
    ap.add_argument('--prior_d', default='10')
    ap.add_argument('--graphs', type=int, default=4)
    ap.add_argument('--C', type=int, default=24, help='stored states per graph (steps spread over 0..T-2)')
    ap.add_argument('--m', type=int, default=16, help='nodes per stored state')
    ap.add_argument('--R', type=int, default=32, help='rollout pairs per (state, node)')
    ap.add_argument('--chunk', type=int, default=8192, help='rollout columns per batch')
    ap.add_argument('--objective', default='energy', choices=['energy', 'decoded'],
                    help='g of a rollout end state: training energy, or -size of the greedily decoded clique')
    ap.add_argument('--horizons', default='',
                    help='comma-separated truncation horizons H: also report labels with g taken after H '
                         'rollout steps (the full rollout if fewer remain), e.g. 0,2,5,10')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ck = os.path.abspath(a.checkpoint)
    out = os.path.abspath(a.out) if a.out else None
    sys.path.insert(0, CAM_DIR)
    os.chdir(CAM_DIR)
    sys.argv = ['main.py']
    from solvers.cam_prior import CAMPrior
    data = os.environ.get('CAM_RB_DIR', '/home/shengyuf/yunyi/data/CAM_data/RB/MIS')
    save = os.path.join(CAM_DIR, '..', 'runs', 'diag', 'sim_noise')
    argv = ['--train_path', f'{data}/RB_small_train', '--valid_path', f'{data}/RB_small_valid',
            '--test_path', f'{data}/RB_small_test', '--problem', 'mcl', '--valid_sample', '4',
            '--samples_per_epoch', '4000', '--batch_size', '1', '--mini_batch_size', '256',
            '--num_t', '50', '--num_k', '20', '--num_tp', '50', '--num_kp', '10', '--num_h', '64',
            '--num_l', '8', '--method', 'CAMPrior', '--model', 'EncodeProcessDecode', '--epochs', '100',
            '--N_anneal', '100', '--lr', '0.0005', '--prior_tau0', a.prior_tau0, '--prior_d', a.prior_d,
            '--prior_recenter', a.recenter, '--seed', '0', '--save_dir', save, '--do_train',
            '--checkpoint', ck, '--wandb_name', 'diag', '--wandb_run_name', 'sim_noise']
    args = main_parser().parse_args(argv)
    torch.manual_seed(a.seed)
    solver = CAMPrior(args)
    solver.model.to(solver.accelerator.device).eval()
    dev = solver.accelerator.device
    solver.tau = solver.temp_scheduler(a.epoch)
    ratio = solver.prior_tau_ratio()
    T, Kc, tau = solver.num_tp, solver.num_kp, float(solver.tau)
    print(f'checkpoint {ck}\nepoch {a.epoch}: tau_train {tau:.4f}, prior ratio {ratio:.3f}, recenter {a.recenter}, '
          f'{a.graphs} graphs x {a.C} states x {a.m} nodes x {a.R} rollout pairs, objective {a.objective}', flush=True)
    rng = np.random.default_rng(a.seed)
    rows = []   # (t, selected, signal, noise power of one pair, cam label)
    hs = sorted(int(h) for h in a.horizons.split(',') if h != '')
    hrows = {h: [] for h in hs}   # (t, signal, noise power, full-horizon signal)
    with torch.no_grad():
        for gi, graph in enumerate(solver.train_loader):
            if gi >= a.graphs:
                break
            graph = graph.to(dev)
            n = graph.num_nodes
            graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, torch.Size((n, n))).to_sparse_csr()
            x = torch.randint(0, 2, size=(n, Kc), device=dev, dtype=torch.float)
            states = []
            for t in range(T):                                    # one training rollout
                states.append(x.clone())
                _, grad = solver.problem.energy_func(graph, x, True)
                logits, _ = solver.flip_logits(graph, x, torch.ones_like(x)*t/T, t, ratio, grad)
                x = torch.where(torch.rand_like(x) < torch.sigmoid(logits), 1 - x, x)
            states.append(x)                                      # X_T (the state after the last step)
            _, g_end = solver.problem.energy_func(graph, states[T - 1], True)
            ts = np.linspace(0, T - 2, a.C).round().astype(int)
            for t in ts:
                k = int(rng.integers(Kc))
                xt, xn = states[t][:, k], states[t + 1][:, k]
                nodes = torch.as_tensor(rng.choice(n, size=min(a.m, n), replace=False), device=dev)
                m = len(nodes)
                flipped = (xn[nodes] != xt[nodes]).float()
                s = 1 - 2*flipped                                 # +1: not flipped at step t
                # columns: for node q and pair r, factual = 2*(q*R + r), toggled = factual + 1
                B = 2*m*a.R
                X0 = xn.unsqueeze(1).repeat(1, B)
                tog = torch.arange(m, device=dev).repeat_interleave(a.R)*2*a.R + 2*torch.arange(a.R, device=dev).repeat(m) + 1
                X0[nodes.repeat_interleave(a.R), tog] = 1 - X0[nodes.repeat_interleave(a.R), tog]
                E = {h: torch.empty(B, device=dev) for h in [-1] + hs}
                for c0 in range(0, B, a.chunk):
                    c1 = min(B, c0 + a.chunk)
                    u = torch.rand(T, n, (c1 - c0)//2, device=dev).repeat_interleave(2, dim=2)
                    o = prior_rollout(solver, graph, X0[:, c0:c1], t + 1, ratio, u, a.objective, hs)
                    for h in E:
                        E[h][c0:c1] = o.get(h, o[-1])                # fewer than h steps left: the end state

                def stats(Eh):
                    Eh = Eh.reshape(m, a.R, 2)
                    lab = s.unsqueeze(1)*(Eh[:, :, 0] - Eh[:, :, 1])   # m x R
                    sig = lab.mean(1)
                    return sig, ((lab - sig.unsqueeze(1))**2).sum(1)/(a.R - 1)
                sig, noi = stats(E[-1])
                for h in hs:
                    sh, nh = stats(E[h])
                    hrows[h] += [(int(t), float(sh[q]), float(nh[q]), float(sig[q])) for q in range(m)]
                cam = (g_end[nodes, k]*(2*xt[nodes] - 1))
                for q in range(m):
                    rows.append((int(t), int(xt[nodes[q]].item()), float(sig[q]), float(noi[q]), float(cam[q])))
            print(f'graph {gi} (n = {n}) done', flush=True)
    arr = np.array(rows)
    res = dict(checkpoint=ck, epoch=a.epoch, tau=tau, recenter=a.recenter, objective=a.objective, graphs=a.graphs, C=a.C, m=a.m, R=a.R,
               stages={})
    print('\nstage / nodes (count) | rms signal | rms noise (1 pair) | SNR for R = 1, 4, 8, 16 | corr for R = 1, 4, 8, 16 |'
          ' mean signal, frac > 0 | CAM: mean, frac > 0, corr with signal')
    for name, (lo, hi) in STAGES.items():
        for cls, cname in ((0, 'unselected'), (1, 'selected'), (None, 'all')):
            sel = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            if cls is not None:
                sel &= arr[:, 1] == cls
            if sel.sum() < 3:
                continue
            sig, noi, cam = arr[sel, 2], arr[sel, 3], arr[sel, 4]
            p_noi = float(noi.mean())
            p_sig = max(float((sig**2).mean()) - p_noi/a.R, 0.0)  # remove the noise left in the R_max mean
            snr = {R: p_sig/max(p_noi/R, 1e-12) for R in RS}
            corr = {R: float(np.sqrt(v/(1 + v))) for R, v in snr.items()}
            c_cam = float(np.corrcoef(sig, cam)[0, 1]) if sig.std() > 0 and cam.std() > 0 else float('nan')
            r = dict(count=int(sel.sum()), rms_signal=float(np.sqrt(p_sig)), rms_noise=float(np.sqrt(p_noi)),
                     snr={str(k): v for k, v in snr.items()}, corr={str(k): v for k, v in corr.items()},
                     mean_signal=float(sig.mean()), frac_signal_pos=float((sig > 0).mean()),
                     cam_mean=float(cam.mean()), cam_frac_pos=float((cam > 0).mean()), cam_corr_with_signal=c_cam)
            res['stages'][f'{name} / {cname}'] = r
            print(f'{name:15s} {cname:10s} ({r["count"]:4d}) | {r["rms_signal"]:6.2f} | {r["rms_noise"]:6.2f} | '
                  + ' '.join(f'{snr[R]:6.2f}' for R in RS) + ' | ' + ' '.join(f'{corr[R]:4.2f}' for R in RS)
                  + f' | {r["mean_signal"]:6.2f} {r["frac_signal_pos"]:4.2f} | {r["cam_mean"]:6.2f} '
                  f'{r["cam_frac_pos"]:4.2f} {c_cam:5.2f}', flush=True)
    if hs:
        res['horizons'] = {}
        print('\ntruncated horizon H, all nodes | rms signal | rms noise | corr for R = 1, 8 | '
              'corr of the signal with the full-rollout signal')
    for h in hs:
        ha = np.array(hrows[h])
        for name, (lo, hi) in STAGES.items():
            sel = (ha[:, 0] >= lo) & (ha[:, 0] < hi)
            sig, noi, full = ha[sel, 1], ha[sel, 2], ha[sel, 3]
            p_noi = float(noi.mean())
            p_sig = max(float((sig**2).mean()) - p_noi/a.R, 0.0)
            snr1 = p_sig/max(p_noi, 1e-12)
            cf = float(np.corrcoef(sig, full)[0, 1]) if sig.std() > 0 and full.std() > 0 else float('nan')
            r = dict(rms_signal=float(np.sqrt(p_sig)), rms_noise=float(np.sqrt(p_noi)),
                     corr1=float(np.sqrt(snr1/(1 + snr1))), corr8=float(np.sqrt(8*snr1/(1 + 8*snr1))),
                     corr_with_full=cf)
            res['horizons'][f'{h} / {name}'] = r
            print(f'H = {h:2d}  {name:15s} | {r["rms_signal"]:6.2f} | {r["rms_noise"]:6.2f} | {r["corr1"]:4.2f} '
                  f'{r["corr8"]:4.2f} | {cf:5.2f}', flush=True)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
