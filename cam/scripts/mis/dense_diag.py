"""Scale and noise of the lambda-weighted labels of --label_mode dense (local addition, see cam/UPSTREAM.md).

Question: if dense labels with lambda = 0.9 / 0.97 train better than CAM's label (lambda = 1), is it
because their noise is smaller (easier to fit) or because their scale is smaller (a larger effective
temperature, since the per-node minimiser of the loss is logit = ref + label / tau)?

On a CAMPrior checkpoint, for C stored states X_t (step t, chain k) per graph: copy X_t into R chains,
run the policy (prior + network) from step t to T-1 with independent randomness, and form for each
lambda the label of _generate_data_labels,
    R_{T-1} = G_{T-1},  R_s = (1 - lam) G_{s+1} + lam R_{s+1},   label = R_t * (2 X_t - 1),
with G_j = grad H(X_j) (lambda = 1 is CAM's single-clip label). Since the loss is linear in the
label, the network regresses to E[label | X_t]; per (state, node):
    signal = E[label | X_t]  (mean over the R chains),   noise = label - signal.
Reported per lambda and stage of t: RMS of a single label, RMS of the signal, RMS of the noise,
signal-to-noise ratio (signal power / noise power per sample), the signal's correlation with
lambda = 1's signal, and RMS(signal) / tau (the size of the logit offset it asks for).

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=7 $CAM_PYTHON cam/scripts/mis/dense_diag.py \
        --checkpoint runs/cam/mis_er700/CAMPrior_p1_e100_s1/epoch_0049.pt --epoch 50 --out runs/diag/dense/s1_ep50.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CAM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
LAMS = (0.0, 0.5, 0.9, 0.97, 1.0)
STAGES = {'early (t < 17)': (0, 17), 'mid (17-33)': (17, 34), 'late (t >= 34)': (34, 50), 'all': (0, 50)}


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
def run(solver, graph, x, t0, ratio):
    """Run the policy from state x (n x B) at step t0 to the end; returns the states X_{t0..T-1}
    (list, the first is x) and their energy gradients."""
    T = solver.num_tp
    states, grads = [], []
    for t in range(t0, T):
        states.append(x.clone())
        _, grad = solver.problem.energy_func(graph, x, True)
        grads.append(grad)
        logits, _ = solver.flip_logits(graph, x, torch.ones_like(x)*t/T, t, ratio,
                                       grad if solver.prior_problem is solver.problem else None)
        x = torch.where(torch.rand_like(x) < torch.sigmoid(logits), 1 - x, x)
    return states, grads


def dense_label(grads, x0, lam):
    """Label of the first state X_t of the run, as in CAMPrior._generate_data_labels: grads[j] is at
    X_{t+j} (grads[0] unused); R_{T-1} = G_{T-1}, then R_s = (1 - lam) G_{s+1} + lam R_{s+1} for
    s = T-2 .. t; returns R_t * (2 x0 - 1)."""
    R = grads[-1]
    for g in reversed(grads[1:]):
        R = (1 - lam)*g + lam*R
    return R*(2*x0 - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--epoch', type=int, required=True, help='training epoch whose tau_train is used')
    ap.add_argument('--graphs', type=int, default=4)
    ap.add_argument('--C', type=int, default=24, help='stored states per graph (steps spread over 0..T-2)')
    ap.add_argument('--R', type=int, default=32, help='independent continuations per stored state')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ck = os.path.abspath(a.checkpoint)
    out = os.path.abspath(a.out) if a.out else None
    sys.path.insert(0, CAM_DIR)
    os.chdir(CAM_DIR)
    sys.argv = ['main.py']
    from solvers.cam_prior import CAMPrior
    data = os.environ.get('CAM_ER700_DIR', '/home/shengyuf/yunyi/data/CAM_data/data_er')
    save = os.path.join(CAM_DIR, '..', 'runs', 'diag', 'dense_diag')
    argv = ['--train_path', f'{data}/er_train', '--valid_path', f'{data}/er_valid', '--test_path', f'{data}/er_test',
            '--problem', 'mis', '--samples_per_epoch', '1000', '--valid_sample', '4', '--batch_size', '1',
            '--mini_batch_size', '512', '--num_t', '50', '--num_k', '20', '--num_tp', '50', '--num_kp', '20',
            '--num_h', '64', '--num_l', '8', '--method', 'CAMPrior', '--model', 'EncodeProcessDecode',
            '--epochs', '100', '--N_anneal', '100', '--lr', '0.0005', '--prior_tau0', '1', '--seed', '1',
            '--save_dir', save, '--do_train', '--checkpoint', ck, '--label_mode', 'dense',
            '--wandb_name', 'diag', '--wandb_run_name', 'dense_diag']
    args = main_parser().parse_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    solver = CAMPrior(args)
    solver.model.to(solver.accelerator.device).eval()
    dev = solver.accelerator.device
    solver.tau = solver.temp_scheduler(a.epoch)
    ratio = solver.prior_tau_ratio()
    T, Kc = solver.num_tp, solver.num_kp
    tau = float(solver.tau)
    print(f'checkpoint {ck}\nepoch {a.epoch}: tau_train {tau:.4f}, prior ratio {ratio:.3f}', flush=True)
    rng = np.random.default_rng(a.seed)
    # per lambda: lists of (t, signal n-vector, noise R x n)
    rec = {lam: [] for lam in LAMS}
    for gi, graph in enumerate(solver.train_loader):
        if gi >= a.graphs:
            break
        graph = graph.to(dev)
        n = graph.num_nodes
        graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, torch.Size((n, n))).to_sparse_csr()
        x0 = torch.randint(0, 2, size=(n, Kc), device=dev, dtype=torch.float)
        states, _ = run(solver, graph, x0, 0, ratio)                     # one training rollout
        ts = np.linspace(0, T - 2, a.C).round().astype(int)
        for t in ts:
            k = int(rng.integers(Kc))
            x = states[t][:, k:k+1].repeat(1, a.R)
            _, grads = run(solver, graph, x, int(t), ratio)
            for lam in LAMS:
                lab = dense_label(grads, x, lam).T                        # R x n
                sig = lab.mean(0)
                rec[lam].append((int(t), sig.cpu(), (lab - sig).cpu()))
        print(f'graph {gi} (n = {n}) done', flush=True)
    res = dict(checkpoint=ck, epoch=a.epoch, tau=tau, graphs=a.graphs, C=a.C, R=a.R, stages={})
    for name, (lo, hi) in STAGES.items():
        rows = {}
        ref = None
        for lam in reversed(LAMS):                                         # lambda = 1 first
            items = [(s, z) for t, s, z in rec[lam] if lo <= t < hi]
            sig = torch.cat([s for s, _ in items])
            noi = torch.cat([z.reshape(-1) for _, z in items])
            # the mean of R samples keeps noise/R of the noise power: remove it from the signal power
            p_sig = max(float(sig.pow(2).mean()) - float(noi.pow(2).mean())/a.R, 0.0)
            p_noi = float(noi.pow(2).mean())*a.R/(a.R - 1)
            if ref is None:
                ref = sig
            c = float(torch.corrcoef(torch.stack([sig, ref]))[0, 1])
            rows[str(lam)] = dict(rms_single=float(np.sqrt(p_sig + p_noi)), rms_signal=float(np.sqrt(p_sig)),
                                  rms_noise=float(np.sqrt(p_noi)), snr=p_sig/max(p_noi, 1e-12),
                                  corr_signal_with_lam1=c, rms_signal_over_tau=float(np.sqrt(p_sig))/tau)
        res['stages'][name] = rows
        print(f'\n{name}: lambda | rms single | rms signal | rms noise | SNR | corr with lambda=1 | signal/tau')
        for lam in LAMS:
            r = rows[str(lam)]
            print(f'  {lam:4} | {r["rms_single"]:7.3f} | {r["rms_signal"]:7.3f} | {r["rms_noise"]:7.3f} | '
                  f'{r["snr"]:6.3f} | {r["corr_signal_with_lam1"]:6.3f} | {r["rms_signal_over_tau"]:8.1f}', flush=True)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
