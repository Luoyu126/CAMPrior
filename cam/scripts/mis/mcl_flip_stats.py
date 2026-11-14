"""What a CAMPrior sampler does on maximum clique, step by step (local addition, see cam/UPSTREAM.md).

Question (docs/status.md 2.11): training makes CAMPrior worse than its RLSA starting point on maximum
clique. Is it because the network stops the sampler from adding nodes (CAM's endpoint label says
"don't add" strongly), or something else?

On a checkpoint, run the training rollout (num_tp steps, num_kp chains, the epoch's temperature) on
validation graphs and record per step t, averaged over chains and graphs:
  size      |S|, the number of selected nodes;   conflicts: selected pairs that are not adjacent
  adds / removes                     actual 0->1 and 1->0 flips
  E[adds], E[removes]                expected under the sampler (sum of p over unselected / selected)
  prior E[adds], prior E[removes]    the same under the RLSA prior alone (residual 0)
  resid(0), resid(1)                 mean of (total logit - prior logit) over unselected / selected
  label(0), label(1)                 mean CAM label grad_H(X_{T-1}) * (2 x_t - 1) over unselected /
                                     selected nodes of X_t; the per-node minimiser of the loss is
                                     logit = ref + label / tau, so a negative label lowers the flip
                                     probability (for an unselected node: "don't add")
plus the decoded clique size at the end (the validation metric, greedy decoding of the last state).

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=7 $CAM_PYTHON cam/scripts/mis/mcl_flip_stats.py \
        --checkpoint runs/cam/mcl_rb_small_pilot/<exp>/epoch_0000.pt --epoch 1 [--recenter on]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CAM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SHOW = (0, 5, 10, 20, 30, 40, 49)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--epoch', type=int, required=True, help='epoch whose tau_train is used (0 = before training)')
    ap.add_argument('--recenter', default='off', choices=['on', 'off'])
    ap.add_argument('--prior_tau0', default='1')
    ap.add_argument('--prior_d', default='10')
    ap.add_argument('--graphs', type=int, default=16)
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
    save = os.path.join(CAM_DIR, '..', 'runs', 'diag', 'mcl_flip_stats')
    argv = ['--train_path', f'{data}/RB_small_train', '--valid_path', f'{data}/RB_small_valid',
            '--test_path', f'{data}/RB_small_test', '--problem', 'mcl', '--valid_sample', '64',
            '--samples_per_epoch', '4000', '--batch_size', '1', '--mini_batch_size', '256',
            '--num_t', '50', '--num_k', '20', '--num_tp', '50', '--num_kp', '10', '--num_h', '64',
            '--num_l', '8', '--method', 'CAMPrior', '--model', 'EncodeProcessDecode', '--epochs', '100',
            '--N_anneal', '100', '--lr', '0.0005', '--prior_tau0', a.prior_tau0, '--prior_d', a.prior_d,
            '--prior_recenter', a.recenter, '--seed', '0', '--save_dir', save, '--do_train',
            '--checkpoint', ck, '--wandb_name', 'diag', '--wandb_run_name', 'mcl_flip_stats']
    args = main_parser().parse_args(argv)
    torch.manual_seed(a.seed)
    solver = CAMPrior(args)
    solver.model.to(solver.accelerator.device).eval()
    dev = solver.accelerator.device
    solver.tau = solver.temp_scheduler(a.epoch)
    ratio = solver.prior_tau_ratio()
    T, Kc, beta, tau = solver.num_tp, solver.num_kp, solver.problem.beta, float(solver.tau)
    print(f'checkpoint {ck}\nepoch {a.epoch}: tau_train {tau:.4f}, prior ratio {ratio:.3f}, recenter {a.recenter}', flush=True)
    keys = ('size', 'conflicts', 'adds', 'removes', 'E_adds', 'E_removes', 'prior_E_adds', 'prior_E_removes',
            'resid0', 'resid1', 'label0', 'label1')
    acc = {k: np.zeros(T) for k in keys}
    decoded = []
    with torch.no_grad():
        for gi, graph in enumerate(solver.valid_loader):
            if gi >= a.graphs:
                break
            graph = graph.to(dev)
            n = graph.num_nodes
            graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, torch.Size((n, n))).to_sparse_csr()
            x = torch.randint(0, 2, size=(n, Kc), device=dev, dtype=torch.float)
            states, rec = [], {k: [] for k in keys}
            for t in range(T):
                states.append(x.clone())
                energy, grad = solver.problem.energy_func(graph, x, True)
                logits, prior = solver.flip_logits(graph, x, torch.ones_like(x)*t/T, t, ratio, grad)
                p, pp = torch.sigmoid(logits), torch.sigmoid(prior)
                sel, un = x, 1 - x
                size = sel.sum(0)
                rec['size'].append(size.mean())
                rec['conflicts'].append(((energy.reshape(-1) + size)/beta).mean())
                rec['E_adds'].append((p*un).sum(0).mean())
                rec['E_removes'].append((p*sel).sum(0).mean())
                rec['prior_E_adds'].append((pp*un).sum(0).mean())
                rec['prior_E_removes'].append((pp*sel).sum(0).mean())
                r = logits - prior
                rec['resid0'].append((r*un).sum()/un.sum().clamp_min(1))
                rec['resid1'].append((r*sel).sum()/sel.sum().clamp_min(1))
                x_new = torch.where(torch.rand_like(x) < p, 1 - x, x)
                rec['adds'].append(((x_new - x) > 0).float().sum(0).mean())
                rec['removes'].append(((x_new - x) < 0).float().sum(0).mean())
                x = x_new
            _, g_end = solver.problem.energy_func(graph, states[-1], True)  # where CAM's label is taken
            for t in range(T):
                lab = g_end*(2*states[t] - 1)
                un = 1 - states[t]
                rec['label0'].append((lab*un).sum()/un.sum().clamp_min(1))
                rec['label1'].append((lab*states[t]).sum()/states[t].sum().clamp_min(1))
            for k in keys:
                acc[k] += torch.stack([torch.as_tensor(v, dtype=torch.float) for v in rec[k]]).cpu().numpy()
            obj, _ = solver.problem.evaluate(graph, x)
            decoded.append(float(-obj.float().mean()) if torch.is_tensor(obj) else float(-np.mean(obj)))
    G = min(a.graphs, len(solver.valid_loader))
    res = {k: (acc[k]/G).tolist() for k in keys}
    res.update(checkpoint=ck, epoch=a.epoch, tau=tau, recenter=a.recenter, graphs=G,
               decoded_clique_mean=float(np.mean(decoded)))
    print(f'decoded clique size (mean over {G} graphs x {Kc} chains, last state): {res["decoded_clique_mean"]:.2f}')
    print('   t |  size  confl | adds  rem | E[add] E[rem] | prior E[add] E[rem] | resid0 resid1 | label0 label1')
    for t in SHOW:
        g = lambda k: res[k][t]
        print(f'  {t:2d} | {g("size"):5.1f} {g("conflicts"):6.1f} | {g("adds"):4.1f} {g("removes"):4.1f} | '
              f'{g("E_adds"):6.2f} {g("E_removes"):6.2f} | {g("prior_E_adds"):6.2f} {g("prior_E_removes"):6.2f} | '
              f'{g("resid0"):6.2f} {g("resid1"):6.2f} | {g("label0"):6.2f} {g("label1"):6.2f}', flush=True)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
