"""Block moves for maximum clique: how large and how noisy is their simulated gain, compared with
single flips (local addition, see cam/UPSTREAM.md).

Question (docs/status.md 2.11): the simulated gain of one flip is small (0.15-0.5 clique nodes)
next to the rollout noise (about 2), because RLSA repairs most single changes. Does a combined
move, "add a seed node, drop the members it conflicts with, then greedily add common neighbours",
give a gain that stands out of the noise, and does screening the seed with a few rollouts find
moves that are really better?

Block move B(x, i) for a state x (0/1) and an unselected seed i with at most --max_conf conflicts
with the selected set (a random seed on RB graphs conflicts with most of the clique, and its block
move mostly destroys it):
    1. x_i = 1;  2. x_j = 0 for every selected j not adjacent to i (its conflicts in the clique
    energy);  3. repeat: candidates = unselected nodes without a conflict with any selected node;
    add the candidate adjacent to the most other candidates (ties: lowest index); until none.

On a checkpoint, run one training rollout (current sampler, the epoch's temperature) per graph,
pick C stored steps t, take the next state X_{t+1} of a random chain as the base, and compare the
moves (each applied to X_{t+1}, then RLSA prior rollouts over steps t+1 .. T-1, as the simulated
labels of solvers/sim_labels.py):
    rand_single       toggle of a random node (--n_rand of them)
    best_single       the single toggle with the best screened gain
    rand_block        block move from a random unselected seed (--n_rand of them)
    block_best_single block move from the seed of best_single
    best_block        the block move with the best screened gain
Screening: --screen candidate nodes (random, unselected for blocks) x --screen_R rollout pairs.
The chosen moves are then evaluated with --R fresh rollout pairs (screening noise does not enter).
Gain of a pair = size(decoded end state after the move) - size(decoded end state without it), the
two rollouts sharing their uniforms (objective = the validation objective, -size of the greedy
decoding). Reported per stage of t and move kind: mean gain, fraction of moves with mean gain > 0,
RMS noise of one pair, the SNR of one pair (gain^2 / noise power) and of R = 8 pairs, the number of
flipped nodes, and the immediate gain (decoded size right after the move, no rollout).

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=7 $CAM_PYTHON cam/scripts/mis/mcl_block_moves.py \
        --checkpoint runs/cam/mcl_rb_small_pilot/<exp>/epoch_0002.pt --epoch 3 --recenter on
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CAM_DIR = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
from sim_noise import STAGES, main_parser, prior_rollout  # noqa: E402

KINDS = ('rand_single', 'best_single', 'rand_block', 'block_best_single', 'best_block')


def conflict_matrix(graph, n, dev):
    """conf[i, j] = True if i != j are not adjacent in G (they cannot be in one clique)."""
    adj = torch.zeros(n, n, dtype=torch.bool, device=dev)
    adj[graph.edge_index[0], graph.edge_index[1]] = True
    return ~adj & ~torch.eye(n, dtype=torch.bool, device=dev)


def block_moves(conf, x, seeds):
    """Block moves of state x (n,) from each seed (S,); returns n x S (0/1 float)."""
    S = len(seeds)
    cols = torch.arange(S, device=x.device)
    X = x.unsqueeze(1).repeat(1, S).bool()
    X[seeds, cols] = True
    X &= ~conf[seeds].T                                          # drop the seed's conflicts
    X[seeds, cols] = True
    free = (~conf).float()
    while True:
        blocked = (conf.float() @ X.float()) > 0                 # conflicts with a selected node
        cand = ~X & ~blocked
        if not cand.any():
            break
        score = (free @ cand.float())*cand                       # adjacent candidates (incl. itself)
        score = torch.where(cand, score, -torch.ones_like(score))
        best = torch.argmax(score, 0)
        live = cand.any(0)
        X[best[live], cols[live]] = True
    return X.float()


def toggles(x, nodes):
    X = x.unsqueeze(1).repeat(1, len(nodes))
    X[nodes, torch.arange(len(nodes), device=x.device)] = 1 - X[nodes, torch.arange(len(nodes), device=x.device)]
    return X


@torch.no_grad()
def gains(solver, graph, x, moves, t, ratio, R, chunk=8192):
    """size(moved) - size(factual) for R rollout pairs per move: (M, R)."""
    n, M = moves.shape
    B = 2*M*R
    X0 = torch.empty(n, B, device=x.device)
    X0[:, 0::2] = x.unsqueeze(1)
    X0[:, 1::2] = moves.repeat_interleave(R, dim=1)
    g = torch.empty(B, device=x.device)
    T = solver.num_tp
    for c0 in range(0, B, chunk):
        c1 = min(B, c0 + chunk)
        u = torch.rand(T, n, (c1 - c0)//2, device=x.device).repeat_interleave(2, dim=2)
        g[c0:c1] = prior_rollout(solver, graph, X0[:, c0:c1], t + 1, ratio, u, 'decoded')[-1]
    g = g.reshape(M, R, 2)
    return g[:, :, 0] - g[:, :, 1]                              # g = -size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--recenter', default='off', choices=['on', 'off'])
    ap.add_argument('--prior_tau0', default='1')
    ap.add_argument('--prior_d', default='10')
    ap.add_argument('--graphs', type=int, default=8)
    ap.add_argument('--C', type=int, default=24)
    ap.add_argument('--screen', type=int, default=64, help='candidate nodes / seeds screened per state')
    ap.add_argument('--screen_R', type=int, default=8, help='rollout pairs per screened candidate')
    ap.add_argument('--n_rand', type=int, default=4, help='random moves of each random kind per state')
    ap.add_argument('--max_conf', type=int, default=2,
                    help='block seeds: unselected nodes with at most this many conflicts with the selected set')
    ap.add_argument('--R', type=int, default=32, help='fresh rollout pairs per evaluated move')
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
    save = os.path.join(CAM_DIR, '..', 'runs', 'diag', 'mcl_block_moves')
    argv = ['--train_path', f'{data}/RB_small_train', '--valid_path', f'{data}/RB_small_valid',
            '--test_path', f'{data}/RB_small_test', '--problem', 'mcl', '--valid_sample', '4',
            '--samples_per_epoch', '4000', '--batch_size', '1', '--mini_batch_size', '256',
            '--num_t', '50', '--num_k', '20', '--num_tp', '50', '--num_kp', '10', '--num_h', '64',
            '--num_l', '8', '--method', 'CAMPrior', '--model', 'EncodeProcessDecode', '--epochs', '100',
            '--N_anneal', '100', '--lr', '0.0005', '--prior_tau0', a.prior_tau0, '--prior_d', a.prior_d,
            '--prior_recenter', a.recenter, '--seed', '0', '--save_dir', save, '--do_train',
            '--checkpoint', ck, '--wandb_name', 'diag', '--wandb_run_name', 'mcl_block_moves']
    args = main_parser().parse_args(argv)
    torch.manual_seed(a.seed)
    solver = CAMPrior(args)
    solver.model.to(solver.accelerator.device).eval()
    dev = solver.accelerator.device
    solver.tau = solver.temp_scheduler(a.epoch)
    ratio = solver.prior_tau_ratio()
    T, Kc = solver.num_tp, solver.num_kp
    print(f'checkpoint {ck}\nepoch {a.epoch}, recenter {a.recenter}, block seeds with <= {a.max_conf} conflicts: '
          f'{a.graphs} graphs x {a.C} states; screen '
          f'{a.screen} x {a.screen_R} pairs; evaluate with {a.R} fresh pairs', flush=True)
    rng = np.random.default_rng(a.seed)
    rows = []   # (t, kind, mean gain, noise power of one pair, flips, immediate gain)
    with torch.no_grad():
        for gi, graph in enumerate(solver.train_loader):
            if gi >= a.graphs:
                break
            graph = graph.to(dev)
            n = graph.num_nodes
            graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, torch.Size((n, n))).to_sparse_csr()
            conf = conflict_matrix(graph, n, dev)
            x = torch.randint(0, 2, size=(n, Kc), device=dev, dtype=torch.float)
            states = []
            for t in range(T):
                states.append(x.clone())
                _, grad = solver.problem.energy_func(graph, x, True)
                logits, _ = solver.flip_logits(graph, x, torch.ones_like(x)*t/T, t, ratio, grad)
                x = torch.where(torch.rand_like(x) < torch.sigmoid(logits), 1 - x, x)
            states.append(x)
            size = lambda X: -solver.problem.evaluate(graph, X)[0].float()
            for t in np.linspace(0, T - 2, a.C).round().astype(int):
                xb = states[t + 1][:, int(rng.integers(Kc))]
                nconf = conf.float() @ xb
                un = torch.nonzero((xb == 0) & (nconf <= a.max_conf)).reshape(-1)   # nearly compatible seeds
                if len(un) == 0:
                    continue
                base_size = size(xb.unsqueeze(1))[0]
                # screening
                cand = torch.as_tensor(rng.choice(n, size=min(a.screen, n), replace=False), device=dev)
                gs = gains(solver, graph, xb, toggles(xb, cand), t, ratio, a.screen_R).mean(1)
                best_single = cand[int(torch.argmax(gs))]
                seeds = un[torch.as_tensor(rng.choice(len(un), size=min(a.screen, len(un)), replace=False), device=dev)]
                Bm = block_moves(conf, xb, seeds)
                gb = gains(solver, graph, xb, Bm, t, ratio, a.screen_R).mean(1)
                moves = {
                    'rand_single': toggles(xb, torch.as_tensor(rng.choice(n, size=a.n_rand, replace=False), device=dev)),
                    'best_single': toggles(xb, best_single.reshape(1)),
                    'rand_block': block_moves(conf, xb, un[torch.as_tensor(
                        rng.choice(len(un), size=min(a.n_rand, len(un)), replace=False), device=dev)]),
                    'block_best_single': (block_moves(conf, xb, best_single.reshape(1)) if xb[best_single] == 0
                                          else toggles(xb, best_single.reshape(1))),   # a removal: no block
                    'best_block': Bm[:, int(torch.argmax(gb))].unsqueeze(1),
                }
                for kind, M in moves.items():
                    G = gains(solver, graph, xb, M, t, ratio, a.R)              # fresh pairs
                    mean = G.mean(1)
                    noise = ((G - mean.unsqueeze(1))**2).sum(1)/(a.R - 1)
                    flips = (M != xb.unsqueeze(1)).float().sum(0)
                    imm = size(M) - base_size
                    for q in range(M.shape[1]):
                        rows.append((int(t), kind, float(mean[q]), float(noise[q]), float(flips[q]), float(imm[q])))
            print(f'graph {gi} (n = {n}) done', flush=True)
    res = dict(checkpoint=ck, epoch=a.epoch, recenter=a.recenter, max_conf=a.max_conf, graphs=a.graphs, C=a.C, screen=a.screen,
               screen_R=a.screen_R, R=a.R, stages={})
    print('\nstage / move (count) | mean gain | frac > 0 | rms noise (1 pair) | SNR 1 pair, 8 pairs | flips | immediate gain')
    for name, (lo, hi) in STAGES.items():
        for kind in KINDS:
            sel = [r for r in rows if lo <= r[0] < hi and r[1] == kind]
            if not sel:
                continue
            g = np.array([r[2] for r in sel])
            p_noi = float(np.mean([r[3] for r in sel]))
            p_sig = max(float((g**2).mean()) - p_noi/a.R, 0.0)
            snr1 = p_sig/max(p_noi, 1e-12)
            r = dict(count=len(sel), mean_gain=float(g.mean()), frac_pos=float((g > 0).mean()),
                     rms_noise=float(np.sqrt(p_noi)), snr1=snr1, snr8=8*snr1,
                     flips=float(np.mean([r[4] for r in sel])), immediate=float(np.mean([r[5] for r in sel])))
            res['stages'][f'{name} / {kind}'] = r
            print(f'{name:15s} {kind:17s} ({r["count"]:3d}) | {r["mean_gain"]:6.2f} | {r["frac_pos"]:4.2f} | '
                  f'{r["rms_noise"]:5.2f} | {snr1:6.2f} {8*snr1:6.2f} | {r["flips"]:5.1f} | {r["immediate"]:6.2f}', flush=True)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
