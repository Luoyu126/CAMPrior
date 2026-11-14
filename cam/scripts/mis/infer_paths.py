"""Inference paths of RLSA, CAM and CAMPrior on ER-[700-800] MIS (local addition, see cam/UPSTREAM.md).

Question: from the same random start, how do the three samplers move to their final solutions?

All three are run with the inference loop of evaluate_single (num_t steps x num_k chains, the last step
soft, then the greedy decoding of problem.evaluate), from the same initial states and with the same
uniforms per step (common random numbers), so differences come from the flip probabilities only:
    RLSA      the CAMPrior prior alone (residual 0) at the epoch-0 scale (ratio 1), i.e. the RLSA
              sampler CAMPrior starts from (test 38.56 on ER-[700-800])
    CAM       a CAM checkpoint (network logits only)
    CAMPrior  a CAMPrior checkpoint (prior + network residual, ratio from its .prior.json)

Recorded per step t = 0 .. num_t-1 (state X_t before the step's flips) and chain, then averaged over
chains and graphs:
    size        |X_t|;  conflicts: edges with both ends in X_t
    decoded     size of the greedy decoding of X_t (as the final decoding does it): an anytime curve,
                as the primal-gap-vs-step curves of the RLSA paper (Feng & Yang 2025, Fig. 1)
    adds, removes   0->1 and 1->0 flips at step t (the learned step size; Neural SA, Correia et al.
                2023, Fig. 3c plots the same for its learned proposal)
    to_final    Hamming distance from X_t to the chain's output (its decoded final solution) / n
    frozen      fraction of nodes that do not flip again from step t on
    diversity   mean pairwise Hamming distance between the chains / n (sum Hamming diversity as in
                PQQA, Ichikawa & Iwashita)
    kl_prior    mean per-node KL(policy || RLSA prior) of the step's flip distribution (CAM, CAMPrior)
plus per graph: the best and mean decoded output, the number of distinct outputs over the chains and
the Jaccard overlap of the best outputs of each pair of methods. For the first --keep graphs the full
binary paths are saved (npz) for path plots.

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=9 $CAM_PYTHON cam/scripts/mis/infer_paths.py \
        --cam runs/cam/mis_er700/CAM_tau0.1_e100_s1/epoch_0090.pt \
        --camprior runs/cam/mis_er700/CAMPrior_p1_e100_s1/epoch_0090.pt --graphs 16
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CAM_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
REPO = os.path.abspath(os.path.join(CAM_DIR, '..'))
METHODS = ('RLSA', 'CAM', 'CAMPrior')
STEP_KEYS = ('size', 'conflicts', 'decoded_mean', 'decoded_best', 'adds', 'removes', 'to_final', 'frozen',
             'diversity', 'kl_prior')


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


def bernoulli_kl(p, q, eps=1e-6):
    p, q = p.clamp(eps, 1 - eps), q.clamp(eps, 1 - eps)
    return p*torch.log(p/q) + (1 - p)*torch.log((1 - p)/(1 - q))


@torch.no_grad()
def run_path(solver, graph, mode, ratio, x0, U):
    """One evaluate_single-style run of num_k chains. mode: RLSA / CAM / CAMPrior.
    Returns the binary states X_0..X_{T-1} (T x n x K bool), the decoded outputs (n x K), their sizes (K,)
    and the per-step KL to the prior (T,)."""
    T = solver.num_t
    x = x0.clone()
    states, kls = [], []
    for t in range(T):
        states.append(x.bool())
        k = int(t*solver.num_tp/T)
        time = torch.ones_like(x)*k/solver.num_tp
        prior = solver.prior_logits(graph, x, k, ratio)
        if mode == 'RLSA':
            logits = prior
        else:
            residual = solver.model(graph, x.unsqueeze(-1), time).squeeze(-1)
            logits = residual if mode == 'CAM' else solver.total_logits(graph, prior, residual)
        p = torch.sigmoid(logits)
        kls.append(float(bernoulli_kl(p, torch.sigmoid(prior)).mean()))
        if t == T - 1:
            x = x*(1 - p) + p*(1 - x)
        else:
            x = torch.where(U[t] < p, 1 - x, x)
    obj, sol = solver.problem.evaluate(graph, x)
    return torch.stack(states), sol.bool(), (-obj).float(), np.array(kls)


@torch.no_grad()
def path_stats(solver, graph, S, D, kls):
    """Per-step statistics (dict of (T,) arrays) of states S (T x n x K bool) with outputs D (n x K)."""
    T, n, K = S.shape
    src, dst = graph.edge_index
    Sf = S.float()
    out = {}
    out['size'] = Sf.sum(1).mean(1)
    out['conflicts'] = (Sf[:, src, :]*Sf[:, dst, :]).sum(1).mean(1)/2
    dec = torch.stack([(-solver.problem.evaluate(graph, Sf[t])[0]).float() for t in range(T)])   # T x K
    out['decoded_mean'], out['decoded_best'] = dec.mean(1), dec.max(1).values
    flips = S[1:] ^ S[:-1]                                                     # T-1 x n x K
    adds = (flips & S[1:]).float().sum(1).mean(1)
    rem = (flips & ~S[1:]).float().sum(1).mean(1)
    z = torch.zeros(1, device=S.device)
    out['adds'], out['removes'] = torch.cat([adds, z]), torch.cat([rem, z])   # step t: X_t -> X_{t+1}
    out['to_final'] = (S ^ D.unsqueeze(0)).float().sum(1).mean(1)/n
    # frozen[t]: no flip at steps t .. T-2 (X_t = X_{t+1} = ... = X_{T-1})
    later = torch.flip(torch.cumsum(torch.flip(flips.float(), [0]), 0), [0])   # flips from step t on
    later = torch.cat([later, torch.zeros_like(later[:1])])
    out['frozen'] = (later == 0).float().mean((1, 2))
    Kf = Sf.permute(0, 2, 1)                                                   # T x K x n
    ham = Kf.sum(2, keepdim=True) + Kf.sum(2).unsqueeze(1) - 2*Kf@Kf.transpose(1, 2)
    out['diversity'] = ham.sum((1, 2))/(K*(K - 1))/n
    out['kl_prior'] = torch.as_tensor(kls, device=S.device)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def jaccard(a, b):
    a, b = a.bool(), b.bool()
    return float((a & b).sum())/max(1, float((a | b).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cam', required=True, help='CAM checkpoint')
    ap.add_argument('--camprior', required=True, help='CAMPrior checkpoint (with its .prior.json)')
    ap.add_argument('--graphs', type=int, default=16, help='first G graphs of the test set')
    ap.add_argument('--keep', type=int, default=2, help='save the full paths of the first graphs')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(REPO, 'runs', 'diag', 'infer_paths'))
    a = ap.parse_args()
    cam_ck, cp_ck, out = (os.path.abspath(p) for p in (a.cam, a.camprior, a.out))
    meta = json.load(open(cp_ck + '.prior.json'))
    sys.path.insert(0, CAM_DIR)
    os.chdir(CAM_DIR)
    sys.argv = ['main.py']
    from solvers.cam_prior import CAMPrior
    data = os.environ.get('CAM_ER700_DIR', '/home/shengyuf/yunyi/data/CAM_data/data_er')
    argv = ['--train_path', f'{data}/er_train', '--valid_path', f'{data}/er_test', '--test_path', f'{data}/er_test',
            '--problem', 'mis', '--valid_samples', str(a.graphs), '--samples_per_epoch', '1000',
            '--batch_size', '1', '--mini_batch_size', '512', '--num_t', '50', '--num_k', '20', '--num_tp', '50',
            '--num_kp', '20', '--num_h', '64', '--num_l', '8', '--method', 'CAMPrior',
            '--model', 'EncodeProcessDecode', '--epochs', '100', '--N_anneal', '100', '--lr', '0.0005',
            '--prior_tau0', str(meta['prior_tau0']), '--prior_d', str(meta['prior_d']), '--seed', '0',
            '--save_dir', os.path.join(out, 'save'), '--do_train', '--wandb_name', 'diag',
            '--wandb_run_name', 'infer_paths']
    args = main_parser().parse_args(argv)
    solver = CAMPrior(args)
    dev = solver.accelerator.device
    solver.model.to(dev)
    weights = {'CAM': torch.load(cam_ck, map_location=dev), 'CAMPrior': torch.load(cp_ck, map_location=dev)}
    ratios = {'RLSA': 1.0, 'CAM': 1.0, 'CAMPrior': float(meta['tau_ratio'])}
    print(f'CAM {cam_ck}\nCAMPrior {cp_ck} (prior ratio {ratios["CAMPrior"]})', flush=True)
    T, K = solver.num_t, solver.num_k
    acc = {m: {k: [] for k in STEP_KEYS} for m in METHODS}
    per_graph = []
    os.makedirs(out, exist_ok=True)
    for gi, graph in enumerate(solver.valid_loader):
        if gi >= a.graphs:
            break
        graph = graph.to(dev)
        n = graph.num_nodes
        graph.A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight, torch.Size((n, n))).to_sparse_csr()
        g = torch.Generator(device=dev)
        g.manual_seed(a.seed*100003 + gi)
        x0 = torch.randint(0, 2, (n, K), generator=g, device=dev).float()
        U = torch.rand(T, n, K, generator=g, device=dev)
        rec, keep = {'graph': gi, 'n': n}, {}
        best = {}
        for m in METHODS:
            if m in weights:
                solver.model.load_state_dict(weights[m])
            solver.model.eval()
            S, D, sizes, kls = run_path(solver, graph, m, ratios[m], x0, U)
            st = path_stats(solver, graph, S, D, kls)
            for k in STEP_KEYS:
                acc[m][k].append(st[k])
            b = int(torch.argmax(sizes))
            best[m] = D[:, b]
            rec[m] = dict(best=float(sizes.max()), mean=float(sizes.mean()),
                          distinct=len({tuple(torch.nonzero(D[:, c]).reshape(-1).tolist()) for c in range(K)}))
            if gi < a.keep:
                keep[f'{m}_states'] = S.cpu().numpy()
                keep[f'{m}_out'] = D.cpu().numpy()
        for i, m1 in enumerate(METHODS):
            for m2 in METHODS[i + 1:]:
                rec[f'jaccard_{m1}_{m2}'] = jaccard(best[m1], best[m2])
        per_graph.append(rec)
        if keep:
            np.savez_compressed(os.path.join(out, f'paths_g{gi}.npz'), edge_index=graph.edge_index.cpu().numpy(),
                                **keep)
        print(f'graph {gi} (n={n}): ' + ', '.join(f'{m} best {rec[m]["best"]:.0f} mean {rec[m]["mean"]:.2f} '
                                                  f'distinct {rec[m]["distinct"]}' for m in METHODS), flush=True)
    res = {'cam': cam_ck, 'camprior': cp_ck, 'ratios': ratios, 'graphs': len(per_graph), 'num_t': T, 'num_k': K,
           'per_graph': per_graph,
           'steps': {m: {k: np.mean(acc[m][k], 0).tolist() for k in STEP_KEYS} for m in METHODS}}
    json.dump(res, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print('\nmean over graphs:')
    for m in METHODS:
        print(f'  {m:8s} best {np.mean([r[m]["best"] for r in per_graph]):.2f}  '
              f'mean {np.mean([r[m]["mean"] for r in per_graph]):.2f}  '
              f'distinct {np.mean([r[m]["distinct"] for r in per_graph]):.1f}')
    for k in [k for k in per_graph[0] if k.startswith('jaccard')]:
        print(f'  {k}: {np.mean([r[k] for r in per_graph]):.3f}')
    print('\n   t | ' + ' | '.join(f'{m}: dec  size confl add rem to_fin frozen div KL' for m in METHODS))
    for t in (0, 1, 2, 5, 10, 20, 30, 40, 45, 49):
        cells = []
        for m in METHODS:
            s = res['steps'][m]
            cells.append(f'{s["decoded_mean"][t]:5.1f} {s["size"][t]:5.1f} {s["conflicts"][t]:6.1f} '
                         f'{s["adds"][t]:4.1f} {s["removes"][t]:4.1f} {s["to_final"][t]:.3f} {s["frozen"][t]:.2f} '
                         f'{s["diversity"][t]:.3f} {s["kl_prior"][t]:.3f}')
        print(f'  {t:2d} | ' + ' | '.join(cells), flush=True)


if __name__ == '__main__':
    main()
