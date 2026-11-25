"""Stage 1A of docs/20261125/plan.md: the static SB coupling between Bernoulli(0.5)^n and the KaMIS multi-solution
label sets of real RB-small training graphs.

For each graph and each omega (strength of the Hamming cost, omega = log((1 + a_T) / (1 - a_T))):
  - Monte Carlo Sinkhorn on K uniform starts z (math_route1.md §7), float32 on the GPU;
  - label-marginal error of the solved g, measured on K_test fresh starts, with its own noise floor
    (2 standard errors of the fresh-sample estimate, relative to pi1);
  - effective number of labels per start, 1 / sum_m P(m | z)^2, averaged over the fresh starts;
  - mean Hamming cost E d(z, y) under the coupling, against n / 2 for the independent coupling;
  - Sinkhorn iterations and wall time.
The float32 solver is checked against utils/sb_coupling.sinkhorn (float64) on the first graphs.

Usage: python scripts/sb/coupling_analysis.py --out <dir> [--graphs 200] [--K 16384 65536 262144]
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils import sb_coupling as sb  # noqa: E402

LABELS = '/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/matched_kamis_loaderorder.pt'


def overlaps(z, labels):
    """Hamming distances d(z, s_m), (K, M) float32."""
    return z.sum(1, keepdim=True) + labels.sum(1)[None, :] - 2 * z @ labels.T


def distances(K, n, labels, gen, chunk=1 << 18):
    """Hamming distances of K fresh uniform starts to the labels, stored as int16 (d <= n < 32768), (K, M)."""
    out = torch.empty(K, labels.shape[0], dtype=torch.int16, device=labels.device)
    for a in range(0, K, chunk):
        b = min(K, a + chunk)
        z = torch.randint(0, 2, (b - a, n), device=labels.device, generator=gen).float()
        out[a:b] = overlaps(z, labels).round().to(torch.int16)
    return out


def solve(D, omega, pi1, iters=5000, tol=1e-4, chunk=1 << 18):
    """One-sided Sinkhorn on distances D (K, M) for cost omega * d, in chunks of starts; returns log g (float64 on the
    GPU), the max relative marginal error on the training starts and the iteration count."""
    K = D.shape[0]
    log_pi1 = torch.log(pi1)
    log_g = log_pi1.clone()
    for it in range(1, iters + 1):
        lg = log_g.float()[None, :]
        marg = torch.zeros_like(pi1)
        for a in range(0, K, chunk):
            marg += torch.softmax(lg - omega * D[a:a + chunk].float(), dim=1).sum(0).double()
        marg /= K
        rel = ((marg - pi1).abs() / pi1).max().item()
        if rel < tol:
            break
        log_g = log_g + log_pi1 - torch.log(marg)
        log_g = log_g - torch.logsumexp(log_g, 0)
    return log_g, rel, it


def evaluate(labels, log_g, omega, pi1, n, K_test, gen, chunk=1 << 18):
    """Fresh-start statistics of the coupling with potential log_g."""
    M = labels.shape[0]
    s1 = torch.zeros(M, dtype=torch.float64, device=labels.device)
    s2 = torch.zeros(M, dtype=torch.float64, device=labels.device)
    eff = 0.0; cost = 0.0; done = 0
    while done < K_test:
        k = min(chunk, K_test - done)
        z = torch.randint(0, 2, (k, n), device=labels.device, generator=gen).float()
        D = overlaps(z, labels)
        P = torch.softmax(log_g.float()[None, :] + (-omega * D), dim=1)
        s1 += P.sum(0).double(); s2 += (P.double() ** 2).sum(0)
        eff += (1.0 / (P.double() ** 2).sum(1)).sum().item()
        cost += (P * D).sum().item()
        done += k
    marg = s1 / K_test
    se = ((s2 / K_test - marg ** 2).clamp(min=0) / K_test).sqrt()
    return dict(rel_err=((marg - pi1).abs() / pi1).max().item(),
                noise_floor=(2 * se / pi1).max().item(),
                eff=eff / K_test, cost=cost / K_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--graphs', type=int, default=200)
    ap.add_argument('--omegas', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.35, 0.5, 1.0])
    ap.add_argument('--K', type=int, nargs='+', default=[1 << 16, 1 << 18, 1 << 20])
    ap.add_argument('--K_test', type=int, default=1 << 22)
    ap.add_argument('--tag', default='')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--check64', type=int, default=3, help='graphs on which to compare with the float64 solver')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda'
    data = torch.load(LABELS, weights_only=False)
    sets, files = data['sets'], data['files']
    order = torch.randperm(len(sets), generator=torch.Generator().manual_seed(args.seed))[:args.graphs].tolist()
    rows = []
    check = set(order[:args.check64])
    order = order[args.shard::args.nshards]
    out_path = os.path.join(args.out, f'coupling_stats{args.tag}_{args.shard}.jsonl')
    fout = open(out_path, 'w')
    for gi_pos, gi in enumerate(order):
        labels = sets[gi].to(dev).float()
        M, n = labels.shape
        pi1 = torch.full((M,), 1.0 / M, dtype=torch.float64, device=dev)
        for omega in args.omegas:
            aT = math.tanh(omega / 2)
            for K in args.K:
                gen = torch.Generator(dev).manual_seed(1000 * gi + K % 997)
                torch.cuda.synchronize(); t0 = time.time()
                D = distances(K, n, labels, gen)
                log_g, train_rel, it = solve(D, omega, pi1) if M > 1 else (torch.zeros(1, dtype=torch.float64, device=dev), 0.0, 0)
                torch.cuda.synchronize(); dt = time.time() - t0
                row = dict(graph=gi, file=files[gi], n=n, M=M, omega=omega, a_T=aT, K=K, iters=it,
                           train_rel=train_rel, time=dt)
                if gi in check and K == args.K[0] and K <= 1 << 18:   # same z stream as distances() (one chunk)
                    z = torch.randint(0, 2, (K, n), device=dev, generator=torch.Generator(dev).manual_seed(1000 * gi + K % 997)).float()
                    lg64, err64, _ = sb.sinkhorn(labels.double(), pi1, aT, z.double(), tol=1e-12)
                    row['diff_vs_float64'] = ((log_g - log_g.mean()) - (lg64 - lg64.mean())).abs().max().item()
                tgen = torch.Generator(dev).manual_seed(10 ** 6 + gi)
                row.update(evaluate(labels, log_g, omega, pi1, n, args.K_test, tgen))
                row['indep_cost'] = n / 2.0
                fout.write(json.dumps(row) + '\n'); fout.flush()
                rows.append(row)
        print(f'[{gi_pos + 1}/{len(order)}] graph {gi} n={n} M={M} done', flush=True)
    fout.close()
    print('wrote', out_path)


if __name__ == '__main__':
    main()
