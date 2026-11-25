"""How much does the start z determine the sampled solution? (plan.md 1C, "起点决定解的程度".)

Input: a pair trace (solvers/difusco.py with CAM_SB_PAIR_TRACE=FILE: chains 2j and 2j+1 start from the same z) and,
optionally, a label-set file of the evaluated graphs (e.g. KaMIS multi-solutions of the first 64 validation graphs).

Metrics (per graph, then averaged over graphs):
  d_same   mean Hamming distance between the two final solutions of a pair (same start)
  d_diff   mean Hamming distance between final solutions of different pairs (different starts)
  ratio    d_same / d_diff: about 1 if the start does not matter (independent coupling), smaller if it does
  With labels:
  hit      fraction of chains whose final solution is closest (Hamming) to the label that overlaps the chain's start
           the most (argmax_m |z AND s_m|, ties broken by the smaller label index)
  hit_shuf the same with the starts shuffled across chains of the graph (chance level)
  d_near / d_rand  mean Hamming distance from the final solution to z's most-overlapping label / to a random label

Usage: python scripts/sb/start_dependence.py TRACE.jsonl [--labels LABELSET.pt] [--seed 0]
"""
import argparse
import json

import numpy as np
import torch


def bits(h, n):
    return np.array(list(bin(int(h, 16))[2:].zfill(n)), dtype=np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--labels', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    sets = torch.load(args.labels, weights_only=False)['sets'] if args.labels else None
    rows = [json.loads(l) for l in open(args.trace)]
    out = {k: [] for k in ('d_same', 'd_diff', 'hit', 'hit_shuf', 'd_near', 'd_rand', 'size')}
    for r in rows:
        n = r['n']
        Z = np.stack([bits(h, n) for h in r['start']]); F = np.stack([bits(h, n) for h in r['final']])
        k = len(Z)
        assert (Z[0::2] == Z[1::2]).all(), 'not a pair trace'
        D = (F[:, None, :] != F[None, :, :]).sum(-1)
        same = [D[2 * j, 2 * j + 1] for j in range(k // 2)]
        diff = [D[a, b] for a in range(k) for b in range(a + 1, k) if a // 2 != b // 2]
        out['d_same'].append(np.mean(same)); out['d_diff'].append(np.mean(diff)); out['size'].append(F.sum(1).mean())
        if sets is not None:
            S = sets[r['idx']].numpy().astype(np.int8)
            assert S.shape[1] == n
            def hit_rate(Zs):
                near = (Zs @ S.T).argmax(1)                                # most-overlapping label of each start
                dist = (F[:, None, :] != S[None, :, :]).sum(-1)            # (k, M)
                return float((dist.argmin(1) == near).mean()), float(dist[np.arange(k), near].mean()), dist
            h, dn, dist = hit_rate(Z)
            hs, _, _ = hit_rate(Z[rng.permutation(k)])
            out['hit'].append(h); out['hit_shuf'].append(hs); out['d_near'].append(dn)
            out['d_rand'].append(float(dist.mean()))
    res = {k: float(np.mean(v)) for k, v in out.items() if v}
    res['ratio'] = res['d_same'] / res['d_diff']
    res['graphs'] = len(rows)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
