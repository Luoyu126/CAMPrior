"""Rule-based choice among the optimal solutions of each graph, and how consistent the choice is
(local addition, UPSTREAM.md item 44).

Rule (degree): among the distinct maximum independent sets of a graph (build_solution_sets.py), take
the one with the smallest sum of node degrees; ties broken by the smallest sum over chosen nodes of
their neighbours' degree sums; remaining ties by the first in the file (the label if it is tied).

Consistency is measured by how well local structure predicts the chosen label: a logistic regression
on per-node features (degree, neighbour-degree statistics, graph size) is fit on 3/4 of the graphs
and scored on the rest (cross-entropy and AUC), for the KaMIS label, the rule label and a uniformly
random member of each set. A choice that follows the structure consistently across graphs is easier
to predict. Writes the rule labels as a one-set-per-graph file for --label_set ... --label_choice first.
    python scripts/mis/rule_labels.py --graphs DIR --sets FILE --out FILE
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import datasets.mis as M  # noqa: E402


def node_features(edge_index, n):
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (n, n)).coalesce()
    deg = torch.sparse.sum(A, 1).to_dense()
    nbr_sum = torch.sparse.mm(A, deg[:, None]).squeeze(1)
    src, dst = edge_index
    nbr_max = torch.zeros(n).scatter_reduce(0, src, deg[dst], 'amax', include_self=False)
    nbr_min = torch.full((n,), 1e9).scatter_reduce(0, src, deg[dst], 'amin', include_self=True)
    f = torch.stack([deg, nbr_sum / deg.clamp_min(1), nbr_max, nbr_min, deg / deg.max(),
                     (deg - deg.mean()) / deg.std().clamp_min(1e-6), torch.full((n,), float(n))], 1)
    return f.numpy(), deg, nbr_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs', required=True)
    ap.add_argument('--sets', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    S = torch.load(a.sets, weights_only=False)
    ds = M.Dataset(a.graphs, torch.float32, len(S['sets']))
    rng = np.random.default_rng(0)
    feats, lab_k, lab_r, lab_u, gid = [], [], [], [], []
    rule_sets, ties1, ties2, changed, share, nsets = [], 0, 0, 0, [], []
    for i in range(len(S['sets'])):
        g = ds[i]
        s = S['sets'][i].float()
        f, deg, nbr_sum = node_features(g.edge_index, g.num_nodes)
        sc1 = s @ deg
        cand = (sc1 == sc1.min()).nonzero().view(-1)
        ties1 += len(cand) > 1
        if len(cand) > 1:
            sc2 = s[cand] @ nbr_sum
            cand = cand[sc2 == sc2.min()]
            ties2 += len(cand) > 1
        r = s[cand[0]]
        y = g.x.view(-1).float()
        changed += not torch.equal(r, y)
        share.append((r * y).sum().item() / y.sum().item())
        nsets.append(s.shape[0])
        rule_sets.append(r[None].to(torch.uint8))
        feats.append(f); lab_k.append(y.numpy()); lab_r.append(r.numpy())
        lab_u.append(s[rng.integers(s.shape[0])].numpy()); gid.append(np.full(g.num_nodes, i))
    out = {'files': S['files'], 'sets': rule_sets, 'meta': {'rule': 'min degree sum, then min neighbour-degree sum',
                                                             'from': os.path.abspath(a.sets)}}
    torch.save(out, a.out)
    n = len(rule_sets)
    print(f"{n} graphs; optimal sets per graph mean {np.mean(nsets):.1f}, median {np.median(nsets):.0f}, "
          f"single {sum(x == 1 for x in nsets)}")
    print(f"rule: tied after degree sum {ties1}/{n} ({ties1 / n:.1%}), still tied after neighbour-degree sum "
          f"{ties2}/{n} ({ties2 / n:.1%}); rule label differs from KaMIS on {changed}/{n}; "
          f"shared with KaMIS {np.mean(share):.1%} of nodes")
    X = np.concatenate(feats); G = np.concatenate(gid)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    tr, te = G < int(0.75 * n), G >= int(0.75 * n)
    for name, L in [('KaMIS label', lab_k), ('rule label', lab_r), ('random optimal set', lab_u)]:
        Y = np.concatenate(L)
        clf = LogisticRegression(max_iter=2000).fit(X[tr], Y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        base = log_loss(Y[te], np.full(te.sum(), Y[tr].mean()))
        print(f"  {name:20s}: held-out CE {log_loss(Y[te], p):.4f} (constant {base:.4f}), AUC {roc_auc_score(Y[te], p):.3f}")


if __name__ == '__main__':
    main()
