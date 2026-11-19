"""Select, per training graph, the M most predictable labels among candidate sets (cross-fitted probe).

Two folds of the training graphs. For each fold, a one-step probe (as in label_probe.py: EncodeProcessDecode,
uninformative input) is trained on the OTHER fold's reference labels (--ref, e.g. the DiffUCO k200 labels) and
scores every candidate set of this fold's graphs by its log-likelihood under the probe (sum over nodes of
log p(label)). Candidates: the unique largest sets among --cands (e.g. all DiffUCO samples, ul_samples.py output).
Writes one label-set file per M with the M highest-scoring candidates of each graph (use --label_choice random).
Local addition, UPSTREAM.md item 52.
    python scripts/mis/probe_select.py --cands FILE --ref FILE --graphs DIR --out_prefix P --M 1 4 16
"""
import argparse, os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import datasets.mis as M
from torch_geometric.loader import DataLoader
from models import EncodeProcessDecode
from utils.label_sets import load_label_sets

ap = argparse.ArgumentParser()
ap.add_argument('--cands', required=True); ap.add_argument('--ref', required=True)
ap.add_argument('--graphs', required=True); ap.add_argument('--out_prefix', required=True)
ap.add_argument('--M', type=int, nargs='+', default=[1, 4, 16])
ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); dev = 'cuda'
C = torch.load(a.cands, weights_only=False)
n = len(C['samples'])
ref = load_label_sets(a.ref, a.graphs, n)
ds = M.Dataset(a.graphs, torch.float32, n)
graphs = [ds[i] for i in range(n)]
for i, g in enumerate(graphs):
    g.x = ref[i][0].view(-1, 1).float()
cands = []
for s in C['samples']:
    S = s['sets']; z = S.float().sum(1); cands.append(torch.unique(S[z == z.max()], dim=0))
fold = torch.arange(n) % 2
scores = [None] * n
for f in (0, 1):
    model = EncodeProcessDecode(n_layers=8, hidden_dim=64, out_channels=2, use_activation_checkpoint=False,
                                problem='mis', time_scale=1).to(dev)
    tr = DataLoader([graphs[i] for i in range(n) if fold[i] != f], batch_size=40, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tr))
    for e in range(a.epochs):
        for g in tr:
            g = g.to(dev); x0 = torch.full((g.num_nodes, 1), 0.5, device=dev)
            loss = F.cross_entropy(model(g, x0, torch.zeros(g.num_nodes, device=dev)), g.x.view(-1).long())
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    model.eval()
    with torch.no_grad():
        for i in range(n):
            if fold[i] != f:
                continue
            g = graphs[i].to(dev); g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=dev)
            lp = torch.log_softmax(model(g, torch.full((g.num_nodes, 1), 0.5, device=dev),
                                         torch.zeros(g.num_nodes, device=dev)), -1)
            c = cands[i].float().to(dev)
            scores[i] = (c @ lp[:, 1] + (1 - c) @ lp[:, 0]).cpu()
    print('fold', f, 'done', flush=True)
for m in a.M:
    sets = [cands[i][torch.argsort(scores[i], descending=True)[:m]].to(torch.uint8) for i in range(n)]
    torch.save({'files': C['files'], 'sets': sets,
                'meta': {'cands': a.cands, 'ref': a.ref, 'M': m, 'rule': 'top-M by cross-fitted probe log-likelihood'}},
               f'{a.out_prefix}_M{m}.pt')
    print(f'M={m}: mean sets per graph {sum(s.shape[0] for s in sets) / n:.2f}, size {sum(float(s[0].sum()) for s in sets) / n:.3f}')
torch.save({'scores': scores, 'cands': cands}, f'{a.out_prefix}_scores.pt')
