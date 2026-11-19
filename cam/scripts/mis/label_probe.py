"""Dataset-level learnability probe of a label set: how well a fresh network predicts the labels of unseen graphs.

Trains the DIFUSCO network (EncodeProcessDecode) to predict each training graph's label in ONE step from an
uninformative input (all node states 0.5), on the first --n_fit training graphs, and reports the node
cross-entropy and the greedy-decoded size on the next --n_eval graphs (labels never seen). A label rule that is
consistent across graphs generalises; arbitrary choices among equally good solutions do not
(cf. SymILO, arXiv 2409.19678, abstract). Local addition, UPSTREAM.md item 51.
    python scripts/mis/label_probe.py --labels FILE|kamis --graphs DIR --out OUT.json
"""
import argparse, json, os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import datasets.mis as M
from torch_geometric.loader import DataLoader
from models import EncodeProcessDecode
from utils.label_sets import load_label_sets

ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True, help="label-set file, or 'kamis' for graph.x")
ap.add_argument('--graphs', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n_fit', type=int, default=3000); ap.add_argument('--n_eval', type=int, default=1000)
ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); dev = 'cuda'
n = a.n_fit + a.n_eval
ds = M.Dataset(a.graphs, torch.float32, n)
sets = None if a.labels == 'kamis' else load_label_sets(a.labels, a.graphs, n)
graphs = []
for i in range(n):
    g = ds[i]
    if sets is not None:
        g.x = sets[i][0].view(-1, 1).float()
    graphs.append(g)
fit = DataLoader(graphs[:a.n_fit], batch_size=40, shuffle=True)
ev = DataLoader(graphs[a.n_fit:], batch_size=40)
model = EncodeProcessDecode(n_layers=8, hidden_dim=64, out_channels=2, use_activation_checkpoint=False,
                            problem='mis', time_scale=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(fit))


def run(loader, train):
    model.train(train); ce = cnt = 0; sizes = []
    for g in loader:
        g = g.to(dev); x0 = torch.full((g.num_nodes, 1), 0.5, device=dev)
        with torch.set_grad_enabled(train):
            logits = model(g, x0, torch.zeros(g.num_nodes, device=dev))
            loss = F.cross_entropy(logits, g.x.view(-1).long())
        if train:
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        else:   # greedy decoding of the heatmap: highest probability first, skip neighbours
            p = logits.softmax(-1)[:, 1]
            for b in range(g.num_graphs):
                lo, hi = g.ptr[b].item(), g.ptr[b + 1].item(); pb = p[lo:hi]
                ei = g.edge_index[:, (g.edge_index[0] >= lo) & (g.edge_index[0] < hi)] - lo
                nb = [[] for _ in range(hi - lo)]
                for u, v in ei.T.tolist(): nb[u].append(v)
                taken = torch.zeros(hi - lo, dtype=torch.bool); blocked = taken.clone()
                for u in torch.argsort(pb, descending=True).tolist():
                    if not blocked[u]:
                        taken[u] = True; blocked[u] = True; blocked[nb[u]] = True
                sizes.append(taken.sum().item())
        ce += loss.item() * g.num_nodes; cnt += g.num_nodes
    return ce / cnt, (sum(sizes) / len(sizes) if sizes else None)


hist = []
for e in range(a.epochs):
    tr, _ = run(fit, True)
    if (e + 1) % 10 == 0 or e + 1 == a.epochs:
        ce, sz = run(ev, False); hist.append({'epoch': e + 1, 'fit_ce': tr, 'eval_ce': ce, 'eval_size': sz})
        print(hist[-1], flush=True)
lab = sum(float(g.x.sum()) for g in graphs[a.n_fit:]) / a.n_eval
json.dump({'labels': a.labels, 'eval_label_size': lab, 'hist': hist, 'args': vars(a)}, open(a.out, 'w'), indent=1)
