"""Learnability of candidate labels under a reference DIFUSCO model.

D(y) = mean over t = 1..T of the node-mean cross-entropy of model(x_t, t) against y, x_t ~ q(x_t | y): the
training loss the label would have under that model (one noise draw per t). Candidates per training graph:
the KaMIS label, the DiffUCO k20 / k200 labels and all unique DiffUCO samples (train_diffuco_all200.pt).
    python scripts/mis/label_learnability.py --ckpt CK --graphs DIR --label_dir DIR --n 1000 --out OUT.pt
Local addition, UPSTREAM.md item 49. Finding (RB-small): D predicts little of whether a label is learned; how often a
few-step sampler reaches the label size predicts most of it (experiments/20261119_difusco_rbsmall_reach/README.md).
"""
import argparse, os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import datasets.mis as M
from models import EncodeProcessDecode
from utils.diffusion_schedulers import CategoricalDiffusion

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--num_tp', type=int, default=50)
ap.add_argument('--schedule', default='cosine'); ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--graphs', required=True, help='training graph directory')
ap.add_argument('--label_dir', required=True, help='dir with train_diffuco.pt, train_diffuco_k200.pt, train_diffuco_all200.pt')
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = 'cuda'
L = a.label_dir
k20 = torch.load(f'{L}/train_diffuco.pt', weights_only=False)['sets']
k200 = torch.load(f'{L}/train_diffuco_k200.pt', weights_only=False)['sets']
allS = torch.load(f'{L}/train_diffuco_all200.pt', weights_only=False)['samples']
model = EncodeProcessDecode(n_layers=8, hidden_dim=64, out_channels=2, use_activation_checkpoint=False,
                            problem='mis', time_scale=1).to(dev)
model.load_state_dict(torch.load(a.ckpt, weights_only=True, map_location=dev)); model.eval()
diff = CategoricalDiffusion(T=a.num_tp, schedule=a.schedule)
ds = M.Dataset(a.graphs, torch.float32, a.n)
out = []
with torch.no_grad():
    for i in range(a.n):
        g = ds[i].to(dev); n = g.num_nodes
        g.batch = torch.zeros(n, dtype=torch.long, device=dev)
        u = allS[i]
        C = torch.cat([g.x.view(1, -1).float().cpu(), k20[i][:1].float(), k200[i][:1].float(), u['sets'].float()])
        src = ['kamis', 'k20', 'k200'] + ['ul'] * u['sets'].shape[0]
        cnt = torch.cat([torch.zeros(3), u['counts'].float()])
        C = C.to(dev); K = C.shape[0]
        oh = F.one_hot(C.T.long(), 2).float()                     # n x K x 2
        D = torch.zeros(K, device=dev)
        for t in range(1, a.num_tp + 1):
            xt = diff.sample(oh, t)                               # n x K
            logits = model(g, xt.unsqueeze(-1), torch.full((n, K), float(t), device=dev))
            D += F.cross_entropy(logits.reshape(-1, 2), C.T.reshape(-1).long(), reduction='none').view(n, K).mean(0)
        out.append({'size': C.sum(1).cpu(), 'D': (D / a.num_tp).cpu(), 'src': src, 'count': cnt})
        if i % 200 == 0:
            print(i, flush=True)
torch.save({'ckpt': a.ckpt, 'graphs': out}, a.out)
print('done')
