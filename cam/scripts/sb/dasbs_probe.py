"""2C diagnostic: what a trained DASBS controller does along a tau-leaping rollout on one RB-small graph (the first
validation-anchor graph, 20 chains). Per step: mean set size and violated edges, the predicted log Phi and flip
probability on selected / unselected bits; at the end state, the ctrl-AM target log ratios vs the prediction.
Used for 2C round 2 (docs/20261125/status.md): every flip probability 1 = the degenerate fixed point.
Usage (from cam/, env.sh sourced): CAM_DIR=$PWD CUDA_VISIBLE_DEVICES=g python scripts/sb/dasbs_probe.py CKPT ALPHA STEPS"""
import sys, os, math, torch
sys.path.insert(0, os.environ['CAM_DIR'])
import models
from datasets.mis import Dataset
from solvers.dasbs import DASBSNet, corrector_log_bound, gamma_integral
from utils import dasbs as D
ckpt, alpha, steps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
ref = D.UniformFlipReference('log_linear', 1.0, alpha)
ctrl = models.EncodeProcessDecodeTime2(n_layers=8, hidden_dim=64, out_channels=1, problem='mis', time_scale=1000)
corr = models.EncodeProcessDecode(n_layers=8, hidden_dim=64, out_channels=1, problem='mis', time_scale=1000)
net = DASBSNet(ctrl, corr, corrector_log_bound(ref), ctrl_scale=True).cuda().eval()
net.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=True))
tau = float(net.ctrl_tau); print('ctrl_tau', tau)
ds = Dataset('/usr0/home/shengyuf/yunyi/data/rb_small_valanchor', torch.float32, 1)
g = ds[0].cuda(); n = g.num_nodes; g.batch = torch.zeros(n, dtype=torch.long, device='cuda')
A = torch.zeros(n, n, device='cuda'); A[g.edge_index[0], g.edge_index[1]] = 1
grid = D.time_grid(steps, 'quadratic')
torch.manual_seed(0)
x = torch.randint(0, 2, (n, 20), device='cuda').float()
with torch.no_grad():
    for k in range(steps):
        t0, t1 = float(grid[k]), float(grid[k + 1]); G = gamma_integral(ref, t0, t1)
        if math.isinf(G):
            q = torch.full_like(x, .5); lp = None
        else:
            lp = net.ctrl(g, x, torch.full_like(x, t0)).double(); q = (G / 2 * lp.exp()).clamp(max=1).float()
        s = x.bool()
        msg = f't0 {t0:.3f} G {G:.3g} |x| {x.sum(0).mean():.1f} viol {((x * (A @ x)).sum(0) / 2).mean():.0f}'
        if lp is not None:
            msg += f' | logPhi sel {lp[s].mean():.1f} [{lp[s].min():.1f},{lp[s].max():.1f}] unsel {lp[~s].mean():.1f} [{lp[~s].min():.1f},{lp[~s].max():.1f}] | q sel {q[s].mean():.3f} unsel {q[~s].mean():.3f}'
        print(msg)
        x = torch.where(torch.rand_like(x) < q, 1 - x, x)
    s = x.bool(); dH = (1 - 2 * x) * (-1 + 1.2 * (A @ x)); lt = (-dH / tau).double() - net.corr(g, x).double()
    lp = net.ctrl(g, x, torch.full_like(x, 0.95)).double()
    print(f'end |x| {x.sum(0).mean():.1f} viol {((x * (A @ x)).sum(0) / 2).mean():.0f}')
    print(f'target logratio sel {lt[s].mean():.1f} [{lt[s].min():.1f},{lt[s].max():.1f}]  unsel {lt[~s].mean():.1f} [{lt[~s].min():.1f},{lt[~s].max():.1f}]')
    print(f'pred logPhi(t=.95) sel {lp[s].mean():.1f}  unsel {lp[~s].mean():.1f}')
