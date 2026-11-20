"""Label ceiling vs learnability, per model, on the first 1000 training graphs.
ceiling   = mean size of the largest label of the graph's label set (what the model is asked to produce)
reached   = model best-of-20 on that graph (renoise, 50 steps)   -> unlearned = ceiling - reached
DiffUCO   = DiffUCO best-of-20 on the same graph (train_diffuco.pt k20 labels)
hit       = fraction of graphs where the model's best-of-20 reaches the label size
    python analyze.py TAG:LABELFILE [TAG:LABELFILE ...]
"""
import sys, torch, numpy as np
U = '/usr0/home/shengyuf/yunyi/data/rb_small_ullabels'
du = np.array([float(s[0].sum()) for s in torch.load(f'{U}/train_diffuco.pt', weights_only=False)['sets'][:1000]])
print(f'DiffUCO best-of-20 on these graphs: {du.mean():.3f}')
print('tag | ceiling | reached | unlearned | hit | per-sample mean | std | vs DiffUCO win/lose')
for arg in sys.argv[1:]:
    tag, lab = arg.split(':', 1)
    P = torch.load(f'tr_{tag}.pt'); idx = sorted(P)[:1000]
    S = torch.load(lab, weights_only=False)['sets']
    ceil = np.array([float(S[i].float().sum(1).max()) for i in idx])
    smp = np.stack([P[i].numpy() for i in idx]); best = smp.max(1); d = du[idx]
    print(f'{tag} | {ceil.mean():.3f} | {best.mean():.3f} | {(ceil - best).mean():.3f} | {(best >= ceil).mean():.3f} | '
          f'{smp.mean():.3f} | {smp.std(1).mean():.3f} | {(best > d).mean():.3f}/{(best < d).mean():.3f}')
