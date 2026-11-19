"""Training sets of this experiment (run once; writes under /usr0/home/shengyuf/yunyi/data/rb_small_replay).

Input: the nearest larger independent set found from each round-1 DiffUCO label (data/rb_small_ullabels/
train_diffuco.pt) by warm-started RLSA, from runs/scratch/stepwise_1119/sets/*.first.pt (five RLSA settings,
30-1000 steps x 200-1000 chains; for every graph whose label is below the KaMIS size, the result of size
>= label+1 nearest to the label). Graphs whose nearest improvement is within Hamming 10 (53 of 4000) get it
as label. train_rep{R}: all 4000 graphs (symlinks) plus R-1 extra copies of those 53;
rep{R}_improved.pt / rep{R}_original.pt: their labels with / without the improvement.
"""
import glob, os, sys
import torch

S = sys.argv[1] if len(sys.argv) > 1 else 'runs/scratch/stepwise_1119/sets'
D = '/usr0/home/shengyuf/yunyi/data/rb_small_replay'; G = '/usr1/shengyuf/MIS/RB_small_train'
UL = torch.load('/usr0/home/shengyuf/yunyi/data/rb_small_ullabels/train_diffuco.pt', weights_only=False)
best = {}
for f in glob.glob(S + '/*.first.pt'):
    for i, s in torch.load(f).items():
        h = int((s.float() - UL['sets'][i][0].float()).abs().sum())
        if i not in best or h < best[i][0]:
            best[i] = (h, s)
imp = {UL['files'][i]: s[None] for i, (h, s) in best.items() if h <= 10}
orig = dict(zip(UL['files'], UL['sets']))
os.makedirs(D, exist_ok=True)
for r in [3, 10]:
    d = f'{D}/train_rep{r}'; os.makedirs(d, exist_ok=True)
    links = [(f, f) for f in UL['files']] + [(f'rep{k}_{f}', f) for f in imp for k in range(1, r)]
    for name, f in links:
        if not os.path.exists(f'{d}/{name}'):
            os.symlink(f'{G}/{f}', f'{d}/{name}')
    files = sorted(os.path.basename(p) for p in glob.glob(d + '/*gpickle'))
    base = lambda f: f.split('_', 1)[1] if f.startswith('rep') else f
    torch.save({'files': files, 'sets': [imp.get(base(f), orig[base(f)]) for f in files]}, f'{D}/rep{r}_improved.pt')
    torch.save({'files': files, 'sets': [orig[base(f)] for f in files]}, f'{D}/rep{r}_original.pt')
    print(r, len(files), 'graphs,', len(imp), 'improved')
