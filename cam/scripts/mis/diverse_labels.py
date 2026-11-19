"""Diverse label sets: per training graph, the unique largest samples of a solver (ul_samples.py / sl_samples.py
output), all of them or M chosen by farthest-point sampling in Hamming distance (start: the medoid), for
--label_choice random. Local addition, UPSTREAM.md item 53.
    python scripts/mis/diverse_labels.py SAMPLES OUT_PREFIX [M ...]      (writes OUT_PREFIX_all.pt, OUT_PREFIX_fpsM.pt)
"""
import sys, torch, numpy as np
src, pre, Ms = sys.argv[1], sys.argv[2], [int(m) for m in sys.argv[3:]]
A = torch.load(src, weights_only=False)
tops = []
for s in A['samples']:
    S = s['sets']; z = S.float().sum(1); tops.append(torch.unique(S[z == z.max()], dim=0))


def fps(t, m):
    if t.shape[0] <= m:
        return t
    f = t.float(); mu = f.mean(0)
    first = (f * mu + (1 - f) * (1 - mu)).sum(1).argmax().item()
    idx = [first]; d = (f - f[first]).abs().sum(1)
    while len(idx) < m:
        j = d.argmax().item(); idx.append(j); d = torch.minimum(d, (f - f[j]).abs().sum(1))
    return t[idx]


def hamming(t):
    if t.shape[0] < 2:
        return 0.0
    f = t.float(); return ((f[:, None] - f[None]).abs().sum(-1).sum() / (t.shape[0] * (t.shape[0] - 1))).item()


meta = {'from': src}
torch.save({'files': A['files'], 'sets': tops, 'meta': dict(meta, rule='all unique largest samples')}, pre + '_all.pt')
print(f"all: sets/graph {np.mean([t.shape[0] for t in tops]):.1f}, size {np.mean([float(t[0].sum()) for t in tops]):.3f}, "
      f"mean pairwise Hamming {np.mean([hamming(t) for t in tops[:500]]):.2f}")
for m in Ms:
    sel = [fps(t, m) for t in tops]
    torch.save({'files': A['files'], 'sets': sel, 'meta': dict(meta, rule=f'farthest-point {m} of the largest samples')},
               f'{pre}_fps{m}.pt')
    print(f"fps{m}: sets/graph {np.mean([t.shape[0] for t in sel]):.1f}, mean pairwise Hamming {np.mean([hamming(t) for t in sel[:500]]):.2f}")
