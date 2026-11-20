"""Labels from a stronger solver, filtered by learnability.

Per training graph, the candidates are the largest sets of the pool {few-step sampler's samples (e.g. DiffUCO,
ul_samples.py)} u {a stronger solver's solutions (e.g. RLSA, build_solution_sets.py)}. Learnability of a candidate =
its Hamming distance to the nearest sample of the few-step sampler (any size): sets the sampler nearly produces
are the ones a DIFUSCO trained on them reproduces (experiments/20261119_difusco_rbsmall_reach, sections 1-3).
Writes, for each K, the K most learnable candidates thinned to M by farthest-point sampling (PREFIX_learnK_fpsM.pt),
and the control PREFIX_any_fpsM.pt (farthest-point M of all candidates, learnability ignored). Use
--label_choice random. Local addition, UPSTREAM.md item 55.
    python scripts/mis/strong_labels.py SAMPLER_SAMPLES STRONG_SETS OUT_PREFIX M K [K ...]
"""
import sys, torch, numpy as np


def fps(t, m):   # farthest-point sampling in Hamming distance, starting at the first row (as in diverse_labels.py)
    if t.shape[0] <= m:
        return t
    f = t.float(); idx = [0]; d = (f - f[0]).abs().sum(1)
    while len(idx) < m:
        j = d.argmax().item(); idx.append(j); d = torch.minimum(d, (f - f[j]).abs().sum(1))
    return t[idx]


def hamming(t):
    if t.shape[0] < 2:
        return 0.0
    f = t.float(); return ((f[:, None] - f[None]).abs().sum(-1).sum() / (t.shape[0] * (t.shape[0] - 1))).item()


src, strong, pre, M, Ks = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), [int(k) for k in sys.argv[5:]]
A = torch.load(src, weights_only=False); B = torch.load(strong, weights_only=False)
assert A['files'] == B['files']
cands, dist, grew = [], [], []
for s, t in zip(A['samples'], B['sets']):
    S = s['sets']; zs = S.float().sum(1)
    pool = torch.unique(torch.cat([S[zs == zs.max()], t.to(S.dtype)]), dim=0); z = pool.float().sum(1)
    c = pool[z == z.max()]
    d = torch.cdist(c.float(), S.float(), p=1).min(1).values      # Hamming to the nearest sampler sample
    o = torch.argsort(d); cands.append(c[o]); dist.append(d[o]); grew.append(z.max().item() > zs.max().item())
print(f'graphs where the stronger solver is larger: {np.mean(grew):.3f}; candidates per graph '
      f'{np.mean([c.shape[0] for c in cands]):.1f}; size {np.mean([float(c[0].sum()) for c in cands]):.3f}')
outs = {f'any_fps{M}': [fps(c, M) for c in cands]}
for K in Ks:
    outs[f'learn{K}_fps{M}'] = [fps(c[:K], M) for c in cands]
for name, sel in outs.items():
    dd = [float(torch.cdist(x.float(), A['samples'][i]['sets'].float(), p=1).min(1).values.mean()) for i, x in enumerate(sel[:500])]
    torch.save({'files': A['files'], 'sets': sel, 'meta': {'from': [src, strong], 'rule': name}}, f'{pre}_{name}.pt')
    print(f'{name}: sets/graph {np.mean([x.shape[0] for x in sel]):.1f}, pairwise Hamming '
          f'{np.mean([hamming(x) for x in sel[:500]]):.2f}, distance to the sampler {np.mean(dd):.2f}', flush=True)
