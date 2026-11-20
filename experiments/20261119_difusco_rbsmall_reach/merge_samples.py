"""Merge several ul_samples.py outputs on the same graphs into one (unique sets per graph, counts summed)."""
import sys, torch
out, srcs = sys.argv[1], sys.argv[2:]
A = [torch.load(s, weights_only=False) for s in srcs]
assert all(a['files'] == A[0]['files'] for a in A)
samples = []
for per in zip(*[a['samples'] for a in A]):
    S = torch.cat([p['sets'] for p in per]); C = torch.cat([p['counts'] for p in per])
    u, inv = torch.unique(S, dim=0, return_inverse=True)
    c = torch.zeros(u.shape[0], dtype=C.dtype).index_add_(0, inv, C)
    samples.append(dict(sets=u, counts=c, sizes=u.float().sum(1), kamis=per[0]['kamis']))
torch.save({'files': A[0]['files'], 'samples': samples, 'meta': {'merged': srcs}}, out)
print('merged', len(samples), 'graphs; max size', sum(float(s['sizes'].max()) for s in samples) / len(samples))
