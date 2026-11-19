"""Reachable labels from a student's own 50-step samples (one round of label co-evolution).

For each training graph, from the samples file (scripts/mis/sl_samples.py, num_k samples per graph):
  self   the medoid of the largest samples (agrees most with the others: most reachable)
  union  self, or the reference label (--ref, e.g. DiffUCO k200) if that one is strictly larger
Also prints reachability statistics: label size, fraction of samples reaching it, vs the reference label.
    python scripts/mis/reach_labels.py SAMPLES REF OUT_PREFIX
Local addition, UPSTREAM.md item 50.
"""
import sys, torch, numpy as np
samples, ref, pre = sys.argv[1], sys.argv[2], sys.argv[3]
A = torch.load(samples, weights_only=False); R = torch.load(ref, weights_only=False)
assert A['files'] == R['files'][:len(A['files'])]
self_sets, union_sets, st = [], [], []
for i, s in enumerate(A['samples']):
    S = s['sets']; z = S.float().sum(1)
    top = S[z == z.max()]
    # medoid of the largest samples: the one agreeing most with the others (highest marginal under the student)
    t = top.float(); m = t.mean(0)
    agree = (t * m + (1 - t) * (1 - m)).sum(1)
    lab = top[agree.argmax()][None]
    r = R['sets'][i][:1].to(torch.uint8)
    self_sets.append(lab)
    union_sets.append(r if r.float().sum() > lab.float().sum() else lab)
    st.append((z.max().item(), (z >= z.max()).float().mean().item(), r.float().sum().item(),
               (z >= r.float().sum()).float().mean().item(), z.mean().item(), s['kamis']))
st = np.array(st)
print(f'graphs {len(st)}; per-sample mean {st[:,4].mean():.3f}; self label {st[:,0].mean():.3f} (reached by {st[:,1].mean()*100:.1f}% of samples); '
      f'ref label {st[:,2].mean():.3f} (reached by {st[:,3].mean()*100:.1f}%); KaMIS {st[:,5].mean():.3f}')
print(f'self > ref on {(st[:,0] > st[:,2]).mean()*100:.1f}% graphs, = {(st[:,0] == st[:,2]).mean()*100:.1f}%, < {(st[:,0] < st[:,2]).mean()*100:.1f}%; '
      f'union mean {np.maximum(st[:,0], st[:,2]).mean():.3f}')
meta = {'from': samples, 'ref': ref}
torch.save({'files': A['files'], 'sets': self_sets, 'meta': dict(meta, rule='medoid of the largest samples')}, pre + '_self.pt')
torch.save({'files': A['files'], 'sets': union_sets, 'meta': dict(meta, rule='self or ref if strictly larger')}, pre + '_union.pt')
