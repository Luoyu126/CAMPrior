"""Labels the current student has not learned yet: the student is the learnability quantifier.

Per training graph, the candidates are the largest samples of a few-step sampler (e.g. DiffUCO, ul_samples.py:
learnable by construction, experiments/20261119_difusco_rbsmall_reach sections 1-3). A trained DIFUSCO student's
own samples on the training graphs (sl_samples.py) show what it has learned: a candidate at Hamming distance d from
every student sample is a solution the student does not produce. The label set is the K candidates farthest from
the student, thinned to M by farthest-point sampling (start: the farthest from the student); learned and not-yet-
learned solutions of the same (largest) size, so only where the student concentrates on a few modes does the label
set move away from them. Control: PREFIX_near (the K nearest, then M). Use --label_choice random.
Local addition, UPSTREAM.md item 57.
    python scripts/mis/coverage_labels.py SAMPLER_SAMPLES STUDENT_SAMPLES OUT_PREFIX M K
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


src, stu, pre, M, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
A = torch.load(src, weights_only=False); B = torch.load(stu, weights_only=False)
assert A['files'] == B['files']
far, near, dfar, dnear, reach = [], [], [], [], []
for s, t in zip(A['samples'], B['samples']):
    S = s['sets']; z = S.float().sum(1); c = S[z == z.max()]
    T = t['sets'].float()
    d = torch.cdist(c.float(), T, p=1).min(1).values             # Hamming to the nearest student sample
    o = torch.argsort(d, descending=True)
    far.append(fps(c[o[:K]], M)); near.append(fps(c[o.flip(0)[:K]], M))
    dfar.append(float(d[o[:K]].mean())); dnear.append(float(d[o.flip(0)[:K]].mean()))
    reach.append(float((T.sum(1) >= z.max()).float().mean()))
reach = np.array(reach)
print(f'student reaches the label size in {reach.mean():.3f} of its samples; graphs it never reaches {np.mean(reach == 0):.3f}')
print(f'distance of the chosen candidates to the student: far {np.mean(dfar):.2f}, near {np.mean(dnear):.2f}')
for name, sel in (('far', far), ('near', near)):
    torch.save({'files': A['files'], 'sets': sel, 'meta': {'from': [src, stu], 'rule': f'{name} {K} from the student, fps {M}'}},
               f'{pre}_{name}{K}_fps{M}.pt')
    print(f'{name}: sets/graph {np.mean([x.shape[0] for x in sel]):.1f}, pairwise Hamming {np.mean([hamming(x) for x in sel[:500]]):.2f}')
