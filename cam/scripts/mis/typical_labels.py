"""Learnability-thresholded labels: per training graph, keep only the label sizes a few-step sampler reaches often.

Learnability of a label size s on a graph = typicality = the fraction of a few-step sampler's samples (e.g. the
DiffUCO samples of ul_samples.py) with size >= s; it predicts how far a trained DIFUSCO falls short of its label
(experiments/20261119_difusco_rbsmall_reach, section 1). For a threshold tau, s*(graph) = the largest s with
typicality >= tau, and the label set is the sampler's own samples of size >= s* (with their multiplicities, so
--label_choice random draws from the sampler's distribution restricted to the learnable sizes). tau = 0 keeps only
the largest samples. Local addition, UPSTREAM.md item 54.
    python scripts/mis/typical_labels.py SAMPLES OUT_PREFIX TAU [TAU ...]      (writes OUT_PREFIX_tau{TAU}.pt)
"""
import sys, torch, numpy as np
src, pre, taus = sys.argv[1], sys.argv[2], [float(t) for t in sys.argv[3:]]
A = torch.load(src, weights_only=False)
for tau in taus:
    out, size, below = [], [], []
    for s in A['samples']:
        S, c = s['sets'], s['counts']; z = S.float().sum(1); N = c.sum().item()
        sstar = z.max().item()
        for v in sorted(set(z.tolist()), reverse=True):
            if c[z >= v].sum().item() / N >= tau:
                sstar = v; break
        k = z >= sstar
        out.append(S[k].repeat_interleave(c[k], 0).to(torch.uint8))
        size.append(float((z[k] * c[k]).sum() / c[k].sum())); below.append(sstar < z.max().item())
    torch.save({'files': A['files'], 'sets': out,
                'meta': {'from': src, 'tau': tau, 'rule': 'samples of size >= largest size with typicality >= tau'}},
               f'{pre}_tau{tau:g}.pt')
    print(f'tau {tau:g}: mean label size {np.mean(size):.3f}, graphs below the largest size {np.mean(below):.3f}, '
          f'labels per graph {np.mean([o.shape[0] for o in out]):.0f}', flush=True)
