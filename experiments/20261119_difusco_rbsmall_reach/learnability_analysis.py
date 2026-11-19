"""Learnability diagnostic (README section 1): which per-graph quantity predicts the gap between a label and what a
trained 50-step DIFUSCO samples on that training graph (best of 20)?  Inputs, in the working directory:
  D_refkamis.pt / D_refk200.pt   cam/scripts/mis/label_learnability.py with the tp50 KaMIS / k200 checkpoints
  train_tp50_*.txt.per_graph.csv main.py test of those checkpoints on the first 1000 training graphs
"""
import torch, numpy as np, csv
from scipy.stats import spearmanr
A = torch.load('D_refkamis.pt', weights_only=False)['graphs']; B = torch.load('D_refk200.pt', weights_only=False)['graphs']
def pg(f):
    r = {int(x['idx']): float(x['obj']) for x in csv.DictReader(open(f))}; return np.array([abs(r[i]) for i in range(1000)])
s_k0, s_k1, s_ka = pg('train_tp50_k200_s0.txt.per_graph.csv'), pg('train_tp50_k200_s1.txt.per_graph.csv'), pg('train_tp50_kamis_s0.txt.per_graph.csv')
n = 1000
size = lambda g, j: float(g['size'][j])
# Q1 within-graph spread among max-size DiffUCO samples (ref = KaMIS model, never saw DiffUCO labels)
spread, corr_cnt, corr_size, between = [], [], [], []
for g in A:
    src = np.array(g['src']); ul = np.where(src == 'ul')[0]; sz = g['size'].numpy(); D = g['D'].numpy(); c = g['count'].numpy()
    top = ul[sz[ul] == sz[ul].max()]
    if len(top) >= 3: spread.append((D[top].std(), D[top].max() - D[top].min()))
    if len(ul) >= 5:
        corr_size.append(spearmanr(sz[ul], D[ul])[0]); corr_cnt.append(spearmanr(c[ul], D[ul])[0])
    between.append(D[top].mean())
spread = np.array(spread)
print(f'Q1 graphs with >=3 max-size DiffUCO sets: {len(spread)}; within-graph std of D {spread[:,0].mean():.4f}, range {spread[:,1].mean():.4f}; between-graph std of mean D {np.std(between):.4f}')
print(f'   within graph, Spearman(D, size) mean {np.nanmean(corr_size):.3f}; Spearman(D, DiffUCO count) mean {np.nanmean(corr_cnt):.3f}')
# Q2 KaMIS label vs DiffUCO max-size sets of the same or smaller size, under the k200 model (never saw KaMIS labels)
rk, dd = [], []
for g in B:
    src = np.array(g['src']); sz = g['size'].numpy(); D = g['D'].numpy()
    ul = np.where(src == 'ul')[0]; same = ul[sz[ul] == sz[0]]
    if len(same) >= 3: rk.append((D[same] < D[0]).mean())
    dd.append(D[0] - D[2])
print(f'Q2 graphs where some DiffUCO sample has the KaMIS size: {len(rk)}; KaMIS label harder than fraction {np.mean(rk):.2f} of them (0.5 = typical)')
print(f'   D(KaMIS) - D(k200 label) under k200 model: mean {np.mean(dd):.4f} (biased: model trained on k200)')
# Q3 does D under a model that never saw the label predict whether it is learned?
lab_k200 = np.array([size(g, 2) for g in A]); gap_k200 = lab_k200 - (s_k0 + s_k1) / 2
D_k200 = np.array([float(g['D'][2]) for g in A])          # KaMIS-model D of the k200 label
typ = []                                                    # DiffUCO typicality: fraction of samples >= label size
for g in A:
    src = np.array(g['src']); ul = np.where(src == 'ul')[0]; c = g['count'].numpy()[ul]; sz = g['size'].numpy()[ul]
    typ.append((c * (sz >= size(g, 2))).sum() / c.sum())
typ = np.array(typ)
print(f'Q3 k200 model: mean gap to own label {gap_k200.mean():.3f}; Spearman(gap, D_refKaMIS(label)) {spearmanr(gap_k200, D_k200)[0]:.3f}; '
      f'Spearman(gap, -typicality) {spearmanr(gap_k200, -typ)[0]:.3f}; Spearman(gap, label size) {spearmanr(gap_k200, lab_k200)[0]:.3f}')
lab_ka = np.array([size(g, 0) for g in B]); gap_ka = lab_ka - s_ka; D_ka = np.array([float(g['D'][0]) for g in B])
print(f'   KaMIS model: mean gap {gap_ka.mean():.3f}; Spearman(gap, D_refk200(KaMIS label)) {spearmanr(gap_ka, D_ka)[0]:.3f}; Spearman(gap, label size) {spearmanr(gap_ka, lab_ka)[0]:.3f}')
# partial: within graphs of equal label size
for nm, gap, D, lab in [('k200', gap_k200, D_k200, lab_k200), ('KaMIS', gap_ka, D_ka, lab_ka)]:
    rs = [spearmanr(gap[lab == s], D[lab == s])[0] for s in np.unique(lab) if (lab == s).sum() >= 30]
    q = np.quantile(D, [0, .25, .5, .75, 1]); bins = np.digitize(D, q[1:-1])
    print(f'   {nm}: within equal label size Spearman(gap, D) mean {np.nanmean(rs):.3f}; gap by D quartile ' + ' '.join(f'{gap[bins==b].mean():.2f}' for b in range(4)))
typ_ka = []
for g in A:
    src = np.array(g['src']); ul = np.where(src == 'ul')[0]; c = g['count'].numpy()[ul]; sz = g['size'].numpy()[ul]
    typ_ka.append((c * (sz >= size(g, 0))).sum() / c.sum())
typ_ka = np.array(typ_ka)
print(f'Q3b KaMIS model: Spearman(gap, -DiffUCO typicality of KaMIS size) {spearmanr(gap_ka, -typ_ka)[0]:.3f}')
for nm, gap, t in [('k200', gap_k200, typ), ('KaMIS', gap_ka, typ_ka)]:
    bins = np.digitize(t, [0.05, 0.2, 0.5])
    print(f'   {nm}: gap by typicality <5% / 5-20% / 20-50% / >=50%: ' + ' '.join(f'{gap[bins==b].mean():.2f}(n={int((bins==b).sum())})' for b in range(4)))
# same graph: KaMIS model gap vs k200 model gap (graph hardness shared?)
print(f'Q3c Spearman(gap_k200, gap_KaMIS) {spearmanr(gap_k200, gap_ka)[0]:.3f}; sampled best k200 vs KaMIS model same graph Spearman {spearmanr((s_k0+s_k1)/2, s_ka)[0]:.3f}')
