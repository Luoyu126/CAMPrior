"""Training with several optimal labels per graph (local addition, UPSTREAM.md item 44).

A label set file (scripts/mis/build_solution_sets.py) holds, per training graph, a uint8 tensor
(M, num_nodes) of distinct maximum independent sets. Each training step picks one of them as the
label of that graph:
    first    the first set (a precomputed choice, e.g. the original label or a rule-based one)
    random   a uniformly random set
    nearest  the set the current model already finds most likely: score_m = sum_i log p(x_i = s_mi)
             under the model's prediction from pure noise (t = T), then argmax over m. A hindsight /
             hard-EM choice over labels (Li, Chen & Koltun 2018, sec. 4.2, eq. 3 take the min over
             outputs instead), which lets the model settle on choices it can learn consistently.
    tilted   a random set drawn with probability proportional to exp(|s_m| / temp) (--label_temp), the
             size-tilted distribution of the stored sets. With sampled solutions stored with repeats
             (e.g. all samples of an unsupervised solver) this is energy-weighted distillation of that
             solver's distribution (Neural Diving, Nair et al. 2020, sec. 6.1, eq. 9-11); temp -> inf is
             'random', temp -> 0 picks the largest.
    soft     (training only, solvers/difusco.py) draw a set uniformly, noise it to x_t, and regress on the
             posterior mean over the whole set, E[x_0 | x_t] = sum_m w_m s_m with w_m proportional to
             q(x_t | s_m) (posterior_mean below). Same minimiser as 'random' (the Bayes x_0-predictor under a
             uniform prior on the set), lower-variance targets: the per-graph exact version of the Stable
             Target Field (Xu, Tong & Jaakkola 2023, arXiv 2302.00670, sec. 3, reference-batch weights).
             Local addition, UPSTREAM.md item 56.
"""
import glob
import os

import torch


def load_label_sets(path, data_dir, num_samples=None):
    sets = torch.load(path, weights_only=False)
    files = [os.path.basename(f) for f in sorted(glob.glob(data_dir + '/*gpickle'))]
    if num_samples is not None and num_samples > 0:
        files = files[:num_samples]
    assert sets['files'][:len(files)] == files, f'label sets {path} do not match the graphs in {data_dir}'
    return sets['sets']


def choose(sets, logp1=None, logp0=None, how='first', temp=1.0):
    """Pick one row of sets (M, n). logp1 / logp0: (n,) log-probabilities of 1 / 0 (for 'nearest')."""
    if how == 'first' or sets.shape[0] == 1:
        return sets[0]
    if how == 'tilted':
        size = sets.float().sum(1)
        w = torch.softmax((size - size.max()) / temp, 0)
        return sets[torch.multinomial(w, 1).item()]
    if how == 'random':
        return sets[torch.randint(sets.shape[0], (1,)).item()]
    if how == 'nearest':
        s = sets.float()
        score = s @ logp1 + (1 - s) @ logp0
        return sets[score.argmax()]
    raise ValueError(how)


def posterior_mean(sets, xt, Qbar_t):
    """E[x_0 | x_t] under a uniform prior on the rows of sets (M, n) and the per-node forward kernel.
    xt: (n,) in {0, 1}; Qbar_t: (2, 2) tensor, Qbar_t[a, b] = q(x_t = b | x_0 = a). Returns (n,) in [0, 1]."""
    s = sets.to(Qbar_t.dtype); x = xt.to(Qbar_t.dtype).view(-1)
    l1 = torch.log(Qbar_t[1, 1]) * x + torch.log(Qbar_t[1, 0]) * (1 - x)   # log q(x_t,i | x_0,i = 1)
    l0 = torch.log(Qbar_t[0, 1]) * x + torch.log(Qbar_t[0, 0]) * (1 - x)
    w = torch.softmax(s @ (l1 - l0), 0)                                     # the sum of l0 cancels
    return w @ s.to(w.dtype)
