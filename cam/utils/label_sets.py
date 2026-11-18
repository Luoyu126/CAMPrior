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


def choose(sets, logp1=None, logp0=None, how='first'):
    """Pick one row of sets (M, n). logp1 / logp0: (n,) log-probabilities of 1 / 0 (for 'nearest')."""
    if how == 'first' or sets.shape[0] == 1:
        return sets[0]
    if how == 'random':
        return sets[torch.randint(sets.shape[0], (1,)).item()]
    if how == 'nearest':
        s = sets.float()
        score = s @ logp1 + (1 - s) @ logp0
        return sets[score.argmax()]
    raise ValueError(how)
