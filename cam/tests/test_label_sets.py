"""Tests for utils/label_sets.py (UPSTREAM.md item 44)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.label_sets import choose  # noqa: E402

SETS = torch.tensor([[1, 0, 1, 0, 0],
                     [0, 1, 0, 1, 0],
                     [1, 0, 0, 0, 1]], dtype=torch.uint8)


def test_first_and_single():
    assert torch.equal(choose(SETS, how='first'), SETS[0])
    assert torch.equal(choose(SETS[1:2], how='nearest'), SETS[1])


def test_random_covers_all_rows():
    torch.manual_seed(0)
    seen = {tuple(choose(SETS, how='random').tolist()) for _ in range(200)}
    assert seen == {tuple(r) for r in SETS.tolist()}


def test_nearest_is_max_likelihood():
    p = torch.tensor([0.9, 0.1, 0.2, 0.1, 0.8])           # prediction close to row 2
    got = choose(SETS, torch.log(p), torch.log(1 - p), 'nearest')
    assert torch.equal(got, SETS[2])
    # brute force: the chosen row maximises the Bernoulli log-likelihood
    ll = [(r.float() * torch.log(p) + (1 - r.float()) * torch.log(1 - p)).sum().item() for r in SETS]
    assert ll.index(max(ll)) == 2


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('ok', name)
