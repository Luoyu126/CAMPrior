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


def test_tilted_choice():
    from utils.label_sets import choose
    sets = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.uint8)
    torch.manual_seed(0)
    assert all(choose(sets, how='tilted', temp=1e-3).sum() == 3 for _ in range(20))
    n = 4000
    sizes = torch.stack([choose(sets, how='tilted', temp=1e6).sum() for _ in range(n)]).float()
    assert abs(sizes.mean().item() - 2.25) < 0.05          # uniform over the four rows
    sizes = torch.stack([choose(sets, how='tilted', temp=1.0).sum() for _ in range(n)]).float()
    p = torch.tensor([1.0, 2.718281828, 2 * 7.389056099]); p = p / p.sum()
    assert abs(sizes.mean().item() - (p * torch.tensor([1., 2., 3.])).sum().item()) < 0.05


def test_posterior_mean():
    from utils.label_sets import posterior_mean
    torch.manual_seed(0)
    Qb = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64)
    xt = torch.tensor([1, 0, 1, 1, 0])
    got = posterior_mean(SETS, xt, Qb)
    lik = torch.tensor([float(torch.prod(Qb[r.long(), xt])) for r in SETS], dtype=torch.float64)   # brute force q(x_t | s_m)
    want = (lik / lik.sum()) @ SETS.double()
    assert torch.allclose(got, want, atol=1e-9)
    # no noise: the posterior puts all mass on the set equal to x_t
    assert torch.allclose(posterior_mean(SETS, SETS[1], torch.eye(2, dtype=torch.float64).clamp(min=1e-12)),
                          SETS[1].double(), atol=1e-6)
