"""Tests for utils/graph_weights.py (UPSTREAM.md item 45)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.graph_weights import GraphWeights, effective_sample_size  # noqa: E402


def test_nsets_static():
    gw = GraphWeights('nsets', 4, nsets=[1, 4, 16, 64])
    w = gw.weights(torch.arange(4))
    assert torch.allclose(w / w[0], torch.tensor([1, .5, .25, .125]))
    assert torch.isclose(gw.static.mean(), torch.tensor(1.0))


def test_ce_unvisited_is_uniform():
    gw = GraphWeights('ce', 5, nsets=[1] * 5)
    assert torch.equal(gw.weights(torch.arange(5)), torch.ones(5))


def test_ce_hard_graph_counts_more_and_large_t_is_not_hard():
    gw = GraphWeights('ce', 4, nsets=[1] * 4)
    # graphs 0,1 at small t, graphs 2,3 at large t; within each t, graph 0 / 2 has twice the CE of 1 / 3
    idx, t = torch.arange(4), torch.tensor([10., 10., 990., 990.])
    ce = torch.tensor([0.02, 0.01, 0.6, 0.3])
    gw.update(idx, t, ce)
    w = gw.weights(idx)
    assert w[0] > w[1] and w[2] > w[3]
    assert torch.isclose(w[0], w[2]) and torch.isclose(w[1], w[3])   # the noise level itself does not count


def test_equal_weights_give_node_mean():
    ce = torch.rand(10); batch = torch.tensor([0] * 3 + [1] * 7)
    w = torch.ones(2)[batch]
    assert torch.isclose((w * ce).sum() / w.sum(), ce.mean())


def test_ess_rb_like():
    w = torch.tensor([1.0] * 50 + [0.1] * 50)
    assert abs(effective_sample_size(w) - 55 ** 2 / 50.5) < 1e-3


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('ok', name)
