"""Several tours per TSP instance as labels (local addition, UPSTREAM.md item 61).

A tour-set file, for a DIFUSCO-format instance file (one instance per line), is
    {'points_digest': sha1 of the point parts of all lines, 'tours': [int16 (M, n)], 'lengths': [float (M,)],
     'meta': {...}}
with, per instance, M distinct tours (0-based, no repeated start node) sorted by length. Tours are compared as
undirected cycles (canonical form: start at node 0, second node smaller than the last).

As edge labels on DIFUSCO's sparse kNN graph a tour marks edge (i, j) when j is the successor or the
predecessor of i, i.e. undirected. The upstream single-tour label marks only i -> successor(i), which gives a
tour and its reversal different targets; every arm of the multi-tour experiments uses the undirected marking.
"""
import hashlib

import numpy as np
import torch


def points_digest(lines):
    h = hashlib.sha1()
    for l in lines:
        h.update(l.split(' output ')[0].strip().encode()); h.update(b'\n')
    return h.hexdigest()


def parse_points(line):
    return np.array(line.split(' output ')[0].split(), dtype=np.float64).reshape(-1, 2)


def canonical(tour):
    t = np.asarray(tour, dtype=np.int64).reshape(-1)
    if len(t) > 1 and t[0] == t[-1]:
        t = t[:-1]
    t = np.roll(t, -int(np.argmin(t)))
    if len(t) > 2 and t[1] > t[-1]:
        t = np.concatenate([t[:1], t[1:][::-1]])
    return t


def tour_length(points, tour):
    t = np.append(tour, tour[0])
    return float(np.linalg.norm(points[t[1:]] - points[t[:-1]], axis=1).sum())


def distinct_sorted(points, tours, max_keep=None):
    """Distinct canonical tours sorted by length: (int16 (M, n) tensor, float (M,) tensor)."""
    seen = {}
    for t in tours:
        c = canonical(t)
        assert sorted(c.tolist()) == list(range(len(points))), 'not a permutation'
        seen.setdefault(c.tobytes(), c)
    cs = list(seen.values())
    ls = [tour_length(points, c) for c in cs]
    order = np.argsort(ls, kind='stable')[:max_keep]
    return (torch.from_numpy(np.stack([cs[i] for i in order]).astype(np.int16)),
            torch.tensor([ls[i] for i in order], dtype=torch.float64))


def load_tour_sets(path, data_file, num_samples=None):
    """The tours of every instance of data_file (all lines must match the file the set was built for)."""
    d = torch.load(path, weights_only=False)
    lines = open(data_file).read().splitlines()
    assert len(d['tours']) == len(lines) and points_digest(lines) == d['points_digest'], \
        f'tour sets {path} do not match the points in {data_file}'
    tours = d['tours']
    return tours[:num_samples] if num_samples is not None and num_samples > 0 else tours


def edge_indicator(tours, e0, e1):
    """tours: (M, n) long; e0, e1: (E,) local node ids of the graph's edges. Returns (M, E) float with 1 where
    e1 is the successor or the predecessor of e0 in the tour."""
    M, n = tours.shape
    nxt = torch.empty_like(tours); prv = torch.empty_like(tours)
    nxt.scatter_(1, tours, tours.roll(-1, 1)); prv.scatter_(1, tours, tours.roll(1, 1))
    return ((nxt[:, e0] == e1) | (prv[:, e0] == e1)).float()
