"""Per-graph weights of the DIFUSCO training loss (local addition, UPSTREAM.md item 45).

The cross-entropy of a batch becomes sum_i w_g(i) ce_i / sum_i w_g(i) over its nodes i, where g(i) is
the graph of node i. With all weights equal this is the plain node mean (the upstream loss).

    nsets  static: w_g = M_g^(-power), M_g the number of distinct optimal sets of graph g found by
           scripts/mis/build_solution_sets.py. Graphs with few optima (where SL trails UL most) count more.
    ce     adaptive, after the hardness-weighted curriculum of Zhang et al. (AAAI 2022, eq. 12):
           w_g = exp(z_g), z_g the standardized log hardness of graph g, clipped to [-3, 3]. Hardness is
           the graph's cross-entropy divided by the running mean cross-entropy at the same noise level
           (t in 20 bins), so a graph that drew a large t is not counted as hard; it is averaged over
           the graph's visits (EMA). Standardizing sets the temperature of their softmax to the spread
           of log hardness, so the weights' effective sample size stays near exp(-1) of the data. Graphs
           not yet visited get weight 1. Weights come from earlier visits (one epoch old), the
           curriculum paper uses the current model.
"""
import torch


class GraphWeights:
    def __init__(self, mode, num_graphs, nsets=None, power=0.5, ema=0.5, bins=20, num_tp=1000, bin_ema=0.99):
        assert mode in ('nsets', 'ce'), mode
        self.mode, self.num_tp, self.bins, self.ema, self.bin_ema = mode, num_tp, bins, ema, bin_ema
        self.num_graphs = num_graphs
        if mode == 'nsets':
            m = torch.as_tensor(nsets, dtype=torch.float)
            assert m.numel() == num_graphs and (m >= 1).all()
            w = m.pow(-power)
            self.static = w / w.mean()
        else:
            self.logh = torch.full((num_graphs,), float('nan'))
            self.base = torch.full((bins,), float('nan'))

    def weights(self, idx):
        """Weights of the graphs idx (long tensor, dataset indices), on CPU."""
        idx = idx.cpu()
        if self.mode == 'nsets':
            return self.static[idx]
        seen = ~torch.isnan(self.logh)
        if seen.sum() < 2:
            return torch.ones(len(idx))
        mu, sd = self.logh[seen].mean(), self.logh[seen].std().clamp_min(1e-6)
        z = ((self.logh[idx] - mu) / sd).nan_to_num(0.0).clamp(-3, 3)
        return z.exp()

    def update(self, idx, t, ce):
        """Record per-graph cross-entropy ce at noise level t (1..num_tp) for the graphs idx ('ce' only)."""
        if self.mode != 'ce':
            return
        idx, t, ce = idx.cpu().long(), t.cpu().float(), ce.detach().cpu().float()
        b = ((t - 1) * self.bins / self.num_tp).long().clamp(0, self.bins - 1)
        for k in b.unique():
            m = ce[b == k].mean()
            self.base[k] = m if torch.isnan(self.base[k]) else self.bin_ema * self.base[k] + (1 - self.bin_ema) * m
        score = torch.log(ce.clamp_min(1e-8) / self.base[b].clamp_min(1e-8))
        old = self.logh[idx]
        self.logh[idx] = torch.where(torch.isnan(old), score, self.ema * old + (1 - self.ema) * score)


def effective_sample_size(w):
    w = torch.as_tensor(w, dtype=torch.float)
    return (w.sum() ** 2 / (w ** 2).sum()).item()

