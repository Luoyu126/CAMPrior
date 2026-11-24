"""Minimum dominating set (local addition, see cam/UPSTREAM.md).

Energy of DiffUCO (Sanokowski et al., 2024, Table 2; A = 1, B = 1.01 as in all their experiments):
    E(x) = A * sum_i x_i + B * sum_i (1 - x_i) prod_{j in N(i)} (1 - x_j),
i.e. set size plus B per node that is neither selected nor next to a selected node. With A < B every
local minimum under single flips is a minimal dominating set. E is of high degree but multilinear
(each x_i appears at most once per monomial), so the energy change of flipping i is exactly
(1 - 2 x_i) * dE/dx_i, as for the quadratic problems: RLSA and CAM's labels need no approximation.

On 0/1 states (sampling rollouts, decoding, RLSA) the exact counts below are used; on probabilities (the DiffUCO
and CAM losses) the multilinear extension (_multilinear). With c_j = x_j + sum_{k in N(j)} x_k the
cover count of j, node j is uncovered iff c_j = 0, and at a 0/1 point
    dE/dx_i = A - B * ( [c_i - x_i = 0] + sum_{j in N(i)} [c_j - x_i = 0] ),
the number of nodes that i alone would cover (x_i = 0) or alone covers (x_i = 1). Two neighbour sums
(index_add over edge_index) per call, against one sparse matvec for MIS.
"""
import torch
from utils.graph_ops import g_sum


class Env(object):
    def __init__(self, args):
        self.A = float(getattr(args, 'mds_a', 1.0))
        self.B = float(getattr(args, 'mds_b', 1.01))
        self.solution_type = 'node'
        self.mis_decoding = getattr(args, 'mis_decoding', 'continuous')

    @staticmethod
    def nbr_sum(graph, v):
        """sum over the neighbours of each node (edge_index holds both directions), v: num_nodes x K"""
        src, dst = graph.edge_index
        return torch.zeros_like(v).index_add_(0, dst, v[src])

    @staticmethod
    def _batch(graph, device):
        b = getattr(graph, 'batch', None)
        return b if b is not None else torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

    def energy_func(self, graph, solution, compute_grad=False):
        x = solution
        if x.requires_grad or not ((x == 0) | (x == 1)).all():
            return self._multilinear(graph, x, compute_grad)
        nb = self.nbr_sum(graph, x)
        c = x + nb
        unc = (c < 0.5).to(x.dtype)
        energy = g_sum(self.A*x + self.B*unc, self._batch(graph, x.device))
        grad = None
        if compute_grad:
            z0 = unc
            z1 = ((c > 0.5) & (c < 1.5)).to(x.dtype)
            own = (nb < 0.5).to(x.dtype)                      # [c_i - x_i = 0]
            alone = torch.where(x > 0.5, self.nbr_sum(graph, z1), self.nbr_sum(graph, z0))
            grad = self.A - self.B*(own + alone)
        return energy, grad

    def _multilinear(self, graph, p, compute_grad):
        """E at probabilities p in [0, 1] (DiffUCO and CAM evaluate the energy at the flip-chain probabilities and
        differentiate through it): the multilinear extension A sum p_i + B sum_i (1 - p_i) prod_{j in N(i)} (1 - p_j),
        which equals E on 0/1 points. The products are exp of neighbour sums of log(1 - p) (1e-12 added where 1 - p = 0;
        a clamp would zero the autograd gradient there), so a fully covered node contributes ~1e-12 instead of 0.
        dE/dp_i = A - B (u_i + sum_{j in N(i)} (1 - p_j) u_j / (1 - p_i)), u_i = prod_{j in N(i)} (1 - p_j),
        which reduces to the 0/1 formula of energy_func."""
        q = 1 - p
        lq = torch.log(q + (q < 1e-12).to(q.dtype).detach()*1e-12)   # offset, not clamp: keeps d/dp exact at p = 1
        L = self.nbr_sum(graph, lq)
        u = torch.exp(L)
        energy = g_sum(self.A*p + self.B*(1 - p)*u, self._batch(graph, p.device))
        grad = None
        if compute_grad:
            src, dst = graph.edge_index
            w = torch.exp(lq[src] + L[src] - lq[dst])          # (1 - p_j) prod_{k in N(j), k != i} (1 - p_k)
            grad = self.A - self.B*(u + torch.zeros_like(p).index_add_(0, dst, w))
        return energy, grad

    def _descend(self, graph, solution):
        """single flips with the largest energy decrease until none decreases it (as envs/mis.py);
        adds nodes that cover uncovered ones, removes redundant ones; ends at a minimal dominating set"""
        batch_idx = torch.arange(solution.shape[1], device=solution.device).long()
        energy, grad = self.energy_func(graph, solution, True)
        delta = grad*(2*solution - 1)/2
        while not (delta <= 1e-9).all():
            value, index = torch.max(delta, dim=0)
            solution[index, batch_idx] = torch.where(value > 1e-9, 1 - solution[index, batch_idx],
                                                     solution[index, batch_idx])
            energy, grad = self.energy_func(graph, solution, True)
            delta = grad*(2*solution - 1)/2
        return energy, solution

    def evaluate(self, graph, solution):
        """decode to a minimal dominating set; returns (set size, solution) with the layouts of
        envs/mis.py (continuous: per-sample total over the batch and N x K; discrete: as energy_func
        and K x N). Lower is better."""
        if self.mis_decoding == 'continuous':
            # greedy by probability: add the most likely node that covers an uncovered node, until
            # every node is covered; then drop redundant nodes (largest energy decrease first)
            x = solution
            out = torch.zeros_like(x)
            cols = torch.arange(x.shape[1], device=x.device)
            while True:
                c = out + self.nbr_sum(graph, out)
                unc = (c < 0.5).to(x.dtype)
                gain = unc + self.nbr_sum(graph, unc)                       # uncovered nodes in N[i]
                cand = (gain > 0.5) & (out < 0.5)
                if not cand.any():
                    break
                prob = torch.where(cand, x, torch.full_like(x, -torch.inf))
                nxt = prob.argmax(0)
                live = cand[nxt, cols]
                out[nxt[live], cols[live]] = 1
            _, out = self._descend(graph, out)
            return out.sum(0), out
        solution = torch.round(solution).squeeze(-1)
        energy, solution = self._descend(graph, solution)
        return energy, solution.T
