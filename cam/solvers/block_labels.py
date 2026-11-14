"""Block-move labels for CAMPrior (--label_mode block; local addition, see cam/UPSTREAM.md).

Why (docs/status.md 2.11, 11-14): on maximum clique the simulated gain of a single flip is small
next to the rollout noise, because RLSA repairs most single changes; a block move found by
screening many candidates with rollouts has a real (if small) gain. This is the rollout algorithm
of Bertsekas / Tesauro & Galperin (1996) with RLSA as the base policy: screen candidate moves by
Monte-Carlo rollouts of the base policy, keep the best one, and let the network learn it.

Conflict matrix A (A_ij > 0: i and j cannot both be chosen): the MIS graph, or the complement graph
for maximum clique (sim_labels.complement_adjacency). For a state x (0/1) and an unselected seed i,
the block move B(x, i) is
    1. x_i = 1;  2. x_j = 0 for every j in conflict with i;
    3. repeat: candidates = unselected nodes in conflict with no selected node; add the candidate
       with the most non-conflicting candidates (itself included; ties: lowest index); until none.

Label of a stored state X_t (step t; one item):
    seeds     up to S unselected nodes with at most max_conf conflicts with the selected set
              (sampled with the item's generator); none: the item's labels are all 0
    screen    each block move F_s = B(X_t, i_s) against the factual RLSA step, with R_screen pairs:
                  move:     X_t -> F_s at step t, then the RLSA prior over steps t+1 .. K-1
                  factual:  X_t -> the RLSA prior over steps t .. K-1
              both sharing the uniforms of steps t+1 .. K-1 (the factual one also uses those of
              step t); gain = size(decoded end of move) - size(decoded end of factual), decoded
              as the validation objective (sim_labels.greedy_decode_size)
    evaluate  the block with the best screened mean gain, with R_eval fresh pairs: G
    label     G on the nodes the block flips (F != X_t), 0 on the others
so the expected label of a node is the gain of the best block it takes part in, measured without
the screening's selection bias (a positive label favours flipping, as CAM's target). With
--prior_recenter on the sampler keeps flipping about d nodes per step, and the label ranks the
block's nodes into the top d.

Random numbers: one generator per item on the compute device, seeded with the item's seed (seed
choice, then the uniforms of the screening, then those of the evaluation), so labels do not depend
on how items are grouped into tasks (CPU and GPU give different numbers).
"""
import torch

from .sim_labels import greedy_decode_size


def block_moves(conf, x, seeds):
    """Block moves of state x (n,) from each seed (S,); conf: n x n bool. Returns n x S (0/1 float)."""
    S = len(seeds)
    cols = torch.arange(S, device=x.device)
    X = x.unsqueeze(1).repeat(1, S).bool()
    X &= ~conf[seeds].T                                          # drop the seed's conflicts
    X[seeds, cols] = True
    free = (~conf).float()
    confl = conf.float()
    while True:
        cand = ~X & ((confl @ X.float()) == 0)                   # in conflict with no selected node
        live = cand.any(0)
        if not live.any():
            break
        score = torch.where(cand, free @ cand.float(), -torch.ones(1, device=x.device))
        best = torch.argmax(score, 0)
        X[best[live], cols[live]] = True
    return X.float()


def rlsa_step(A, b, X, j, temps, d, eps, beta_prior):
    """Flip probabilities of RLSA's step j (the prior of CAMPrior) for the columns of X."""
    grad = 2*beta_prior*(A@X) + b
    delta = grad*(2*X - 1)/2
    thr = -torch.kthvalue(-delta, min(int(d), X.shape[0]), dim=0, keepdim=True).values
    diff = delta - thr
    if eps != 0:
        diff = diff + eps
    return torch.sigmoid(diff/temps[j])


@torch.no_grad()
def move_gains(A, b, x, moves, t, R, gen, temps, d, eps, beta_prior):
    """(M, R) gains size(move) - size(factual) of M moves (n x M) from state x (n,) at step t, R pairs
    each (see the module docstring); uniforms from gen, (M*R) x n per step from step t on."""
    n, M = moves.shape
    K = len(temps)
    Xf = x.unsqueeze(1).repeat(1, M*R)
    Xm = moves.repeat_interleave(R, dim=1)
    b = b.reshape(n, 1)
    for j in range(t, K):
        u = torch.rand(M*R, n, generator=gen, device=x.device).T
        Xf = torch.where(u < rlsa_step(A, b, Xf, j, temps, d, eps, beta_prior), 1 - Xf, Xf)
        if j > t:
            Xm = torch.where(u < rlsa_step(A, b, Xm, j, temps, d, eps, beta_prior), 1 - Xm, Xm)
    sizes = greedy_decode_size(A, torch.cat([Xm, Xf], 1))
    return (sizes[:M*R] - sizes[M*R:]).reshape(M, R)


@torch.no_grad()
def block_labels(A, b, x_t, starts, seeds, temps, d, eps, beta_prior, S=32, R_screen=64, R_eval=128,
                 max_conf=2, stats=None):
    """Labels (C, n) of C stored states x_t (n x C, 0/1) at steps starts (C,), on A's device.

    A: n x n conflict matrix (dense), b: (n, 1) linear term; temps: the K prior temperatures
    tau_rlsa(j) * ratio. stats (list, optional): one dict per item (gain, block size, seeds)."""
    dev = A.device
    n, C = x_t.shape
    conf = A > 0
    b = torch.as_tensor(b, dtype=torch.float32, device=dev).reshape(n, 1)
    X = torch.as_tensor(x_t, dtype=torch.float32, device=dev)
    out = torch.zeros(C, n, device=dev)
    for c in range(C):
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seeds[c]))
        x, t = X[:, c], int(starts[c])
        cand = torch.nonzero((x == 0) & ((conf.float() @ x) <= max_conf)).reshape(-1)
        if len(cand) == 0 or t >= len(temps):
            if stats is not None:
                stats.append(dict(gain=0.0, size=0, seeds=0))
            continue
        pick = cand[torch.randperm(len(cand), generator=gen, device=dev)[:S]]
        moves = block_moves(conf, x, pick)
        screen = move_gains(A, b, x, moves, t, R_screen, gen, temps, d, eps, beta_prior).mean(1)
        F = moves[:, int(torch.argmax(screen))]
        G = move_gains(A, b, x, F.unsqueeze(1), t, R_eval, gen, temps, d, eps, beta_prior).mean()
        flip = (F != x).float()
        out[c] = G*flip
        if stats is not None:
            stats.append(dict(gain=float(G), size=int(flip.sum()), seeds=len(pick)))
    return out
