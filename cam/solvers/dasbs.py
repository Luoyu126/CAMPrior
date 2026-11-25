"""Cross-graph discrete adjoint Schrödinger bridge sampler (DASBS, Guo et al., arXiv 2602.08243) for node problems
with a penalised energy (MIS). Route 2 of docs/20261125/plan.md, stages 2B / 2C; derivations and the conditions it
relies on: docs/20261125/math_route2.md. Local addition, UPSTREAM.md item 71.

Process (DASBS conventions, time t in [0, 1]): X_0 ~ Bernoulli(0.5)^n, target nu ∝ exp(-H / tau) at t = 1 with the
env's energy H (MIS: -|x| + beta * #violated edges, so the flip difference is (1 - 2 x_i) * grad_i). Reference: every
bit flips at rate gamma_t / 2 (utils/dasbs.UniformFlipReference; log-linear with alpha = 0 is memoryless).
Two GNNs that take the graph as input (math_route2.md §8):
  controller  log Phi_t(x)_i  (args.model, needs a time input, e.g. EncodeProcessDecodeTime2)
  corrector   log Phihat(y)_i (EncodeProcessDecode, no time), output limited to the exact range
              |log Phihat| <= log coth(Gamma(0, 1) / 2) (math_route2.md §5); zero output (ratio 1) at initialisation.
Training alternates stages of controller-AM and corrector-DM (math_route2.md §4, §5, §7): every training iteration
rolls out the current controller on a batch of graphs (num_kp chains per graph, num_tp tau-leaping steps), keeps
(x0, x1) per chain, and takes gradient steps on
  ctrl-AM: x_t ~ reference bridge(x0, x1) at t ~ U(0, 1); target log ratio -dH_i(x1)/tau - log Phihat(x1)_i
  corr-DM: target log K_{t,1}(x_t, x1^i) / K_{t,1}(x_t, x1) at t = 0 (x_t = x0) or t ~ U[0, 0.99]
with the generalised KL in log parameterisation (utils/dasbs.ratio_bregman, float64, no clipping). The phase of each
iteration follows --dasbs_stage C,R: C controller iterations, then R corrector iterations (R = 0: no corrector, the
"discrete adjoint matching without SB" baseline when the reference is memoryless).
Sampling (training rollouts and evaluation): tau-leaping on a time grid (uniform or denser near t = 1), bit i flips
with probability min(1, Gamma(t_k, t_k+1) / 2 * Phi_tk(x)_i) (DASBS Prop. B.7, math_route2.md §10); a step with
Gamma = inf (first step of the memoryless reference) resamples every bit uniformly. Evaluation decodes the last
step's per-bit probability of being 1 greedily (env.evaluate), as DiffUCO / CAM do.
"""
import math

import torch
from torch import nn

from .meta_solver import ULSolver
from utils import dasbs as D
from utils.data_utils import BatchBuffer


class DASBSNet(nn.Module):
    """Controller + corrector in one module, so that one optimizer, checkpoint and accelerator.prepare serve both.
    Only the network of the current phase receives gradients (the other one is used under no_grad or not at all), and
    AdamW skips parameters whose grad is None, so the idle network is not touched by its optimizer state."""

    def __init__(self, controller, corrector, log_bound):
        super().__init__()
        self.controller, self.corrector = controller, corrector
        self.log_bound = float(log_bound)
        head = corrector.head
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def ctrl(self, graph, x, t):
        """x, t: (N, K) -> log Phi_t(x)_i, (N, K)."""
        return self.controller(graph, x.unsqueeze(-1), t).squeeze(-1)

    def corr(self, graph, y):
        """y: (N, K) -> log Phihat(y)_i, (N, K), within [-log_bound, log_bound]."""
        raw = self.corrector(graph, y.unsqueeze(-1)).squeeze(-1)
        if self.log_bound == 0:                     # memoryless reference: the exact corrector is 1
            return torch.zeros_like(raw)
        if math.isfinite(self.log_bound):
            return self.log_bound * torch.tanh(raw / self.log_bound)
        return raw

    def forward(self, graph, x, t=None, which='ctrl'):
        """Gradient-carrying calls go through forward (so a distributed wrapper sees them)."""
        return self.ctrl(graph, x, t) if which == 'ctrl' else self.corr(graph, x)


def gamma_integral(reference, s, t):
    """Gamma(s, t) in float64 (python floats would become float32 tensors)."""
    f = lambda v: torch.tensor(v, dtype=torch.float64)
    return float(reference.integral(f(s), f(t)))


def corrector_log_bound(reference):
    """max |log Phihat| = log coth(Gamma(0, 1) / 2) = log((1 + a)/(1 - a)), a = exp(-Gamma(0, 1)); 0 if memoryless."""
    a = math.exp(-gamma_integral(reference, 0.0, 1.0))
    return math.log((1 + a) / (1 - a))


def flip_delta(problem, graph, x):
    """Exact H(x with bit i flipped) - H(x) for the env's quadratic node energy: (1 - 2 x_i) * grad_i(x)."""
    _, grad = problem.energy_func(graph, x, True)
    return (1 - 2 * x) * grad


@torch.no_grad()
def controller_target(problem, net, graph, x1, tau):
    """ctrl-AM target log ratio (math_route2.md eq. 21): -dH_i(x1)/tau - log Phihat(x1)_i, float64."""
    return -flip_delta(problem, graph, x1).double() / tau - net.corr(graph, x1).double()


def corrector_pairs(reference, x0, x1, time):
    """corr-DM inputs: (t, x_t, target log K_{t,1}(x_t, x1^i)/K_{t,1}(x_t, x1)); time 'zero' (x_t = x0) or 'uniform'."""
    if time == 'zero':
        t = torch.zeros_like(x0)
        xt = x0
    else:
        t = (torch.rand(x0.shape[1], device=x0.device) * 0.99)[None, :].expand_as(x0)
        xt = reference.sample_bridge(x0, x1, t).to(x0.dtype)
    return t, xt, reference.endpoint_flip_log_ratio(xt, x1, t, torch.ones_like(t))


@torch.no_grad()
def tau_leap(net, graph, x, reference, grid, return_last=False):
    """Run the tau-leaping sampler from x (N, K) over grid (increasing, 0 -> 1). Returns the final state, and with
    return_last also the state before the last step and the last step's flip probabilities."""
    prev, q = x, torch.zeros_like(x)
    for k in range(len(grid) - 1):
        t0, t1 = float(grid[k]), float(grid[k + 1])
        G = gamma_integral(reference, t0, t1)
        prev = x
        if math.isinf(G):
            q = torch.full_like(x, 0.5)
        else:
            t = torch.full_like(x, t0)
            q = (G / 2 * net.ctrl(graph, x, t).double().exp()).clamp(max=1.0).to(x.dtype)
        flip = torch.rand_like(x) < q
        x = torch.where(flip, 1 - x, x)
    return (x, prev, q) if return_last else x


class DASBS(ULSolver):
    def __init__(self, args):
        super(DASBS, self).__init__(args)
        assert self.solution_type == 'node', 'DASBS: node problems only'
        self.reference = D.UniformFlipReference(args.dasbs_schedule, args.dasbs_gamma, args.dasbs_alpha)
        ctrl_iters, corr_iters = (int(v) for v in args.dasbs_stage.split(','))
        assert ctrl_iters >= 1 and corr_iters >= 0
        self.stage_iters = (ctrl_iters, corr_iters)
        self.replicas = args.dasbs_replicas
        self.corrector_time = args.dasbs_corrector_time
        self.grid_train = D.time_grid(self.num_tp, args.dasbs_grid)
        self.grid_eval = D.time_grid(self.num_t, args.dasbs_grid)
        self.iteration = 0
        self.phase = 'controller'
        import models
        corrector = models.EncodeProcessDecode(n_layers=args.num_l, hidden_dim=args.num_h, out_channels=1,
                                               use_activation_checkpoint=args.use_activation_checkpoint,
                                               problem=args.problem, time_scale=self.time_scale)
        self.model = DASBSNet(self.model, corrector, corrector_log_bound(self.reference))
        if getattr(self, '_pending_state', None) is not None:     # test checkpoint, or --checkpoint to resume
            self.model.load_state_dict(self._pending_state)
            self._pending_state = None
        if args.do_train:
            self.configure_optimizer(args)        # again: the optimizer must see both networks

    def load_ckpt(self, ckpt):
        if isinstance(self.model, DASBSNet):
            self.model.load_state_dict(ckpt)
        else:                                   # called by NNSolver.__init__ before the wrapper exists
            self._pending_state = ckpt

    def full_batches_per_step(self):
        return math.ceil(self.num_kp * self.args.batch_size / self.mini_batch_size)

    @torch.no_grad()
    def generate_data(self, graph):
        C, R = self.stage_iters
        self.phase = 'controller' if self.iteration % (C + R) < C else 'corrector'
        self.iteration += 1
        net = self.accelerator.unwrap_model(self.model)
        x0 = torch.randint(0, 2, size=(graph.num_nodes, self.num_kp), device=self.accelerator.device).float()
        x1 = tau_leap(net, graph, x0.clone(), self.reference, self.grid_train)
        buffer = BatchBuffer()
        buffer.add(list(zip(x0.unbind(1), x1.unbind(1))))
        return buffer.get_dataset()

    def train_batch(self, graph, batch):
        x0, x1 = (b.T.repeat_interleave(self.replicas, 1) for b in batch)       # (N, B * replicas)
        net = self.accelerator.unwrap_model(self.model)
        if self.phase == 'controller':
            t = torch.rand(x0.shape[1], device=x0.device)[None, :].expand_as(x0)
            xt = self.reference.sample_bridge(x0, x1, t).to(x0.dtype)
            target = controller_target(self.problem, net, graph, x1, self.tau)
            return D.ratio_bregman(self.model(graph, xt, t.to(x0.dtype)), target)
        _, _, target = corrector_pairs(self.reference, x0, x1, self.corrector_time)
        return D.ratio_bregman(self.model(graph, x1, None, 'corr'), target)

    @torch.no_grad()
    def evaluate_single(self, graph):
        best = []
        net = self.accelerator.unwrap_model(self.model)
        A = torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight,
                                    torch.Size((graph.num_nodes, graph.num_nodes))).to_sparse_csr()
        graph.A = A
        for _ in range(self.num_repetition):
            x = torch.randint(0, 2, size=(graph.num_nodes, self.num_k), device=self.accelerator.device).float()
            _, prev, q = tau_leap(net, graph, x, self.reference, self.grid_eval, return_last=True)
            score = prev * (1 - q) + (1 - prev) * q          # P(bit = 1 after the last step)
            if not self.skip_decode:
                energy, _ = self.problem.evaluate(graph, score)
            else:
                energy, _ = self.problem.energy_func(graph, score)
            best.append(energy)
        return torch.cat(best).min()
