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
Options for low temperature (stage 2C; all off by default, so 2B is unchanged):
  --dasbs_ctrl_scale tau   the controller network outputs u and log Phi = u / tau (tau of the current epoch, kept in
                           the checkpoint as the buffer ctrl_tau): the log ratios grow like 1/tau, u stays O(1). A
                           reparameterisation; the regression target and its minimiser are unchanged.
  --dasbs_loss_norm pred   the ctrl-AM loss of each (node, chain) is divided by max(1, Phi_pred) (detached). The
                           weight depends on x_t only, so the stationary point is still Phi = E[ratio | x_t]
                           (gradient in log Phi: (Phi - ratio) / max(1, Phi)); it stops the few nodes with huge
                           ratios at low tau from setting the scale of every update.
  --dasbs_skip_overflow    a controller batch whose loss or per-element gradient is not representable (exp of a
                           log ratio above ~85 reaches the float32 network as inf) is skipped (zero gradient) and
                           counted (self.skipped, logged per epoch as train/skipped) instead of stopping the run.
                           Off: such a batch raises, as in 2A / 2B. At low tau with random early rollouts on dense
                           graphs (RB-small: degree up to ~70, so -dH/tau up to ~800 at tau = 0.1) this is needed;
                           the skip fraction is reported because skipped batches bias the fit.
  --dasbs_rescale_overflow  instead of skipping: such a batch's per-element gradients (Phi - ratio) / weight are
                           computed in log space and divided by one common positive constant, so that they are
                           representable; with gradient-norm clipping (--grad_clip) the optimizer step is then exactly
                           the exact-arithmetic one (rescaled_loss). Counted as train/skipped ("rescaled" in the
                           epoch line). Reason (2C round 1): skipping removes exactly the batches with the large
                           ratios that carry the signal at low tau (E[ratio | x_t] is dominated by rare endpoints one
                           flip from a low-energy state); by epoch 60 of round 0 nearly every batch was skipped.
  --dasbs_ema d            exponential moving average of the parameters (DASBS App. C); evaluation and best.pt use
                           the average, rollouts use the current parameters; <file>_raw.pt keeps the raw weights.
"""
import copy
import json
import math
import os

import torch
from torch import nn

from .meta_solver import ULSolver
from utils import dasbs as D
from utils.data_utils import BatchBuffer


class DASBSNet(nn.Module):
    """Controller + corrector in one module, so that one optimizer, checkpoint and accelerator.prepare serve both.
    Only the network of the current phase receives gradients (the other one is used under no_grad or not at all), and
    AdamW skips parameters whose grad is None, so the idle network is not touched by its optimizer state."""

    def __init__(self, controller, corrector, log_bound, ctrl_scale=False):
        super().__init__()
        self.controller, self.corrector = controller, corrector
        self.log_bound = float(log_bound)
        self.ctrl_scale = bool(ctrl_scale)
        if self.ctrl_scale:
            self.register_buffer('ctrl_tau', torch.tensor(1.0, dtype=torch.float64))
        head = corrector.head
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def ctrl(self, graph, x, t):
        """x, t: (N, K) -> log Phi_t(x)_i, (N, K)."""
        out = self.controller(graph, x.unsqueeze(-1), t).squeeze(-1)
        return out / self.ctrl_tau.to(out.dtype) if self.ctrl_scale else out

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


def normalized_bregman(log_prediction, log_target):
    """ratio_bregman with each term divided by max(1, exp(prediction)) (detached): same stationary point, see the
    module docstring (--dasbs_loss_norm pred). float64; non-finite values raise."""
    p = log_prediction.double()
    t = log_target.detach().double()
    a, b = t.exp(), p.exp()
    value = (a * (t - p) - a + b) / b.detach().clamp(min=1.0)
    if not torch.isfinite(value).all():
        raise FloatingPointError('ratio overflow: increase temperature or inspect model logits')
    return value.mean()


GRAD_LIMIT = 1e30          # per-element gradient magnitude of the loss that the float32 backward still represents
RESCALE_TO = 1e20          # --dasbs_rescale_overflow: the largest per-element gradient after the common rescaling


def log_abs_grad(log_prediction, log_target, normalized):
    """sign and log|g| of the per-element loss gradient w.r.t. log Phi, g = (Phi - ratio) / weight (weight
    max(1, Phi) if normalized, else 1), computed in log space so that ratios beyond float64 (log > 709) are fine."""
    p = log_prediction.detach().double(); t = log_target.detach().double()
    hi, lo = torch.maximum(p, t), torch.minimum(p, t)
    log_abs = hi + torch.log1p(-torch.exp(lo - hi))          # log|e^p - e^t|; -inf where equal
    if normalized:
        log_abs = log_abs - p.clamp(min=0)
    return torch.sign(p - t), log_abs


def rescaled_loss(log_prediction, log_target, normalized, loss):
    """--dasbs_rescale_overflow. If every per-element gradient is representable (|g| <= GRAD_LIMIT), the plain loss.
    Otherwise a surrogate with value 0 whose gradient w.r.t. log Phi is c * g / numel for ONE positive constant c
    (the largest |g| becomes RESCALE_TO). Training clips the gradient norm (--grad_clip; 2C uses 1), and such a batch's
    true gradient norm is astronomically above it, so the optimizer receives grad / |grad| in both cases: the update
    is the one the exact-arithmetic algorithm makes. Nothing is dropped or clipped per element (math_route2.md §6;
    clipping the target is a bias, measured with scripts/sb/dasbs_clip_effect.py). Returns (loss, rescaled?)."""
    sign, log_abs = log_abs_grad(log_prediction, log_target, normalized)
    m = float(log_abs.max())
    if m <= math.log(GRAD_LIMIT):
        return loss(log_prediction, log_target), False
    g = sign * torch.exp(log_abs + (math.log(RESCALE_TO) - m))
    p = log_prediction.double()
    return (g * (p - p.detach())).sum() / p.numel(), True


def check_representable(log_prediction, log_target, normalized):
    """Raise FloatingPointError if the loss gradient w.r.t. log Phi of some element, (Phi - ratio) / weight, is not
    finite or above GRAD_LIMIT (it would be inf after the float32 cast in backward)."""
    with torch.no_grad():
        p = log_prediction.double(); t = log_target.double()
        g = (p.exp() - t.exp())
        if normalized:
            g = g / p.exp().clamp(min=1.0)
        if not torch.isfinite(g).all() or g.abs().max() > GRAD_LIMIT:
            raise FloatingPointError('ratio overflow')


@torch.no_grad()
def ema_update(ema, model, decay):
    """ema <- decay * ema + (1 - decay) * model for parameters; buffers copied."""
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


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
        self.ctrl_scale = getattr(args, 'dasbs_ctrl_scale', 'none') == 'tau'
        self.loss_norm = getattr(args, 'dasbs_loss_norm', 'none') == 'pred'
        self.ema_decay = float(getattr(args, 'dasbs_ema', 0.0) or 0.0)
        self.skip_overflow = bool(getattr(args, 'dasbs_skip_overflow', False))
        self.rescale_overflow = bool(getattr(args, 'dasbs_rescale_overflow', False))
        assert not (self.skip_overflow and self.rescale_overflow), 'choose one overflow treatment'
        self.skipped = 0
        self.ctrl_batches = 0
        self.model = DASBSNet(self.model, corrector, corrector_log_bound(self.reference), self.ctrl_scale)
        if getattr(self, '_pending_state', None) is not None:     # test checkpoint, or --checkpoint to resume
            self.model.load_state_dict(self._pending_state)
            self._pending_state = None
        if args.do_train:
            self.configure_optimizer(args)        # again: the optimizer must see both networks
        self.ema = None
        if args.do_train and self.ema_decay > 0:
            self.ema = copy.deepcopy(self.accelerator.unwrap_model(self.model)).eval()
            self.ema_device = next(self.ema.parameters()).device
            for q in self.ema.parameters():
                q.requires_grad_(False)

    def log(self, logger):
        if 'train/loss' in logger:          # per-epoch training record: cumulative overflow skips (--dasbs_skip_overflow)
            logger = dict(logger, **{'train/skipped': self.skipped, 'train/ctrl_batches': self.ctrl_batches,
                                     'train/tau': float(self.tau)})
            self.accelerator.print(f"epoch {logger.get('Epoch')}: tau {float(self.tau):.4g}, "
                                   f"{'rescaled' if self.rescale_overflow else 'skipped'} "
                                   f"{self.skipped}/{self.ctrl_batches} controller batches")
        return super().log(logger)

    def eval_net(self):
        return self.ema if self.ema is not None else self.accelerator.unwrap_model(self.model)

    def save_ckpt(self, save_file):
        if not self.accelerator.is_local_main_process:
            return
        net = self.accelerator.unwrap_model(self.model)
        if self.ema is None:
            torch.save(net.state_dict(), save_file)
            return
        torch.save(self.ema.state_dict(), save_file)
        torch.save(net.state_dict(), save_file[:-3] + '_raw.pt' if save_file.endswith('.pt') else save_file + '_raw')

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
        if self.ctrl_scale:
            net.ctrl_tau.fill_(float(self.tau))
            if self.ema is not None:
                self.ema.ctrl_tau.fill_(float(self.tau))
        x0 = torch.randint(0, 2, size=(graph.num_nodes, self.num_kp), device=self.accelerator.device).float()
        x1 = tau_leap(net, graph, x0.clone(), self.reference, self.grid_train)
        buffer = BatchBuffer()
        buffer.add(list(zip(x0.unbind(1), x1.unbind(1))))
        return buffer.get_dataset()

    def train_batch(self, graph, batch):
        x0, x1 = (b.T.repeat_interleave(self.replicas, 1) for b in batch)       # (N, B * replicas)
        net = self.accelerator.unwrap_model(self.model)
        if self.ema is not None:            # the previous optimizer step's parameters enter the average here
            if self.ema_device != x0.device:
                self.ema.to(x0.device); self.ema_device = x0.device
            ema_update(self.ema, net, self.ema_decay)
        if self.phase == 'controller':
            t = torch.rand(x0.shape[1], device=x0.device)[None, :].expand_as(x0)
            xt = self.reference.sample_bridge(x0, x1, t).to(x0.dtype)
            target = controller_target(self.problem, net, graph, x1, self.tau)
            loss = normalized_bregman if self.loss_norm else D.ratio_bregman
            pred = self.model(graph, xt, t.to(x0.dtype))
            self.ctrl_batches += 1
            if self.rescale_overflow:
                value, rescaled = rescaled_loss(pred, target, self.loss_norm, loss)
                self.skipped += int(rescaled)          # counted as "rescaled" (logged under the same key)
                return value
            if self.skip_overflow:
                try:
                    check_representable(pred, target, self.loss_norm)
                    return loss(pred, target)
                except FloatingPointError:
                    self.skipped += 1
                    return (pred * 0).sum()
            return loss(pred, target)
        _, _, target = corrector_pairs(self.reference, x0, x1, self.corrector_time)
        return D.ratio_bregman(self.model(graph, x1, None, 'corr'), target)

    @torch.no_grad()
    def evaluate_single(self, graph):
        best = []
        net = self.eval_net()
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
            if os.environ.get('CAM_DASBS_RAW'):
                self.write_raw(graph, torch.bernoulli(score), energy)
        return torch.cat(best).min()

    def write_raw(self, graph, x, energy):
        """Diagnostic (CAM_DASBS_RAW=FILE, stage 2C): the sampler's own end states x (n, K) before decoding. One JSON
        line per graph: share of chains without a violated edge, mean size and violated edges of the raw states,
        mean size of the feasible ones, and mean / best size after greedy decoding."""
        e0, e1 = graph.edge_index
        viol = (x[e0] * x[e1]).sum(0) / 2                      # undirected edges stored twice
        size = x.sum(0); ok = viol == 0
        rec = dict(idx=int(torch.as_tensor(graph.idx).reshape(-1)[0]), n=int(x.shape[0]), chains=int(x.shape[1]),
                   raw_feasible=float(ok.float().mean()), raw_size=float(size.mean()), raw_violations=float(viol.mean()),
                   raw_size_feasible=float(size[ok].mean()) if ok.any() else None,
                   decoded_mean=float((-energy).float().mean()), decoded_best=float((-energy).max()))
        with open(os.environ['CAM_DASBS_RAW'], 'a') as f:
            f.write(json.dumps(rec) + '\n')
