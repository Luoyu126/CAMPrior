"""Single-graph DASBS training (stage 2A of docs/20261125/plan.md): reproduce the 11-15 small-graph runs of the DASBS
worktree (`sb_worktrees/DASBS/mis_sb/train_dasbs.py`) with the network-independent parts taken from utils/dasbs.py.

The network is the worktree's single-graph RopeSiT (DASBS model/sit_rope.py, commit df22917): node ids on a row-major
ceil(sqrt(n)) grid, trained per instance. It is only for this reproduction; the cross-graph version (2B) uses a GNN.
Training: `stages` alternations of controller-AM (fixed corrector) and corrector-DM (rollouts of the current
controller), checkpoint and evaluation after every stage. Same CLI and checkpoint layout as the worktree, plus
`--corrector-time uniform|zero` and `--grid uniform|quadratic` (utils/dasbs.py; defaults = worktree behaviour).

Usage (from cam/):
  python scripts/sb/dasbs_single.py --graph path6 --output DIR --seed 0 --stages 3 --controller-steps 80 \
      --corrector-steps 40 --batch-size 32 --replicas 4 --refresh 10 --sampling-steps 30 --eval-samples 1024
  python scripts/sb/dasbs_single.py --resume DIR/checkpoint.pt --output DIR2 --stages 15 --eval-samples 4096
--graph: path<n> | cycle<n> | er<n>_p<p>_s<seed> | a JSON file {num_nodes, edges, penalty}.
"""
import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from utils import dasbs as D   # noqa: E402


# ------------------------------------------------------------------ RopeSiT (worktree dasbs_model.py, unchanged)

def sinusoid(t, dim, max_period=10000):
    freq = torch.exp(-math.log(max_period) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    angle = t[..., None] * freq
    return torch.cat([angle.cos(), angle.sin()], -1)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class RotaryAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        dim = width // heads
        mag = 10.0 ** (-torch.arange(0, dim, 4).float() / dim)
        angles = torch.rand(heads, 1) * 2 * torch.pi
        fx = torch.cat([angles.cos() * mag, (angles + torch.pi / 2).cos() * mag], -1)
        fy = torch.cat([angles.sin() * mag, (angles + torch.pi / 2).sin() * mag], -1)
        self.freqs = nn.Parameter(torch.stack([fx, fy]))

    def forward(self, x, coords):
        b, n, w = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        angle = torch.einsum('nd,dhc->hnc', coords, self.freqs)
        cis = torch.polar(torch.ones_like(angle), angle)

        def rotate(z):
            zc = torch.view_as_complex(z.float().reshape(*z.shape[:-1], -1, 2))
            return torch.view_as_real(zc * cis).flatten(-2).to(z.dtype)
        out = F.scaled_dot_product_attention(rotate(q), rotate(k), v)
        return self.proj(out.transpose(1, 2).reshape(b, n, w))


class RotaryBlock(nn.Module):
    def __init__(self, width, heads, timed):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.attn = RotaryAttention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.timed = timed
        if timed:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        else:
            self.gamma1 = nn.Parameter(torch.full((width,), 1e-4))
            self.gamma2 = nn.Parameter(torch.full((width,), 1e-4))

    def forward(self, x, coords, c):
        if self.timed:
            s1, z1, g1, s2, z2, g2 = self.ada(c).chunk(6, -1)
            x = x + g1[:, None] * self.attn(modulate(self.norm1(x), s1, z1), coords)
            return x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, z2))
        x = x + self.gamma1 * self.attn(self.norm1(x), coords)
        return x + self.gamma2 * self.mlp(self.norm2(x))


class DASBSModel(nn.Module):
    """Output: log ratio for flipping each bit (controller: timed; corrector: time-free)."""

    def __init__(self, nodes, width=32, layers=6, heads=4, timed=True):
        super().__init__()
        if nodes < 1 or width % heads or (width // heads) % 4 or width % 4:
            raise ValueError('positive nodes and width/head dimensions divisible by four required')
        self.nodes, self.timed = nodes, timed
        side = math.ceil(math.sqrt(nodes))
        ids = torch.arange(nodes)
        coords = torch.stack([ids % side, ids // side], -1).float()
        self.register_buffer('coords', coords)
        freq = 10000.0 ** (-torch.arange(width // 4).float() / (width // 4))
        ax, ay = coords[:, 0, None] * freq, coords[:, 1, None] * freq
        self.position = nn.Parameter(torch.cat([ax.sin(), ax.cos(), ay.sin(), ay.cos()], -1)[None])
        self.embedding = nn.Parameter(torch.empty(2, width))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
        if timed:
            self.time_embed = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
            self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.blocks = nn.ModuleList([RotaryBlock(width, heads, timed) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(width, 2)

    def forward(self, x, time, graph=None):
        if x.ndim != 2 or x.shape[1] != self.nodes:
            raise ValueError('model is bound to a fixed node count')
        if self.timed and time is None:
            raise ValueError('controller requires time')
        h = self.embedding[x.long()] + self.position
        c = self.time_embed(sinusoid(time.reshape(-1) * 1000, 256)) if self.timed else None
        for block in self.blocks:
            h = block(h, self.coords, c)
        h = self.final_norm(h)
        if self.timed:
            shift, scale = self.final_ada(c).chunk(2, -1)
            h = modulate(h, shift, scale)
        logits = self.head(h)
        return logits.gather(-1, (1 - x.long()).unsqueeze(-1)).squeeze(-1)


# ------------------------------------------------------------------ graphs

def load_graph(spec):
    m = re.fullmatch(r'(path|cycle)(\d+)', spec)
    if m:
        return getattr(D.MISGraph, m.group(1))(int(m.group(2)))
    m = re.fullmatch(r'er(\d+)_p([0-9.]+)_s(\d+)', spec)
    if m:
        return D.MISGraph.erdos_renyi(int(m.group(1)), float(m.group(2)), int(m.group(3)))
    data = json.loads(Path(spec).read_text())
    return D.MISGraph.from_edges(data['num_nodes'], data['edges'], data.get('penalty', 2.0))


def save_graph(graph, path):
    edges = torch.nonzero(torch.triu(graph.adjacency.cpu(), 1)).tolist()
    Path(path).write_text(json.dumps({'num_nodes': graph.n, 'edges': edges, 'penalty': graph.penalty}, indent=2) + '\n')


# ------------------------------------------------------------------ training (worktree train_dasbs.py)

KEPT = ('temperature', 'width', 'layers', 'heads', 'controller_steps', 'corrector_steps', 'batch_size', 'replicas',
        'refresh', 'sampling_steps', 'lr', 'seed', 'corrector_time', 'grid', 'schedule', 'gamma', 'alpha')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--graph')
    p.add_argument('--output', required=True)
    p.add_argument('--resume')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--stages', type=int, default=5)
    p.add_argument('--controller-steps', type=int, default=400)
    p.add_argument('--corrector-steps', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--replicas', type=int, default=8)
    p.add_argument('--refresh', type=int, default=20)
    p.add_argument('--sampling-steps', type=int, default=100)
    p.add_argument('--eval-samples', type=int, default=2048)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--schedule', default='log_linear', choices=['constant', 'log_linear'])
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--corrector-time', default='uniform', choices=['uniform', 'zero'])
    p.add_argument('--grid', default='uniform', choices=['uniform', 'quadratic'])
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if checkpoint:
        old = checkpoint['config']
        for key in KEPT:
            if key in old:
                setattr(args, key, old[key])
        graph = D.MISGraph(checkpoint['adjacency'], checkpoint['penalty'])
    else:
        if not args.graph:
            p.error('--graph is required without --resume')
        graph = load_graph(args.graph)
    if not np.isfinite(args.temperature) or args.temperature <= 0 or min(
            args.batch_size, args.replicas, args.refresh, args.sampling_steps, args.eval_samples, args.stages,
            args.controller_steps, args.corrector_steps) < 1:
        p.error('positive temperature, counts and intervals required')
    if args.evaluate_only and not checkpoint:
        p.error('--evaluate-only requires --resume')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.output)
    if (out / 'checkpoint.pt').exists() and not args.resume:
        p.error('output already has a checkpoint; use --resume or a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    controller = DASBSModel(graph.n, args.width, args.layers, args.heads, True).to(args.device)
    corrector = DASBSModel(graph.n, args.width, args.layers, args.heads, False).to(args.device)
    opts = [torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0) for m in (controller, corrector)]
    start = 0
    if checkpoint:
        controller.load_state_dict(checkpoint['controller'])
        corrector.load_state_dict(checkpoint['corrector'])
        for opt, state in zip(opts, checkpoint['optimizers']):
            opt.load_state_dict(state)
        start = checkpoint['stage']
        torch.set_rng_state(checkpoint['torch_rng'])
        if torch.cuda.is_available() and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    else:
        # zero log corrector = unit ratio before the first stage (math_route2.md, corollary 7.2)
        torch.nn.init.zeros_(corrector.head.weight)
        torch.nn.init.zeros_(corrector.head.bias)
    reference = D.UniformFlipReference(args.schedule, args.gamma, args.alpha)
    exact = D.solve_bridge(graph, reference, args.temperature) if graph.n <= 10 else None
    graph = graph.to(args.device)
    config = vars(args).copy()
    config['method'] = 'DASBS'
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    save_graph(graph, out / 'graph.json')

    def evaluate(stage):
        # evaluation must not change the RNG stream used by training / resume
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(args.seed + 100000 + stage)
        chunks = []
        for k in range(0, args.eval_samples, args.batch_size):
            roll = D.sample_paths(controller, graph, reference, min(args.batch_size, args.eval_samples - k),
                                  args.sampling_steps, device=args.device, keep_path=False, grid=args.grid)
            chunks.append(roll.x1)
        metrics = D.evaluate_samples(graph, torch.cat(chunks), exact)
        metrics['stage'] = stage
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        (out / f'metrics_{stage:03}.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics), flush=True)
        return metrics

    if args.evaluate_only:
        evaluate(start)
        return
    for stage in range(start, args.stages):
        for phase, model, opt, count in [('controller', controller, opts[0], args.controller_steps),
                                         ('corrector', corrector, opts[1], args.corrector_steps)]:
            controller.eval(); corrector.eval(); model.train()
            for step in range(count):
                if step % args.refresh == 0:
                    rollout = D.sample_paths(controller, graph, reference, args.batch_size, args.sampling_steps,
                                             device=args.device, keep_path=False, grid=args.grid)
                opt.zero_grad(set_to_none=True)
                if phase == 'controller':
                    loss = D.controller_loss(controller, corrector, rollout, graph, reference, args.temperature,
                                             args.replicas)
                else:
                    loss = D.corrector_loss(corrector, rollout, graph, reference, args.replicas,
                                            time=args.corrector_time)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                opt.step()
                if step % args.refresh == 0 or step == count - 1:
                    record = {'stage': stage + 1, 'phase': phase, 'step': step, 'loss': float(loss.detach()),
                              'grad_norm': float(norm)}
                    with (out / 'train.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
                    print(json.dumps(record), flush=True)
        evaluate(stage + 1)
        save = {'version': 1, 'config': config, 'stage': stage + 1, 'adjacency': graph.adjacency.cpu(),
                'penalty': graph.penalty, 'controller': controller.state_dict(), 'corrector': corrector.state_dict(),
                'optimizers': [o.state_dict() for o in opts], 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        tmp = out / 'checkpoint.tmp'
        torch.save(save, tmp)
        tmp.replace(out / 'checkpoint.pt')


if __name__ == '__main__':
    main()
