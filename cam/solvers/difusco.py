from .meta_solver import SLSolver
from utils.diffusion_schedulers import *
from utils.heated_forward import HeatedForward
from utils.label_sets import load_label_sets, choose, posterior_mean
from utils.tour_sets import load_tour_sets, edge_indicator
from utils.graph_weights import GraphWeights, effective_sample_size
from utils import sb_coupling
import glob
import json
import os
import torch
from torch import nn
import torch.nn.functional as F

class DIFUSCO(SLSolver):
    def __init__(self, args):
        super(DIFUSCO, self).__init__(args)            
        self.diffusion = CategoricalDiffusion(T=args.num_tp, schedule=args.diffusion_schedule)
        self.inference_schedule = DiffusionInferenceSchedule(inference_schedule=args.inference_schedule,
                                              T=args.num_tp, inference_T=args.num_t,
                                              t_start=args.inference_t_start)
        self.loss_func = nn.CrossEntropyLoss()
        # local addition (see UPSTREAM.md item 42): energy-heated forward and re-noising sampler
        self.sampler = getattr(args, 'difusco_sampler', 'posterior')
        self.decode_two_opt = getattr(args, 'decode_two_opt', 0)
        self.heated = HeatedForward(self.diffusion.beta, getattr(args, 'heat_lambda', 0.0),
                                    getattr(args, 'heat_until', 0.5), args.beta)
        # local addition (UPSTREAM.md item 44): several optimal labels per training graph
        self.label_sets, self.label_choice = None, getattr(args, 'label_choice', 'first')
        self.label_temp = getattr(args, 'label_temp', 1.0)
        if getattr(args, 'label_set', None) and args.do_train:
            if self.solution_type == 'edge':   # UPSTREAM.md item 61: several tours per TSP instance
                assert self.label_choice in ('random', 'soft', 'first'), 'tour sets: first / random / soft only'
                self.label_sets = load_tour_sets(args.label_set, args.train_path, args.train_samples)
            else:
                self.label_sets = load_label_sets(args.label_set, args.train_path, args.train_samples)
        # local addition (UPSTREAM.md item 45): per-graph weights of the training loss
        self.graph_weights = None
        mode = getattr(args, 'graph_weight', 'none')
        if mode != 'none' and args.do_train:
            sets = load_label_sets(args.graph_weight_sets, args.train_path, args.train_samples)
            self.graph_weights = GraphWeights(mode, len(sets), nsets=[s.shape[0] for s in sets],
                                              power=args.graph_weight_power, num_tp=args.num_tp)
        # local addition (UPSTREAM.md item 69): Schrödinger-bridge coupling between Bernoulli(0.5)^n and the label
        # set (docs/20261125/math_route1.md). The reference keeps a signal a_T = tanh(omega / 2) at the noise end;
        # the existing samplers run on its Q_bar (posterior = sampler A; renoise* bridge back to the chain's start
        # z = sampler B). Off by default; --sb_omega 0 reproduces the existing schedule and independent coupling.
        self.sb = getattr(args, 'sb_coupling', False)
        if self.sb:
            assert self.solution_type == 'node', 'SB coupling: node problems only'
            assert self.heated.heat_lambda == 0, 'SB coupling needs --heat_lambda 0'
            assert self.sampler in ('posterior', 'renoise', 'renoise_decode'), self.sampler
            self.sb_omega = float(args.sb_omega)
            self.sb_a = sb_coupling.schedule_for_omega(self.diffusion.Q_bar, self.sb_omega)
            self.diffusion.Q_bar = sb_coupling.signal_to_Q_bar(self.sb_a, as_numpy=True)
            self.sb_z = None
            self.sb_log_g = None
            if args.do_train:
                assert self.label_sets is not None and self.label_choice in ('random', 'soft')
                assert self.graph_weights is None
                if self.sb_omega > 0:
                    d = torch.load(args.sb_logg, weights_only=False)
                    assert abs(d['omega'] - self.sb_omega) < 1e-12 and d['num_tp'] == args.num_tp, args.sb_logg
                    assert abs(d['a_end'] - self.sb_a[-1].item()) < 1e-12
                    files = [os.path.basename(f) for f in sorted(glob.glob(args.train_path + '/*gpickle'))]
                    files = files[:len(self.label_sets)]
                    assert d['files'][:len(files)] == files, 'log g file does not match the graphs'
                    self.sb_log_g = d['log_g']
                    assert len(self.sb_log_g) >= len(self.label_sets)
                    for g_, s_ in zip(self.sb_log_g, self.label_sets):
                        assert g_.shape[0] == s_.shape[0]
                elif args.sb_logg:   # independent coupling with a weighted pi1 (stage 1E): log g = log pi1 from the file
                    d = torch.load(args.sb_logg, weights_only=False)
                    assert d['omega'] == 0 and d['label_set'] == args.label_set, args.sb_logg
                    files = [os.path.basename(f) for f in sorted(glob.glob(args.train_path + '/*gpickle'))]
                    assert d['files'][:len(self.label_sets)] == files[:len(self.label_sets)], 'log g file does not match'
                    self.sb_log_g = d['log_g']
                    for g_, s_ in zip(self.sb_log_g, self.label_sets):
                        assert g_.shape[0] == s_.shape[0]
                else:   # independent coupling (up to the ~1e-6 signal the cosine table keeps at T): g = pi1, uniform
                    self.sb_log_g = [torch.zeros(s_.shape[0], dtype=torch.float64) for s_ in self.label_sets]

    def edge_slices(self, graph):
        """Per graph of the batch: (dataset index, edge slice, local (e0, e1)); edges are stored graph by graph."""
        idx = torch.as_tensor(graph.idx).reshape(-1).tolist(); ptr = graph.ptr
        cnt = torch.bincount(graph.batch[graph.edge_index[0]], minlength=len(idx)).tolist()
        out, a = [], 0
        for g, i in enumerate(idx):
            e = graph.edge_index[:, a:a + cnt[g]] - ptr[g]
            out.append((int(i), slice(a, a + cnt[g]), e[0], e[1])); a += cnt[g]
        return out

    def tour_labels(self, i, e0, e1):
        """(M, E_g) undirected edge labels of the tours of training instance i (UPSTREAM.md item 61)."""
        return edge_indicator(self.label_sets[i].to(e0.device).long(), e0, e1)

    def pick_labels(self, graph):
        """One label per graph of the batch from its label set (UPSTREAM.md item 44), shape like graph.x."""
        if self.solution_type == 'edge':
            how = 'random' if self.label_choice == 'soft' else self.label_choice
            return torch.cat([choose(self.tour_labels(i, e0, e1), how=how)
                              for i, _, e0, e1 in self.edge_slices(graph)]).view(-1, 1)
        dev = graph.x.device
        idx = torch.as_tensor(graph.idx).reshape(-1).tolist()
        ptr = graph.ptr.tolist()
        logp = None
        if self.label_choice == 'nearest':
            with torch.no_grad():
                x_T = torch.bernoulli(torch.full_like(graph.x.float(), 0.5))
                t_T = torch.full((graph.num_nodes,), float(self.num_tp), device=dev)
                logp = torch.log_softmax(self.model(graph, x_T, t_T).float(), -1)
        parts = []
        for g, i in enumerate(idx):
            s = self.label_sets[int(i)].to(dev)
            assert s.shape[1] == ptr[g + 1] - ptr[g], 'label set does not match the graph size'
            lp1 = logp[ptr[g]:ptr[g + 1], 1] if logp is not None else None
            lp0 = logp[ptr[g]:ptr[g + 1], 0] if logp is not None else None
            how = 'random' if self.label_choice == 'soft' else self.label_choice
            parts.append(choose(s, lp1, lp0, how, self.label_temp))
        return torch.cat(parts).view_as(graph.x).to(graph.x.dtype)
    
    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """Sample from the categorical posterior for a given time step.
           See https://arxiv.org/pdf/2107.03006.pdf for details.
        """
        diffusion = self.diffusion
        
        if target_t is None:
            target_t = t - 1
        
        Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
        Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device) # 2 x 2

        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device) # 2 x 2
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device) # 2 x 2

        xt = F.one_hot(xt.long(), num_classes=2).float() # |V| x B x x 2
        xt = xt.reshape(x0_pred_prob.shape) # |V| x B x  2 
        
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous()) # |V| x B x 2
        x_t_target_prob_part_2 = Q_bar_t_target[0] # 2
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True) # |V| x B x 1

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3 # |V| x B x 2

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0] # |V| x B 
        x_t_target_prob_part_2_new = Q_bar_t_target[1] # 2
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True) # |V| x B 
        
        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new # |V| x B x 2
        
        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1] # |V| x B 

        if target_t > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)
        
        return xt
    
    def tour2adj(self, tour, points, sparse, sparse_factor, edge_index):
        adj_matrix = np.zeros(points.shape[0], dtype=np.int64)
        adj_matrix[tour[:-1]] = tour[1:]
        adj_matrix = torch.from_numpy(adj_matrix)
        adj_matrix = adj_matrix.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
        adj_matrix = torch.eq(edge_index[1].cpu(), adj_matrix).to(torch.int)
        return adj_matrix
    
    def guided_categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
        torch.set_grad_enabled(True)
        xt = xt.float()  # b, n, n
        xt.requires_grad = True
        t = torch.from_numpy(t).view(1)
        
        # [b, 2, n, n]
        with torch.inference_mode(False):
            # print(points.shape, xt.shape)
            x0_pred = self.forward(
                points.float().to(device),
                xt.to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )
        
        if not self.sparse:
            x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        else:
            x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)
        
        dis_matrix = torch.sqrt(torch.sum((points[edge_index.T[:, 0]] - points[edge_index.T[:, 1]]) ** 2, dim=1))
        dis_matrix = dis_matrix.reshape((1, points.shape[0], -1))
        cost_est = (dis_matrix * x0_pred_prob[..., 1]).sum()
        cost_est.requires_grad_(True)
        cost_est.backward()
        
        if self.args.norm is True:
            xt.grad = nn.functional.normalize(xt.grad, p=2, dim=-1)
            xt = self.guided_categorical_posterior(target_t, t, x0_pred_prob, xt)
     
        return xt.detach()
    
    def train_batch(self, graph):
        batch_size = len(graph.idx)
        # t in 1..T as in DIFUSCO (t=0 would be the clean label itself)
        time = np.random.randint(1, self.num_tp+1, batch_size)
        
        # SB pairs come from sb_draw; CAM_SB_LEGACY_DRAWS=1 (only with --sb_omega 0) keeps the existing draws so a
        # run can be compared step by step with the existing soft-target implementation (plan.md 1B criterion 2)
        sb_pairs = self.sb and not (self.sb_omega == 0 and os.environ.get('CAM_SB_LEGACY_DRAWS'))
        if self.solution_type == 'edge':
            time = time[graph.batch[graph.edge_index[0]].cpu().numpy()]
            solution_0 = graph.edge_attr if self.label_sets is None else self.pick_labels(graph)
        else:
            time = time[graph.batch.cpu().numpy()]
            if sb_pairs:
                solution_0, solution_t = self.sb_draw(graph, time)
            else:
                solution_0 = graph.x if self.label_sets is None else self.pick_labels(graph)

        if sb_pairs:
            pass
        elif self.heated.heat_lambda == 0:
            solution_0_onehot = F.one_hot(solution_0.long(), num_classes=2).float()
            solution_t = self.diffusion.sample(solution_0_onehot, time)
        else:
            assert self.solution_type == 'node'
            A = self.adjacency(graph)
            t_nodes = torch.from_numpy(time).long().to(solution_0.device)
            solution_t = self.heated.sample(A, graph.b, solution_0.float().view(-1, 1), t_nodes)
        soft = None
        if self.label_sets is not None and self.label_choice == 'soft':
            soft = self.sb_soft_targets(graph, solution_t, time) if self.sb else self.soft_targets(graph, solution_t, time)
        time = torch.from_numpy(time).float().to(self.accelerator.device)
        target = solution_0.long().squeeze(1)          
        
        logits = self.model(graph, solution_t, time)
        if soft is not None:   # UPSTREAM.md item 56: posterior mean over the label set as the target
            assert self.graph_weights is None
            lp = torch.log_softmax(logits.float(), -1)
            loss = -(soft * lp[:, 1] + (1 - soft) * lp[:, 0]).mean()
            if self.sb and not sb_pairs and os.environ.get('CAM_LOSS_TRACE'):
                # regression check: the existing soft target on the same x_t and logits, traced next to ours
                ref = self.soft_targets(graph, solution_t, time.long().cpu().numpy())
                return self.trace_loss(loss, -(ref * lp[:, 1] + (1 - ref) * lp[:, 0]).mean())
            return self.trace_loss(loss)

        # Compute loss
        if self.graph_weights is None:
            return self.trace_loss(self.loss_func(logits, target))
        return self.weighted_loss(graph, logits, target, time)

    def trace_loss(self, loss, other=None):
        """Diagnostic (env CAM_LOSS_TRACE=FILE, local addition, UPSTREAM.md item 69): append every step's loss
        (and a second loss on the same step, if given)."""
        f = os.environ.get('CAM_LOSS_TRACE')
        if f:
            with open(f, 'a') as fh:
                fh.write(f'{loss.item():.10f}' + ('' if other is None else f' {other.item():.10f}') + '\n')
        return loss

    def sb_draw(self, graph, time):
        """SB training pair (math_route1.md Cor. 3.3): per graph z ~ Bernoulli(0.5)^n, label m ~ pi*(m | z)
        = softmax(log g_m + log K_{a_T}(s_m | z)), x_t ~ the per-bit reference bridge between s_m and z.
        time: per-node numpy t (one t per graph). Returns (label, x_t), both shaped like graph.x."""
        dev = graph.x.device
        idx = torch.as_tensor(graph.idx).reshape(-1).tolist(); ptr = graph.ptr.tolist()
        a_end = self.sb_a[-1].item()
        ys, xs = [], []
        for g, i in enumerate(idx):
            s = self.label_sets[int(i)].to(dev)
            assert s.shape[1] == ptr[g + 1] - ptr[g], 'label set does not match the graph size'
            z = torch.bernoulli(torch.full((s.shape[1],), 0.5, device=dev))
            logit = self.sb_log_g[int(i)].to(dev) + sb_coupling.log_kernel(z[None], s, a_end)[0]
            y = s[torch.multinomial(torch.softmax(logit, 0), 1).item()].double()
            t = int(time[ptr[g]])
            xs.append(torch.bernoulli(sb_coupling.bridge_prob1(y, z, self.sb_a[t].item(), a_end)))
            ys.append(y)
        return torch.cat(ys).view_as(graph.x).to(graph.x.dtype), torch.cat(xs).view(-1, 1).float()

    def sb_soft_targets(self, graph, solution_t, time):
        """Per node E*[y | x_t] = sum_m P*(m | x_t) s_m with P*(m | x_t) ∝ g_m K_{a_t}(s_m | x_t) (math_route1.md
        Thm 3.1, Cor. 3.2); time: per-node numpy t."""
        dev = solution_t.device
        xt = solution_t.view(-1); out = []
        idx = torch.as_tensor(graph.idx).reshape(-1).tolist(); ptr = graph.ptr.tolist()
        for g, i in enumerate(idx):
            a_t = self.sb_a[int(time[ptr[g]])].item()
            out.append(sb_coupling.posterior_mean(xt[ptr[g]:ptr[g + 1]][None], self.label_sets[int(i)].to(dev),
                                                  self.sb_log_g[int(i)], a_t)[0].float())
        return torch.cat(out)

    def soft_targets(self, graph, solution_t, time):
        """Per node E[x_0 | x_t] over the graph's label set (UPSTREAM.md item 56); time: per-node numpy t."""
        dev = solution_t.device
        xt = solution_t.view(-1); out = []
        if self.solution_type == 'edge':   # time: per-edge numpy t
            for i, sl, e0, e1 in self.edge_slices(graph):
                Qb = torch.from_numpy(self.diffusion.Q_bar[int(time[sl.start])]).double().to(dev)
                out.append(posterior_mean(self.tour_labels(i, e0, e1), xt[sl], Qb).float())
            return torch.cat(out)
        idx = torch.as_tensor(graph.idx).reshape(-1).tolist(); ptr = graph.ptr.tolist()
        for g, i in enumerate(idx):
            Qb = torch.from_numpy(self.diffusion.Q_bar[int(time[ptr[g]])]).double().to(dev)
            out.append(posterior_mean(self.label_sets[int(i)].to(dev), xt[ptr[g]:ptr[g + 1]], Qb).float())
        return torch.cat(out)

    def weighted_loss(self, graph, logits, target, time):
        """Per-graph weighted node cross-entropy (UPSTREAM.md item 45); equal weights give the node mean."""
        assert self.solution_type == 'node'
        ce = F.cross_entropy(logits, target, reduction='none')
        idx = torch.as_tensor(graph.idx, device=ce.device).reshape(-1).long()
        nb = idx.numel()
        cnt = torch.zeros(nb, device=ce.device).index_add_(0, graph.batch, torch.ones_like(ce))
        ce_g = torch.zeros(nb, device=ce.device).index_add_(0, graph.batch, ce.detach()) / cnt
        t_g = torch.zeros(nb, device=ce.device).index_add_(0, graph.batch, time) / cnt
        w = self.graph_weights.weights(idx).to(ce.device)[graph.batch]
        # every process records the hardness of all graphs of the step, so the tables stay identical
        acc = self.accelerator
        allv = acc.gather(acc.pad_across_processes(torch.stack([idx.float(), t_g, ce_g], 1), dim=0, pad_index=-1))
        allv = allv[allv[:, 0] >= 0]
        self.graph_weights.update(allv[:, 0].long(), allv[:, 1], allv[:, 2])
        if self.global_step % 100 == 0:   # effective sample size of the weights over all training graphs
            wa = self.graph_weights.weights(torch.arange(self.graph_weights.num_graphs))
            self.log({'train/weight_ess_frac': effective_sample_size(wa) / len(wa)})
        return (w * ce).sum() / w.sum()

    @staticmethod
    def adjacency(graph):
        return torch.sparse_coo_tensor(graph.edge_index, graph.edge_weight,
                                       torch.Size((graph.num_nodes, graph.num_nodes))).to_sparse_csr()

    def renoise(self, graph, x0_pred_prob, t2):
        """Re-noising sampler: draw x0 ~ Bernoulli(p) and run the training forward from it to t2.

        It does not use x_t. It is the reverse of the forward q(x_{t-1} | x_t, x_0) = q(x_{t-1} | x_0),
        which has the same marginals q(x_t | x_0) as the training forward (DDIM, Song et al. 2021 §4).
        At t2 = 0 it returns the predicted probabilities, as the posterior sampler does."""
        n_var = graph.edge_index.shape[1] if self.solution_type == 'edge' else graph.num_nodes
        p1 = x0_pred_prob[..., 1].reshape(n_var, -1)
        if t2 == 0:
            return p1.clamp(0, 1)
        x0 = torch.bernoulli(p1.clamp(0, 1))
        if self.solution_type == 'edge':   # edge variables (TSP): the plain training forward, UPSTREAM.md item 60
            assert self.sampler in ('renoise', 'renoise_decode'), 'renoise_feasible is for node problems only'
            if self.sampler == 'renoise_decode':
                x0 = self.decode_tour(graph, x0, p1)
            return self.diffusion.sample(F.one_hot(x0.long(), num_classes=2).float(), int(t2))
        if self.sampler in ('renoise_decode', 'renoise_feasible'):
            x0 = self.decode_x0(graph, x0, p1)
        if self.sb:   # sampler B (math_route1.md §5): the reference bridge between x0 and the chain's start z
            return torch.bernoulli(sb_coupling.bridge_prob1(x0, self.sb_z, self.sb_a[int(t2)].item(),
                                                            self.sb_a[-1].item())).float()
        t_nodes = torch.full((graph.num_nodes,), int(t2), dtype=torch.long, device=x0.device)
        return self.heated.sample(graph.A, graph.b, x0, t_nodes)

    @torch.no_grad()
    def trace_inputs(self, graph, trace, final):
        """Diagnostic (env CAM_TRACE_INPUTS=FILE, local addition, UPSTREAM.md item 59): train-test consistency of
        the network inputs along the sampling trajectory. For every step, compare the sampler's input x_t with a
        training-form input at the same t, i.e. the chain's own final decoded solution noised by the training
        forward: number of 1s, number of edges with both ends 1, and the cross-entropy of the prediction
        against the final solution. Appends one JSON line per graph."""
        e0, e1 = graph.edge_index
        def stats(x, p):
            ce = -(final * torch.log(p.clamp_min(1e-8)) + (1 - final) * torch.log((1 - p).clamp_min(1e-8))).mean(0)
            return [x.sum(0).mean().item(), (x[e0] * x[e1]).sum(0).mean().item() / 2, ce.mean().item()]
        rows = []
        for t1, x, p in trace:
            t_nodes = torch.full((graph.num_nodes,), int(t1), dtype=torch.long, device=x.device)
            xr = self.heated.sample(graph.A, graph.b, final, t_nodes)
            time = torch.full_like(xr, float(t1))
            pr = self.model(graph, xr.unsqueeze(-1), time).softmax(-1)[..., 1].reshape(graph.num_nodes, -1)
            rows.append([int(t1)] + stats(x, p) + stats(xr, pr))
        with open(os.environ['CAM_TRACE_INPUTS'], 'a') as f:
            f.write(json.dumps({'n': graph.num_nodes, 'final': final.sum(0).mean().item(), 'steps': rows}) + '\n')

    def decode_tour(self, graph, x0, p1):
        """TSP counterpart of decode_x0 (UPSTREAM.md item 62): the sampled edges x0 (score 1 + p, the rest p) are
        merged into a tour by the env's greedy edge insertion, then --decode_two_opt iterations of 2-opt (0: none);
        the tour is returned as undirected edge labels on the sparse graph, the form the tour-set labels have."""
        two = self.problem.two_opt_iterations
        self.problem.two_opt_iterations = self.decode_two_opt
        try:
            _, tours = self.problem.evaluate(graph, x0 * (1 + p1) + (1 - x0) * p1)
        finally:
            self.problem.two_opt_iterations = two
        e0, e1 = graph.edge_index
        return edge_indicator(tours.T[:, :-1].contiguous(), e0, e1).T

    def decode_x0(self, graph, x0, p1):
        """Make the sampled x0 a feasible solution with the problem's own decoder before re-noising, so the
        network input at every step is a noised feasible solution, as in training (local addition, UPSTREAM.md
        item 59). The greedy decoder takes the sampled nodes first (score 1 + p) and then the rest (score p):
        'renoise_decode' keeps the resulting maximal solution, 'renoise_feasible' only its sampled nodes
        (conflicts removed, nothing added)."""
        _, sol = self.problem.evaluate(graph, x0 * (1 + p1) + (1 - x0) * p1)
        sol = sol.float()
        return sol * x0 if self.sampler == 'renoise_feasible' else sol

    def write_pair_trace(self, graph, start, final):
        """Append one JSON line per graph (env CAM_SB_PAIR_TRACE=FILE, UPSTREAM.md item 69): the graph's dataset
        index, and per chain its start z and decoded final solution as hex bit strings (node order of the loader)."""
        def hexs(m):   # (n, k) 0/1 -> k hex strings of the n bits
            bits = m.round().long().cpu().T.tolist()
            return [format(int(''.join(map(str, b)), 2), 'x') for b in bits]
        idx = int(torch.as_tensor(graph.idx).reshape(-1)[0]) if hasattr(graph, 'idx') else -1
        with open(os.environ['CAM_SB_PAIR_TRACE'], 'a') as f:
            f.write(json.dumps({'idx': idx, 'n': graph.num_nodes, 'start': hexs(start), 'final': hexs(final.float())}) + '\n')

    def evaluate_single(self, graph):
        best_energies = []
        for _ in range(self.num_repetition):
            if self.solution_type == 'edge':
                num_edges = graph.edge_index.shape[1]
                solution = torch.randint(0,2,size=(num_edges,self.num_k), device=self.accelerator.device, dtype=torch.float)
            else:
                A = torch.sparse_coo_tensor(
                    graph.edge_index, 
                    graph.edge_weight, 
                    torch.Size((graph.num_nodes, graph.num_nodes))
                ).to_sparse_csr()
                graph.A = A
                num_nodes = graph.num_nodes
                solution = torch.randint(0,2,size=(num_nodes,self.num_k), device=self.accelerator.device, dtype=torch.float)
                if os.environ.get('CAM_SB_PAIR_TRACE'):   # diagnostic (UPSTREAM.md item 69): chains 2j, 2j+1
                    assert self.num_k % 2 == 0           # share their start
                    solution[:, 1::2] = solution[:, 0::2]
                start = solution.clone()
                if self.sb:
                    self.sb_z = solution.clone()   # the noise end z of every chain (sampler B bridges back to it)
    
            trace = [] if os.environ.get('CAM_TRACE_INPUTS') else None
            for t in range(self.num_t):
                t1, t2 = self.inference_schedule(t)
                time = torch.ones((solution.shape[0],self.num_k), device=self.accelerator.device, dtype=torch.float)*t1            
                logits = self.model(graph, solution.unsqueeze(-1), time)
                x0_pred_prob = logits.softmax(dim=-1)
                if trace is not None:
                    trace.append((t1, solution.clone(), x0_pred_prob[..., 1].reshape(graph.num_nodes, -1).clone()))
                if self.sampler.startswith('renoise'):
                    solution = self.renoise(graph, x0_pred_prob, t2)
                else:
                    solution = self.categorical_posterior(t2, t1, x0_pred_prob, solution)
    
            best_energy, final = self.problem.evaluate(graph, solution)
            if self.solution_type == 'node' and os.environ.get('CAM_SB_PAIR_TRACE'):
                self.write_pair_trace(graph, start, final)
            if trace is not None:
                self.trace_inputs(graph, trace, final.float())
            best_energies.append(best_energy)
        return torch.cat(best_energies).min() 