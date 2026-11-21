from .meta_solver import SLSolver
from utils.diffusion_schedulers import *
from utils.heated_forward import HeatedForward
from utils.label_sets import load_label_sets, choose, posterior_mean
from utils.graph_weights import GraphWeights, effective_sample_size
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
        self.heated = HeatedForward(self.diffusion.beta, getattr(args, 'heat_lambda', 0.0),
                                    getattr(args, 'heat_until', 0.5), args.beta)
        # local addition (UPSTREAM.md item 44): several optimal labels per training graph
        self.label_sets, self.label_choice = None, getattr(args, 'label_choice', 'first')
        self.label_temp = getattr(args, 'label_temp', 1.0)
        if getattr(args, 'label_set', None) and args.do_train:
            self.label_sets = load_label_sets(args.label_set, args.train_path, args.train_samples)
        # local addition (UPSTREAM.md item 45): per-graph weights of the training loss
        self.graph_weights = None
        mode = getattr(args, 'graph_weight', 'none')
        if mode != 'none' and args.do_train:
            sets = load_label_sets(args.graph_weight_sets, args.train_path, args.train_samples)
            self.graph_weights = GraphWeights(mode, len(sets), nsets=[s.shape[0] for s in sets],
                                              power=args.graph_weight_power, num_tp=args.num_tp)

    def pick_labels(self, graph):
        """One label per graph of the batch from its label set (UPSTREAM.md item 44), shape like graph.x."""
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
        
        if self.solution_type == 'edge':
            time = time[graph.batch[graph.edge_index[0]].cpu().numpy()]
            solution_0 = graph.edge_attr
        else:
            time = time[graph.batch.cpu().numpy()]
            solution_0 = graph.x if self.label_sets is None else self.pick_labels(graph)

        if self.heated.heat_lambda == 0:
            solution_0_onehot = F.one_hot(solution_0.long(), num_classes=2).float()
            solution_t = self.diffusion.sample(solution_0_onehot, time)
        else:
            assert self.solution_type == 'node'
            A = self.adjacency(graph)
            t_nodes = torch.from_numpy(time).long().to(solution_0.device)
            solution_t = self.heated.sample(A, graph.b, solution_0.float().view(-1, 1), t_nodes)
        soft = None
        if self.label_sets is not None and self.label_choice == 'soft':
            soft = self.soft_targets(graph, solution_t, time)
        time = torch.from_numpy(time).float().to(self.accelerator.device)
        target = solution_0.long().squeeze(1)          
        
        logits = self.model(graph, solution_t, time)
        if soft is not None:   # UPSTREAM.md item 56: posterior mean over the label set as the target
            assert self.graph_weights is None
            lp = torch.log_softmax(logits.float(), -1)
            return -(soft * lp[:, 1] + (1 - soft) * lp[:, 0]).mean()

        # Compute loss
        if self.graph_weights is None:
            return self.loss_func(logits, target)
        return self.weighted_loss(graph, logits, target, time)

    def soft_targets(self, graph, solution_t, time):
        """Per node E[x_0 | x_t] over the graph's label set (UPSTREAM.md item 56); time: per-node numpy t."""
        dev = solution_t.device
        idx = torch.as_tensor(graph.idx).reshape(-1).tolist(); ptr = graph.ptr.tolist()
        xt = solution_t.view(-1); out = []
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
        p1 = x0_pred_prob[..., 1].reshape(graph.num_nodes, -1)
        if t2 == 0:
            return p1.clamp(0, 1)
        x0 = torch.bernoulli(p1.clamp(0, 1))
        if self.sampler in ('renoise_decode', 'renoise_feasible'):
            x0 = self.decode_x0(graph, x0, p1)
        t_nodes = torch.full((graph.num_nodes,), int(t2), dtype=torch.long, device=x0.device)
        return self.heated.sample(graph.A, graph.b, x0, t_nodes)

    def decode_x0(self, graph, x0, p1):
        """Make the sampled x0 a feasible solution with the problem's own decoder before re-noising, so the
        network input at every step is a noised feasible solution, as in training (local addition, UPSTREAM.md
        item 59). The greedy decoder takes the sampled nodes first (score 1 + p) and then the rest (score p):
        'renoise_decode' keeps the resulting maximal solution, 'renoise_feasible' only its sampled nodes
        (conflicts removed, nothing added)."""
        _, sol = self.problem.evaluate(graph, x0 * (1 + p1) + (1 - x0) * p1)
        sol = sol.float()
        return sol * x0 if self.sampler == 'renoise_feasible' else sol

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
    
            for t in range(self.num_t):
                t1, t2 = self.inference_schedule(t)
                time = torch.ones((solution.shape[0],self.num_k), device=self.accelerator.device, dtype=torch.float)*t1            
                logits = self.model(graph, solution.unsqueeze(-1), time)
                x0_pred_prob = logits.softmax(dim=-1)
                if self.sampler.startswith('renoise'):
                    solution = self.renoise(graph, x0_pred_prob, t2)
                else:
                    solution = self.categorical_posterior(t2, t1, x0_pred_prob, solution)
    
            best_energy, _ = self.problem.evaluate(graph, solution)
            best_energies.append(best_energy)
        return torch.cat(best_energies).min() 