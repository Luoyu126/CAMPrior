from .meta_solver import SLSolver
from utils.diffusion_schedulers import *
import torch
from torch import nn
import os
import glob
from utils.bridge_paths import sample_path, step_flip_prob


def load_bank(path, data_dir, num_samples):
    # Local addition (UPSTREAM.md item 43): MCMC start states from scripts/mis/build_start_bank.py,
    # one uint8 (num_nodes, K) tensor per graph, in the dataset's sorted file order.
    bank = torch.load(path, weights_only=False)
    files = [os.path.basename(f) for f in sorted(glob.glob(data_dir + '/*gpickle'))]
    if num_samples is not None and num_samples > 0:
        files = files[:num_samples]
    assert bank['files'][:len(files)] == files, f'start bank {path} does not match the graphs in {data_dir}'
    return bank['x']


class FlowMatching(SLSolver):
    def __init__(self, args):
        super(FlowMatching, self).__init__(args)
        self.diffusion = CategoricalFM()
        self.inference_schedule = FMInferenceSchedule(inference_schedule=args.inference_schedule, inference_T=args.num_t)
        
        self.loss_func = nn.BCEWithLogitsLoss()

        # Local addition (UPSTREAM.md item 43): start from a bank of MCMC samples instead of
        # Bernoulli(1/2), with the mixture path (lambda = 0) or a uniform-flip bridge (lambda > 0).
        self.bridge_lambda = getattr(args, 'fm_bridge_lambda', 0.0)
        self.bank_train = self.bank_eval = None
        if getattr(args, 'fm_bank_train', None) and args.do_train:
            self.bank_train = load_bank(args.fm_bank_train, args.train_path, args.train_samples)
        if getattr(args, 'fm_bank_eval', None):
            eval_dir, eval_n = (args.valid_path, args.valid_samples) if args.do_train else (args.test_path, args.test_samples)
            self.bank_eval = load_bank(args.fm_bank_eval, eval_dir, eval_n)
        if self.bridge_lambda > 0 or self.bank_train is not None:
            assert self.solution_type == 'node', 'start banks / bridges are implemented for node problems'

    def bank_starts(self, graph, bank, k=None):
        # one random bank column per graph (k is None), or the first k columns (evaluation)
        idx = torch.as_tensor(graph.idx).reshape(-1).tolist()
        parts = []
        for i in idx:
            x = bank[int(i)]
            if k is None:
                x = x[:, torch.randint(x.shape[1], (1,)).item()]
            else:
                assert x.shape[1] >= k, f'start bank has {x.shape[1]} chains, need {k}'
                x = x[:, :k]
            parts.append(x)
        return torch.cat(parts, 0).to(self.accelerator.device, torch.float)
      
    def train_batch(self, graph):
        batch_size = len(graph.idx)
        time = torch.rand(batch_size, device=self.accelerator.device)

        if self.solution_type == 'edge':
            time = time[graph.batch[graph.edge_index[0]]]
            solution_0 = graph.edge_attr.squeeze(-1)
        else:
            time = time[graph.batch]
            solution_0 = graph.x.squeeze(-1)

        if self.bank_train is None and self.bridge_lambda == 0:
            solution_t = self.diffusion.sample(solution_0, time)
        else:
            x0 = self.bank_starts(graph, self.bank_train) if self.bank_train is not None \
                else torch.randint(0, 2, solution_0.shape, device=solution_0.device).to(solution_0.dtype)
            solution_t = sample_path(x0.to(solution_0.dtype), solution_0, time, self.bridge_lambda)
        target = solution_0.float()
        
        logits = self.model(graph, solution_t.unsqueeze(-1), time).squeeze(-1)
        
        # Compute loss
        loss = self.loss_func(logits, target)
        
        return loss 
    
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
                if self.bank_eval is not None:
                    solution = self.bank_starts(graph, self.bank_eval, self.num_k)
                else:
                    solution = torch.randint(0,2,size=(num_nodes,self.num_k), device=self.accelerator.device, dtype=torch.float)
    
    
            for t in range(self.num_t):
                t1, t2 = self.inference_schedule(t)
                time = torch.ones((solution.shape[0],self.num_k), device=self.accelerator.device, dtype=torch.float)*t1            
                logits = self.model(graph, solution.unsqueeze(-1), time).squeeze(-1)
    
                if t == self.num_t-1:
                    solution = torch.sigmoid(logits)
                elif self.bank_eval is not None or self.bridge_lambda > 0:
                    flip = step_flip_prob(solution, torch.sigmoid(logits), t1, t2, self.bridge_lambda)
                    solution = torch.where(torch.rand_like(solution) < flip, 1 - solution, solution)
                else:
                    # Euler step of the velocity (x1_hat - x_t) / (1 - t) for x_t = t*x1 + (1-t)*0.5
                    solution_prob = solution + (torch.sigmoid(logits)-solution)*(t2-t1)/(1-t1)
                    solution = torch.bernoulli(solution_prob.clamp(0,1))
    
            best_energy, _ = self.problem.evaluate(graph, solution)
            best_energies.append(best_energy)
            
        return torch.cat(best_energies).min()
    
