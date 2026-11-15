import os
import hashlib
import numpy as np
import torch
import pickle
import glob
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, is_undirected, to_undirected
import time

class Dataset(Dataset):
    def __init__(self, data_dir, dtype=torch.float32, num_samples=-1, args=None):
        super(Dataset, self).__init__()
        self.dtype = dtype
             
        self.files = sorted(glob.glob(data_dir + '/*gpickle'))
        self.load_data = self.load_gpickle
    
        if os.path.exists(os.path.join(data_dir, 'anno')):
            self.data_label_path = os.path.join(data_dir, 'anno')
        else:
            self.data_label_path = None

        if num_samples>0:
            self.files = self.files[:num_samples]
    
    def __len__(self):
        return len(self.files)
    
    def load_dimacs(self, idx):
        G = nx.Graph()
        
        with open(self.files[idx], 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Parse the line
            parts = line.split()
            
            # Problem line (p edge NODES EDGES)
            if parts[0] == 'p' and parts[1] == 'edge':
                num_nodes = int(parts[2])
                # Pre-add all nodes
                G.add_nodes_from(range(1, num_nodes + 1))
            
            # Edge line (e NODE1 NODE2)
            elif parts[0] == 'e':
                node1 = int(parts[1])
                node2 = int(parts[2])
                G.add_edge(node1, node2)

        data = from_networkx(G)
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_weight = torch.ones_like(data.edge_index[0], dtype=self.dtype)/2
        data.x = torch.ones(data.num_nodes,1, dtype=self.dtype)
        data.b = -torch.ones_like(data.x)
        data.idx = idx
        return data

    def load_gpickle(self, idx):
        # Local addition (see UPSTREAM.md): with CAM_GRAPH_CACHE=<dir> the parsed graph (edge_index
        # as loaded below, int16, plus the label) is kept as a .pt file, because parsing a dense
        # ER-[700-800] gpickle with networkx takes ~1-3 s per graph and starves supervised
        # training, which reads every training graph each epoch. The cached Data is identical.
        cache_dir = os.environ.get('CAM_GRAPH_CACHE')
        if cache_dir:
            src = os.path.abspath(self.files[idx])
            key = hashlib.sha1(os.path.dirname(src).encode()).hexdigest()[:12]
            cache_file = os.path.join(cache_dir, key, os.path.basename(src) + '.pt')
            if os.path.exists(cache_file):
                c = torch.load(cache_file, weights_only=True)
                return self._make_data(c['edge_index'].long(), int(c['num_nodes']), c['solution'], idx)
        data = self._load_gpickle_uncached(idx)
        if cache_dir:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            tmp = f'{cache_file}.{os.getpid()}.tmp'
            sol = None if self._last_solution is None else torch.tensor(self._last_solution, dtype=torch.int8)
            assert data.num_nodes < 32768
            torch.save({'edge_index': data.edge_index.to(torch.int16), 'num_nodes': data.num_nodes,
                        'solution': sol}, tmp)
            os.replace(tmp, cache_file)
        return data

    def _make_data(self, edge_index, num_nodes, solution, idx):
        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        data.edge_weight = torch.ones_like(data.edge_index[0], dtype=self.dtype)/2
        if solution is None:
            data.x = torch.ones(data.num_nodes,1, dtype=self.dtype)
        else:
            data.x = solution.reshape(data.num_nodes,1).to(self.dtype)
        data.b = -torch.ones_like(data.x)
        data.idx = idx
        return data

    def _load_gpickle_uncached(self, idx):
        if self.data_label_path is not None:
            file_name = os.path.basename(self.files[idx])
            anno_path = os.path.join(self.data_label_path, file_name.replace('.gpickle', '_unweighted.result'))
            if os.path.exists(anno_path):
                solution = open(anno_path).readlines()
                solution = [int(x.strip()) for x in solution]
            else:
                solution = None
        else:
            solution = None
            
        G = nx.read_gpickle(self.files[idx])
        if solution is not None:
            # upstream 1d3f4ae: the .result labels are indexed by sorted(G.nodes()), while
            # from_networkx below indexes nodes by G's insertion order: realign the label to that order
            nodes = list(G.nodes())
            sorted_idx = {v: j for j, v in enumerate(sorted(nodes))}
            solution = [solution[sorted_idx[v]] for v in nodes]
        self._last_solution = solution
        data = from_networkx(G)
        if not is_undirected(data.edge_index):
            data.edge_index = to_undirected(data.edge_index)
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_weight = torch.ones_like(data.edge_index[0], dtype=self.dtype)/2
        if solution is None:
            data.x = torch.ones(data.num_nodes,1, dtype=self.dtype)
        else:
            data.x = torch.tensor(solution).reshape(data.num_nodes,1).to(self.dtype)

        data.b = -torch.ones_like(data.x)
        data.idx = idx
        return data
        
    def __getitem__(self, idx):
        return self.load_data(idx)
    