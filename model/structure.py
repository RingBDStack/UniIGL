import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_n_hop_neighbors(adj, node, n_hop):
    if sparse.issparse(adj):
        return get_n_hop_neighbors_sparse(adj, node, n_hop)
    else:
        return get_n_hop_neighbors_dense(adj, node, n_hop)

def get_n_hop_neighbors_dense(adj, node, n_hop):
    num_nodes = adj.shape[0]
    visited = set()
    queue = deque([(node, 0)])
    visited.add(node)
    
    while queue:
        u, dist = queue.popleft()
        if dist >= n_hop:
            continue
        neighbors = np.where(adj[u] != 0)[0]
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                queue.append((v, dist + 1))
    return visited

def get_n_hop_neighbors_sparse(adj, node, n_hop):
    visited = set()
    queue = deque([(node, 0)])
    visited.add(node)
    
    while queue:
        u, dist = queue.popleft()
        if dist >= n_hop:
            continue
        neighbors = adj[u].indices
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                queue.append((v, dist + 1))
    return visited

def adjust_structure(values, embeddings, adj, k, n, m):
    if sparse.issparse(adj):
        adj = adj.tolil()
        new_adj = adj.copy().tolil()
    else:
        new_adj = np.copy(adj)
    
    # Step 1: 选择values最低的top-k个节点
    # top_k_nodes = np.argsort(values)[::-1][:k]
    top_k_nodes = np.argsort(values)[:k]
   
    # Step 2: 获取每个top节点的n跳邻居区域
    regions = []
    for node in top_k_nodes:
        region = get_n_hop_neighbors(adj, node, n)
        regions.append(region)
    
    # Step 3: 对每个区域进行处理
    for region in regions:
        nodes_in_region = list(region)
        if len(nodes_in_region) < 2:
            continue  # 没有边可以处理
        
        # 提取嵌入并计算相似度矩阵
        sub_emb = embeddings[nodes_in_region]
        sim_matrix = cosine_similarity(sub_emb)
        
        # 收集所有可能的无向边对（i < j）
        existing_edges = []
        non_existing_edges = []
        num_nodes = len(nodes_in_region)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                u = nodes_in_region[i]
                v = nodes_in_region[j]
                similarity = sim_matrix[i, j]
                # 检查边是否存在
                if new_adj[u, v] != 0:
                    existing_edges.append( (u, v, similarity) )
                else:
                    non_existing_edges.append( (u, v, similarity) )
        
        # Step 4: 选择要删除和添加的边
        # 删除：existing中相似度最低的m条
        existing_sorted = sorted(existing_edges, key=lambda x: x[2])
        to_remove = existing_sorted[:m]
        
        # 添加：non_existing中相似度最高的m条
        non_existing_sorted = sorted(non_existing_edges, key=lambda x: -x[2])
        to_add = non_existing_sorted[:m]
        
        # 更新邻接矩阵（处理无向边）
        for u, v, _ in to_remove:
            new_adj[u, v] = 0
            new_adj[v, u] = 0
        for u, v, _ in to_add:
            new_adj[u, v] = 1
            new_adj[v, u] = 1
    
    new_adj = torch.tensor(new_adj, dtype=torch.float32)
    if sparse.issparse(adj):
        return new_adj.tocsr()  # 转换为常用格式
    else:
        return new_adj
    
def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sigmoid(logits, tau=1.0, hard=False, bias=-1):
    gumbels = sample_gumbel(logits.shape).to(logits.device)
    y = logits + gumbels + bias
    y = torch.sigmoid(y / tau) 
    
    if hard:
        y_hard = (y > 0.5).float() 
        y = (y_hard - y).detach() + y  
    return y

class GraphLearner(torch.nn.Module):
    def __init__(self, num_nodes, num, init_param, tau=0.1):
        super(GraphLearner, self).__init__()
        self.num_nodes = num_nodes
        self.num = num
        self.tau = tau
        self.init_adj = init_param
        
        self.theta = torch.nn.Parameter(torch.randn(num, num_nodes-num))  

    def forward(self, original_adj_matrix):
        orinum = self.num_nodes - self.num  
        newnum = self.num 

        adj_upper_left = original_adj_matrix[:orinum, :orinum]  
        adj_lower_right = original_adj_matrix[orinum:, orinum:]  

        logits = self.theta
        adj_lower_left_learned = gumbel_sigmoid(logits, self.tau, hard=True) 
        adj_lower_left_learned = torch.mul(adj_lower_left_learned,self.init_adj)
        adj_upper_right_learned = adj_lower_left_learned.T  

        adj_upper = torch.cat([adj_upper_left, adj_upper_right_learned], dim=1) 
        adj_lower = torch.cat([adj_lower_left_learned, adj_lower_right], dim=1)  
        
        adj_matrix = torch.cat([adj_upper, adj_lower], dim=0)
        
        adj_matrix = normalize_adj_torch(adj_matrix)

        return adj_matrix
    
def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx

class GraphLearner2(nn.Module):
    def __init__(self, embed_dim, new_embed_dim, mixnum, num_heads=4):
        super(GraphLearner2, self).__init__()
        self.num_heads = num_heads
        self.new_embed_dim = new_embed_dim
        self.mixnum = mixnum

        self.linear_layers = nn.ModuleList([ 
            nn.Linear(embed_dim, new_embed_dim, bias=False) for _ in range(num_heads)
        ])
        
        self.head_weights = nn.Parameter(torch.ones(num_heads)) 

    def forward(self, node_embed):
        num = self.mixnum
        last_embeddings = node_embed[-num:]
        previous_embeddings = node_embed[:-num:]

        all_head_outs = []
        for i in range(self.num_heads):
            last_combine = self.linear_layers[i](last_embeddings)
            prev_combine = self.linear_layers[i](previous_embeddings)
            
            last_norm = F.normalize(last_combine, p=2, dim=-1)
            prev_norm = F.normalize(prev_combine, p=2, dim=-1)
            cosine_similarity = torch.mm(last_norm, prev_norm.transpose(-1, -2))
            
            all_head_outs.append(cosine_similarity * self.head_weights[i])  

        avg_cosine_similarity = torch.mean(torch.stack(all_head_outs), dim=0)
        
        return F.relu(avg_cosine_similarity)