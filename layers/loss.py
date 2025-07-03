import torch
import torch.nn.functional as F
import random

# for one-hot softlabel
def mixup_cross_entropy(pred, softlabel):
    return -torch.sum(softlabel * torch.log_softmax(pred, dim=1), dim=1).mean()

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss*1e-3

def alpha_entmax(input_tensor, alpha=1.8, dim=-1):    
    sorted_tensor, _ = torch.sort(input_tensor, descending=True, dim=dim)
    
    cumulative_sum = torch.cumsum(sorted_tensor, dim=dim)
    
    k = torch.arange(1, input_tensor.size(dim) + 1, device=input_tensor.device, dtype=input_tensor.dtype)
    
    threshold = (cumulative_sum - 1) / k
    
    k = (sorted_tensor > threshold).sum(dim=dim, keepdim=True)
    
    tau = (cumulative_sum.gather(dim, k - 1) - 1) / k.float()   
    output = torch.relu(input_tensor - tau)
    
    return output.pow(alpha - 1)

def get_neighbors(adj, node):
    """
    给定邻接矩阵 adj (假设形状 [N, N] 或稀疏格式等)，
    返回一个节点的邻居列表(存在边的节点)。
    """
    # 如果是稠密矩阵，可以这样写：
    neighbors = (adj[node] > 0).nonzero().view(-1).tolist()
    return neighbors

def sample_triplets_with_multi_neg(data, num_triplets=2000, num_neg=3):
    """
    从图 data 中随机采样若干三元组 (v, a, B)，其中:
      - (v,a) 是存在边的正例
      - B = {b1, ..., b_m} 与 v 不相连的多负例集合
    num_triplets: 需要采样的三元组数量
    num_neg: 每个三元组包含多少个负例
    """
    triplets = []
    n_nodes = data.num_nodes

    for _ in range(num_triplets):
        # 1) 随机选一个正例边 (v,a)
        v = random.randint(0, n_nodes - 1)
        neighbors_v = get_neighbors(data.adj, v)
        if not neighbors_v:
            continue  # 如果这个 v 没有邻居，就跳过
        a = random.choice(neighbors_v)
        
        # 2) 采样多负例 B
        #   简单做法: 在 [0..n_nodes-1] 范围内随机抽 num_neg 个节点，
        #   要求和 v 无边，且不与 v=a 重复
        negs = []
        while len(negs) < num_neg:
            b = random.randint(0, n_nodes - 1)
            if b != v and b != a and b not in neighbors_v:
                negs.append(b)

        triplets.append((v, a, negs))

    return triplets

def cosine_sim(x, y):
    """
    计算批量余弦相似度，返回形状[..., 1]或[...,]的张量
    x, y 的最后一维是特征维度
    """
    # 如果 x,y 均是 [d], 则 F.cosine_similarity(x,y,dim=0)
    # 如果 x,y 是批量 [batch_size, d]，则 dim=-1
    return F.cosine_similarity(x, y, dim=-1)

def linkpred_multi_neg_loss(emb, triplets, tau=0.5):
    """
    实现多负样本的 InfoNCE 风格链路预测损失:
      L = - Σ_{(v,a,B)} ln( exp(sim(v,a)/tau) / Σ_{b_i in B} exp(sim(v,b_i)/tau) )

    emb: [N, d] 的所有节点表示 (GNN输出)
    triplets: [(v,a,[b1,b2,...]), ...] 三元组
    tau: 温度参数
    """
    device = emb.device
    total_loss = 0.0

    for (v, a, neg_list) in triplets:
        s_v = emb[v]
        s_a = emb[a]
        s_neg = emb[neg_list]  # 形状 [num_neg, d]
        
        # 计算相似度
        sim_va = cosine_sim(s_v, s_a) / tau  # 标量
        sim_vneg = cosine_sim(s_v.unsqueeze(0), s_neg) / tau  # [num_neg]向量

        numerator = torch.exp(sim_va)
        denominator = numerator + torch.sum(torch.exp(sim_vneg))
        
        loss_triplet = -torch.log(numerator / denominator)
        total_loss += loss_triplet

    return total_loss / len(triplets)



