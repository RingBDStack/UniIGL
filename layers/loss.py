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
    neighbors = (adj[node] > 0).nonzero().view(-1).tolist()
    return neighbors

def sample_triplets_with_multi_neg(data, num_triplets=2000, num_neg=3):
    triplets = []
    n_nodes = data.num_nodes

    for _ in range(num_triplets):
        v = random.randint(0, n_nodes - 1)
        neighbors_v = get_neighbors(data.adj, v)
        if not neighbors_v:
            continue  
        a = random.choice(neighbors_v)
        
        negs = []
        while len(negs) < num_neg:
            b = random.randint(0, n_nodes - 1)
            if b != v and b != a and b not in neighbors_v:
                negs.append(b)

        triplets.append((v, a, negs))

    return triplets

def cosine_sim(x, y):
    return F.cosine_similarity(x, y, dim=-1)

def linkpred_multi_neg_loss(emb, triplets, tau=0.5):
    device = emb.device
    total_loss = 0.0

    for (v, a, neg_list) in triplets:
        s_v = emb[v]
        s_a = emb[a]
        s_neg = emb[neg_list]  
        
        sim_va = cosine_sim(s_v, s_a) / tau 
        sim_vneg = cosine_sim(s_v.unsqueeze(0), s_neg) / tau 

        numerator = torch.exp(sim_va)
        denominator = numerator + torch.sum(torch.exp(sim_vneg))
        
        loss_triplet = -torch.log(numerator / denominator)
        total_loss += loss_triplet

    return total_loss / len(triplets)



