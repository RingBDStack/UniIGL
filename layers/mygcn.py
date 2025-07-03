import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

class GCNLayer(nn.Module):
    def __init__(self, in_feats, n_hidden, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_feats, n_hidden)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(n_hidden)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(n_hidden) if batch_norm else None


    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)
        return output


    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, graph_hops, dropout, batch_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.n_classes = n_classes

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(in_feats, n_hidden, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GCNLayer(n_hidden, n_hidden, batch_norm=batch_norm))

        self.graph_encoders.append(GCNLayer(n_hidden, n_hidden, batch_norm=False))

    def forward(self, x, adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, adj))
            # x = F.leaky_relu(encoder(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, adj)
        return x
    
    def load_pretrained_weights(self, path):
        state_dict = torch.load(path, map_location='cuda', weights_only=True)
        self.load_state_dict(state_dict)
        print(f"Pretrained parameters loaded from {path}")

    @torch.no_grad()
    def compute_second_term_logits(
        self, x, adj, prototypes, tau2: float, T: int = 20
    ) -> torch.Tensor:
        self.train()                    
        logits_samples = []
        for _ in range(T):
            emb = self.forward(x, adj) 
            logits_t = F.cosine_similarity(
                emb.unsqueeze(1), prototypes.unsqueeze(0), dim=-1
            ) / tau2                    # (N, C)
            logits_samples.append(logits_t.unsqueeze(0))  

        logits_mc = torch.cat(logits_samples, dim=0)     
        self.eval()                 
        return logits_mc
    
class Prompt_module(nn.Module):
    def __init__(self, hidden_dim: int):
        super(Prompt_module, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, node_embedding):
        return node_embedding * self.weight
    
    def compute_prototype_logits(
        self,
        node_embeddings: torch.Tensor,
        labels: torch.Tensor, 
        train_idx: torch.Tensor,
        tau2: float,
        return_prototypes: bool = False
    ) :
        num_classes = labels.shape[1]
        device = node_embeddings.device

        class_prototypes = torch.zeros(num_classes, node_embeddings.size(1), device=device)
        train_nodes = train_idx
        # node_embeddings = F.normalize(node_embeddings, dim=-1)
        
        for c in range(num_classes):
            class_weights = labels[train_nodes, c]
            weighted_sum = torch.sum(node_embeddings[train_nodes] * class_weights.unsqueeze(-1), dim=0)
            total_weight = class_weights.sum()
            
            class_prototypes[c] = weighted_sum / total_weight
            
        # class_prototypes = F.normalize(class_prototypes, dim=-1)
            
        if return_prototypes:
            return class_prototypes
        else:
            train_embeddings = node_embeddings[train_idx]  # [num_train, D]
            sim_matrix = F.cosine_similarity(
                train_embeddings.unsqueeze(1),  # [B, 1, D]
                class_prototypes.unsqueeze(0),  # [1, C, D]
                dim=-1
            ) / tau2

            return sim_matrix, labels[train_idx]  

    def compute_prototype_loss(
        self,
        logits: torch.Tensor,
        soft_labels: torch.Tensor  
    ) -> torch.Tensor:

        exp_sim = torch.exp(logits)
        probs = exp_sim / (exp_sim.sum(dim=1, keepdim=True)) + 1e-8  

        weighted_loss = -torch.sum(soft_labels * torch.log(probs), dim=1)
        return weighted_loss.mean()
    
class Topology_Prompt(nn.Module):
    def __init__(self, new_num, old_num):
        super(Topology_Prompt, self).__init__()
        self.new_num = new_num
        self.old_num = old_num
        self.pmt = nn.Parameter(torch.Tensor(new_num, old_num))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.pmt, 0, 1) 
            
    def forward(self, ori_adj):
            
        top_left = ori_adj[:self.old_num, :self.old_num]
        top_right = ori_adj[:self.old_num, self.old_num:]
        bottom_left = ori_adj[self.old_num:, :self.old_num]
        bottom_right = ori_adj[self.old_num:, self.old_num:]

        new_bottom_left = F.normalize(bottom_left + self.pmt, p=1, dim=1)
        new_top_right = F.normalize(top_right + self.pmt.t(), p=1, dim=1)

        new_top = torch.cat([top_left, new_top_right], dim=1)
        new_bottom = torch.cat([new_bottom_left, bottom_right], dim=1)
        new_adj = torch.cat([new_top, new_bottom], dim=0)
        
        return new_adj