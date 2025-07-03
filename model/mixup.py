import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def one_hot_encoding(labels, num_classes):
    return F.one_hot(labels, num_classes).float()

def mix_feature(data, mixpool,lam,args):
    num = args.mixnum
    new_train_idx = data.train_idx.clone()  

    gen_features = []
    new_labels = []
    new_ids = []
    origin_nodes_pairs = []  

    current_max_id = data.num_nodes 

    for _ in range(num):
        i, j = mixpool[_]

        new_label = lam * data.labels[i] + (1 - lam) * data.labels[j]
        # new_label =  data.labels[i]
        new_feature = lam * data.features[i] + (1 - lam) * data.features[j]

        gen_features.append(new_feature.unsqueeze(0))
        new_labels.append(new_label.unsqueeze(0))
        new_ids.append(current_max_id)
        origin_nodes_pairs.append((i, j))  

        current_max_id += 1

    new_labels = torch.cat(new_labels, dim=0)
    gen_features = torch.cat(gen_features, dim=0)
    new_features = torch.cat([data.features, gen_features], dim=0)
    # new_features = F.dropout(new_features,args.mix_dropout)
    new_labels = torch.cat([data.labels, new_labels], dim=0)
    new_train_idx = torch.cat([new_train_idx, torch.tensor(new_ids)], dim=0)
    
    adj_size = data.adj.shape[0]
    add_num = len(new_ids)
    new_adj = torch.zeros((adj_size + add_num, adj_size + add_num), dtype=data.adj.dtype, device=data.adj.device)

    new_adj[:adj_size, :adj_size] = data.adj

    for idx, (i, j) in enumerate(origin_nodes_pairs):
        new_adj_row = torch.clamp(data.adj[i, :] + data.adj[j, :], min=0.0, max=1.0).unsqueeze(0)
        new_adj_col = torch.clamp(data.adj[:, i] + data.adj[:, j], min=0.0, max=1.0).unsqueeze(1)
        
        new_adj[adj_size + idx, :adj_size] = new_adj_row
        new_adj[:adj_size, adj_size + idx] = new_adj_col.squeeze()
        new_adj[adj_size + idx, adj_size + idx] = torch.clamp(data.adj[i, i] + data.adj[j, j], min=0.0, max=1.0)
   
    return new_features, new_labels, new_train_idx, new_adj

def mix_emb(data, mixpool,embeddings,lam,args):
    num = args.mixnum
    new_train_idx = data.train_idx.clone()  

    gen_embeddings = []
    gen_features = []
    new_labels = []
    new_ids = []
    origin_nodes_pairs = []  

    current_max_id = data.num_nodes 

    for _ in range(num):
        if args.select == 'information':
            i, j = mixpool[_]
        elif  args.select == 'topo':
            top_indices = mixpool["top"]
            bottom_indices = mixpool["bottom"]
            i = np.random.choice(top_indices, 1)[0]
            j = np.random.choice(bottom_indices, 1)[0]
        else:
            i, j = np.random.choice(mixpool, 2, replace=False)

        # lam = 0.5

        new_embedding = lam * embeddings[i] + (1 - lam) * embeddings[j]
        new_label = lam * data.labels[i] + (1 - lam) * data.labels[j]
        new_feature = lam * data.features[i] + (1 - lam) * data.features[j]

        gen_embeddings.append(new_embedding.unsqueeze(0))
        gen_features.append(new_feature.unsqueeze(0))
        new_labels.append(new_label.unsqueeze(0))
        new_ids.append(current_max_id)
        origin_nodes_pairs.append((i, j))  

        current_max_id += 1

    new_labels = torch.cat(new_labels, dim=0)
    gen_embeddings = torch.cat(gen_embeddings, dim=0)
    gen_features = torch.cat(gen_features, dim=0)

    new_embeddings = torch.cat([embeddings, gen_embeddings], dim=0)
    new_features = torch.cat([data.features, gen_features], dim=0)
    # new_features = F.dropout(new_features,args.mix_dropout)
    new_labels = torch.cat([data.labels, new_labels], dim=0)
    new_train_idx = torch.cat([new_train_idx, torch.tensor(new_ids)], dim=0)
    
    adj_size = data.adj.shape[0]
    add_num = len(new_ids)
    new_adj = torch.zeros((adj_size + add_num, adj_size + add_num), dtype=data.adj.dtype, device=embeddings.device)

    new_adj[:adj_size, :adj_size] = data.adj

    for idx, (i, j) in enumerate(origin_nodes_pairs):
        new_adj_row = torch.clamp(data.adj[i, :] + data.adj[j, :], min=0.0, max=1.0).unsqueeze(0)
        new_adj_col = torch.clamp(data.adj[:, i] + data.adj[:, j], min=0.0, max=1.0).unsqueeze(1)
        
        new_adj[adj_size + idx, :adj_size] = new_adj_row
        new_adj[:adj_size, adj_size + idx] = new_adj_col.squeeze()
        new_adj[adj_size + idx, adj_size + idx] = torch.clamp(data.adj[i, i] + data.adj[j, j], min=0.0, max=1.0)
   
    return new_embeddings, new_features, new_labels, new_train_idx, new_adj

class Mixer(nn.Module):
    def __init__(self, feature_dim, embed_dim, dropout):
        super(Mixer, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.translator = nn.Linear(feature_dim,embed_dim,bias=False)
        
    def forward(self,features):
        emb = self.translator(features)
        emb = F.dropout(emb,self.dropout,training=self.training)
        
        return emb




