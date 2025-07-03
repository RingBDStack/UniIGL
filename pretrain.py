import os
from layers.mygcn import GCN
import torch
import torch.nn.functional as F
import random
from model.structure import adjust_structure
from model.cal import calculate_rc_and_sc

class PreTrainer:
    def __init__(self, config, data, path1, path2, device ='cuda'):
        self.config = config
        self.data = data
        self.model_path = path1
        self.structure_path = path2
        self.device = device
        self.new_adj_norm = data.adj_norm
        self.new_adj = data.adj
        
    def train(self):
        self.set_model()
    
        cosine_datasets = {'Photo', 'Chameleon', 'Squirrel', 'Actor'}
        use_cosine = getattr(self.data, 'data_name', None) in cosine_datasets
        
        if os.path.exists(self.model_path) and os.path.exists(self.structure_path):
            print("Model and structure files exist. Loading parameters...")
            self.model.load_pretrained_weights(self.model_path)
            self.load_structure(self.structure_path)
            return self.model, self.new_adj, self.new_adj_norm
        else:
            print("Model file not found. Starting training...")
            
            for i in range(self.config.iters):
                # print(f"Structure Refinement Iteration {iter+1}/{self.config.iters}") 
                loss_pre = 9999    
                for epoch in range(self.config.pre_epochs):
                    self.model.train()
                    self.optimizer.zero_grad()
                    
                    triplets = sample_triplets_with_multi_neg(self.data, self.config.imbtype, num_triplets=self.config.num_triplets)
                    
                    if self.config.adj == 'norm':
                        emb = self.model(self.data.features, self.new_adj_norm)
                    else:
                        emb = self.model(self.data.features, self.new_adj)
                        
                    loss = linkpred_multi_neg_loss(emb, triplets, tau=self.config.tau1, use_cosine=use_cosine)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    print(f"Epoch {epoch+1}, LinkPred Loss: {loss.item():.6f}")
                    if epoch % 5 == 0:
                        if loss.item() < loss_pre:
                            loss_pre = loss.item()
                            break_epoch = epoch
                            
                        if epoch > self.config.least_Pre_epochs and epoch - break_epoch > self.config.early_stop:
                            print(f'Pre-Training Finished! Stop at {epoch+1}th epoch!')
                            # print(f"Model saved at epoch {epoch+1} with best validation accuracy: {best_acc:.4f}")
                            break   
                    
                scores = calculate_rc_and_sc(self.new_adj.cpu(), self.data.train_idx)
                with torch.no_grad():
                    # self.model.eval()
                    emb = self.model(self.data.features, self.new_adj_norm)
                new_adj = adjust_structure(scores, emb.cpu(), self.new_adj.cpu(), 
                                           self.config.k, self.config.n, self.config.m).to('cuda')
                self.new_adj = new_adj
                self.new_adj_norm = normalize_adj_torch(new_adj)
                # print(f"Structure adjusted after iteration {iter+1}")
            
            loss_pre = 9999
            for epoch in range(self.config.pre_epochs):
                # loss_pre = 9999
                self.model.train()
                self.optimizer.zero_grad()
                
                triplets = sample_triplets_with_multi_neg(self.data, self.config.imbtype, num_triplets=self.config.num_triplets)
                
                if self.config.adj == 'norm':
                    emb = self.model(self.data.features, self.new_adj_norm)
                else:
                    emb = self.model(self.data.features, self.new_adj)
                
                loss = linkpred_multi_neg_loss(emb, triplets, tau=self.config.tau1, use_cosine=use_cosine)
                
                loss.backward()
                self.optimizer.step()
                
                print(f"Epoch {epoch+1}, LinkPred Loss: {loss.item():.6f}")
                # if epoch % 5 == 0:
                #     if loss.item() < loss_pre:
                #         loss_pre = loss.item()
                #         break_epoch = epoch
                        
                #     if epoch > self.config.least_Pre_epochs and epoch - break_epoch > self.config.early_stop:
                #         print(f'Pre-Training Finished! Stop at {epoch+1}th epoch!')
                #         # print(f"Model saved at epoch {epoch+1} with best validation accuracy: {best_acc:.4f}")
                #         break
                    
            
            torch.save(self.model.state_dict(), self.model_path)
            self.save_structure(self.structure_path)
            print(f"Model saved to {self.model_path}, Structure saved to {self.structure_path}")
            self.model.load_pretrained_weights(self.model_path)
            return self.model, self.new_adj, self.new_adj_norm         
            
            
    def set_model(self):
        self.model = GCN(self.data.features.shape[1], self.config.nhid, self.data.nclass, 
                         self.config.nlayer, self.config.dropout)
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.pre_lr, weight_decay=self.config.weight_decay)
        
    def save_structure(self, path):
            structure = {
                'new_adj': self.new_adj.cpu(),  
                'new_adj_norm': self.new_adj_norm.cpu() if self.new_adj_norm is not None else None
            }
            torch.save(structure, path)
        
    def load_structure(self, path):
            structure = torch.load(path)
            self.new_adj = structure['new_adj'].to(self.device)
            self.new_adj_norm = structure['new_adj_norm'].to(self.device) if structure['new_adj_norm'] is not None else None
        
def get_neighbors(adj, node):
    neighbors = (adj[node] > 0).nonzero().view(-1).tolist()
    return neighbors

def sample_triplets_with_multi_neg(data, imbtype ,num_triplets=2000, num_neg=3):
    if data.data_name == 'CiteSeer':
        num_neg = 3
    elif data.data_name == 'Actor'and imbtype=='topology'  :
        num_neg = 3
        
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

def linkpred_multi_neg_loss_pre(emb, triplets, tau=0.1):
    # emb = F.normalize(emb, p=2, dim=-1)  
    v_idx = torch.tensor([t[0] for t in triplets], device=emb.device)
    a_idx = torch.tensor([t[1] for t in triplets], device=emb.device)
    neg_idx = torch.tensor([t[2] for t in triplets], device=emb.device)  # [num_triplets, num_neg]
    
    s_v = emb[v_idx]  # [num_triplets, emb_dim]
    s_a = emb[a_idx]  # [num_triplets, emb_dim]
    s_neg = emb[neg_idx]  # [num_triplets, num_neg, emb_dim]
    
    sim_va = (s_v * s_a).sum(dim=-1) / tau  # [num_triplets]
    sim_vneg = torch.bmm(s_v.unsqueeze(1), s_neg.transpose(1, 2)).squeeze(1) / tau  # [num_triplets, num_neg]
    
    numerator = torch.exp(sim_va)
    denominator = numerator + torch.exp(sim_vneg).sum(dim=-1)
    loss = -torch.log(numerator / (denominator + 1e-8))
    return loss.mean()

def linkpred_multi_neg_loss(emb,
                            triplets,
                            tau: float = 0.1,
                            use_cosine: bool = False):
    """
    use_cosine=True  →  cosine similarity / tau
    use_cosine=False →  dot product / tau     (consistent with old implementation)
    """
    v_idx  = torch.tensor([t[0] for t in triplets], device=emb.device)
    a_idx  = torch.tensor([t[1] for t in triplets], device=emb.device)
    neg_idx = torch.tensor([t[2] for t in triplets], device=emb.device)  # [B, K]

    z_v   = emb[v_idx]               # [B, d]
    z_a   = emb[a_idx]               # [B, d]
    z_neg = emb[neg_idx]             # [B, K, d]

    if use_cosine:
        # —— Cosine Similarity —— #
        sim_va   = F.cosine_similarity(z_v, z_a, dim=-1) / tau          # [B]
        sim_vneg = F.cosine_similarity(
            z_v.unsqueeze(1).expand_as(z_neg), z_neg, dim=-1
        ) / tau                                                         # [B, K]
    else:
        # —— Dot Product —— #
        sim_va   = (z_v * z_a).sum(dim=-1) / tau                        # [B]
        sim_vneg = torch.bmm(
            z_v.unsqueeze(1), z_neg.transpose(1, 2)
        ).squeeze(1) / tau                                              # [B, K]

    numerator   = torch.exp(sim_va)                     # [B]
    denominator = numerator + torch.exp(sim_vneg).sum(dim=-1)
    loss = -torch.log(numerator / (denominator + 1e-8))
    return loss.mean()


def cosine_sim(x, y):
    return F.cosine_similarity(x, y, dim=-1)

def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx