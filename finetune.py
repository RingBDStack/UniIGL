import torch
import torch.nn.functional as F
from layers.mygcn import Prompt_module, GCN, Topology_Prompt
from utils.metric import acc,mf1,wf1,bacc,auroc
from model.structure import GraphLearner2
import copy

class Tuner:
    def __init__(self, config, data, model_path, ori_data, device='cuda'):
        self.config = config
        self.data = data
        self.ori_data = ori_data
        self.model_path = model_path
        self.device = device
        self.mixnum = self.config.mixnum
        self.now_adj = self.data.adj_norm
        self.set_model()
        
    def set_model(self):
        self.model = GCN(self.data.features.shape[1], self.config.nhid, self.data.labels.shape[1], 
                         self.config.nlayer, self.config.dropout)
        
        self.model.load_pretrained_weights(self.model_path)
        self.model = self.model.to(self.device)
            
        self.optimizer_gnn = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, 
                                              weight_decay=self.config.weight_decay)
        
        self.prompt = Prompt_module(self.config.nhid)
        self.optimizer_pmt = torch.optim.Adam(self.prompt.parameters(), lr=self.config.pmt_lr, 
                                              weight_decay=self.config.pmt_weight_decay)
        self.prompt = self.prompt.to(self.device)
        
        self.topo_pmt = Topology_Prompt(self.config.mixnum, self.data.features.shape[0]-self.config.mixnum)
        self.optimizer_topo = torch.optim.Adam(self.topo_pmt.parameters(), lr=self.config.topo_lr, 
                                              weight_decay=self.config.topo_weight_decay)
        self.topo_pmt = self.topo_pmt.to(self.device)
        
        self.start_ratio = getattr(self.config, 'start_ratio', 0.05)
        self.end_ratio = getattr(self.config, 'end_ratio', 0.95)
        self.anneal_epochs = getattr(self.config, 'epochs', 200)
        
        
    def run_epoch(self):
        self.model.train()
        self.prompt.train()
        
        self.optimizer_gnn.zero_grad()
        self.optimizer_pmt.zero_grad()
        self.optimizer_topo.zero_grad()
        
        # if self.current_epoch == 0:
        #     with torch.no_grad():
        #         node_embeddings = self.model(self.data.features, self.data.adj_norm)
        #         self.node_embeddings = F.dropout(node_embeddings, p=self.config.gsl_dropout)

        progress = min(self.current_epoch / self.anneal_epochs, 1.0)
        current_ratio = self.start_ratio + 0.5 * (self.end_ratio - self.start_ratio) * \
            (1 - torch.cos(torch.tensor(progress * torch.pi)).item())
        # current_ratio = self.start_ratio + progress * (self.end_ratio - self.start_ratio)
        
        loss_all = 0
        for i in range(self.config.iters2):
          
            new_adj = self.topo_pmt(self.now_adj)
                
            new_adj = self.config.graph_skip_conn * self.data.adj_norm + (1 - self.config.graph_skip_conn) * new_adj
            self.now_adj = current_ratio * new_adj + (1 - current_ratio) * self.now_adj.detach()

            node_embeddings = self.model(self.data.features, self.now_adj)
            self.node_embeddings = F.dropout(node_embeddings,p=self.config.gsl_dropout)
            node_embeddings = self.prompt(node_embeddings)
            
            output, train_labels = self.prompt.compute_prototype_logits(
                                node_embeddings, self.data.labels, self.data.train_idx, self.config.tau2)
            
            train_loss = self.prompt.compute_prototype_loss(output, train_labels)
            loss_all += train_loss
            
        loss_all /= self.config.iters2
        loss_all.backward(retain_graph=True)
        
        self.optimizer_gnn.step()
        self.optimizer_pmt.step()
        self.optimizer_topo.step()
        
        return loss_all.item()
    
    def train(self):
        min_loss = 9999
        best_acc = 0
        for epoch in range(self.config.epochs):
            torch.cuda.empty_cache()
            self.current_epoch = epoch
            train_loss = self.run_epoch()
            print(f'Epoch {epoch+1}, Train Loss: {train_loss}')
            
            acc_val = self.val()
            
            if self.config.metric == 'acc':
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_epoch = epoch
                    best_model = copy.deepcopy(self.model)
                    best_prompt = copy.deepcopy(self.prompt)
                    
            elif self.config.metric == 'loss':
                if train_loss < min_loss:
                    min_loss = train_loss
                    best_epoch = epoch
                    best_model = copy.deepcopy(self.model)
                    best_prompt = copy.deepcopy(self.prompt)

            if epoch - best_epoch > self.config.early_stop and epoch >self.config.least_epoch:
                print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
                break
            
        return best_model, best_prompt
            
    def val(self):
        self.model.eval()
        self.prompt.eval()
        
        with torch.no_grad():
            node_embeddings = self.model(self.ori_data.features, self.ori_data.adj_norm)
            node_embeddings = self.prompt(node_embeddings)
            prototypes = self.prompt.compute_prototype_logits(
                            node_embeddings, self.ori_data.labels, self.ori_data.train_idx, self.config.tau2,
                            return_prototypes=True)
            val_embeddings = node_embeddings[self.ori_data.val_idx]   
            
            val_sim = F.cosine_similarity(
                val_embeddings.unsqueeze(1),
                prototypes.unsqueeze(0),
                dim=-1
            ) / self.config.tau2
            
            pred = torch.softmax(val_sim, dim=-1)
            labels_val = torch.argmax(self.ori_data.labels[self.ori_data.val_idx], dim=-1)

            acc_val = acc(labels_val, pred)
            mf1_val = mf1(labels_val, pred)
            print(f'ACC VAL: {acc_val:.4f} || MF1 VAL: {mf1_val:.4f}')
            return (acc_val+mf1_val) /2    
        
            