import scipy.sparse as sp
import numpy as np
import torch
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon,Actor,WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset

import networkx as nx

class New_Data:
    def __init__(self, adj, adj_norm, features, labels, train_idx, val_idx, test_idx):
        self.adj_norm = adj_norm
        self.adj = adj
        self.features = features
        self.labels = labels
        self.train_idx = train_idx
        # self.nclass = labels.numpy().max().item() + 1
        self.val_idx = val_idx
        self.test_idx =test_idx
        
    def to(self, device): 
        self.adj_norm = self.adj_norm.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.adj = self.adj.to(device)
        return self  

class Data:
    def __init__(self, adj,adj_norm, features, labels, edge_index,num_nodes, args):
        self.edge_index = edge_index
        self.adj_norm = adj_norm
        self.adj = adj
        self.features = features
        self.labels = labels
        self.args = args
        self.num_nodes = num_nodes
        self.nclass = labels.numpy().max().item() + 1
        
    def get_topo_imb_split(self,target_data,shuffle_seed):
        all_idx = [i for i in range(target_data.num_nodes)]
        all_label = target_data.labels.numpy()
        nclass = target_data.labels.numpy().max().item() + 1
        random.seed(shuffle_seed)
        random.shuffle(all_idx) 
        
        train_each = (len(all_idx)*0.1) // nclass
        valid_each = (len(all_idx)*0.1) // nclass
        
        # train_each = 5
        # valid_each = 5
        train_list = [0 for _ in range(nclass)]
        train_node = [[] for _ in range(nclass)]
        train_idx  = []
        
        for iter1 in all_idx:
            iter_label = all_label[iter1]
            if train_list[iter_label] < train_each:
                train_list[iter_label]+=1
                train_node[iter_label].append(iter1)
                train_idx.append(iter1)

            if sum(train_list)==train_each*nclass:break
        assert sum(train_list)==train_each*nclass
        after_train_idx = list(set(all_idx)-set(train_idx))
        random.shuffle(after_train_idx)
        
        valid_list = [0 for _ in range(nclass)]
        valid_idx  = []
        for iter2 in after_train_idx:
            iter_label = all_label[iter2]
            if valid_list[iter_label] < valid_each:
                valid_list[iter_label]+=1
                valid_idx.append(iter2)
            if sum(valid_list)==valid_each*nclass:break

        assert sum(valid_list)==valid_each*nclass
        test_idx = list(set(after_train_idx)-set(valid_idx))
        
        labeled_node = [[] for _ in range(nclass)]
        for iter in train_idx:
            iter_label = all_label[iter]
            labeled_node[iter_label].append(iter)
        
        target_data.train_idx = torch.LongTensor(train_idx)
        target_data.val_idx = torch.LongTensor(valid_idx)
        target_data.test_idx = torch.LongTensor(test_idx)
        target_data.labeled_nodes = labeled_node
        
        return target_data
    
    def get_classnum_imb_split(self, target_data, imb_ratio,shuffle_seed):
        random.seed(shuffle_seed)
        total_items = target_data.labels.shape[0]
        shuffled_indices = list(range(total_items))
        random.shuffle(shuffled_indices)
        
        class_num_list, indices, inv_indices = sort(data=target_data)
        
        total_items = target_data.labels.shape[0]
        n_classes = target_data.labels.max().item() + 1
        n_train = int(total_items * 0.1)
        n_val = int(total_items * 0.1)
        n_test = total_items - n_train - n_val
        
        class_num_list_train = split_lt(
            class_num_list=class_num_list,
            indices=indices,
            inv_indices=inv_indices,
            imb_ratio=imb_ratio,
            n_cls=n_classes,
            n=n_train
        )

        n_val_per_class = n_val // n_classes
        class_num_list_val = torch.full((n_classes,), n_val_per_class, dtype=torch.long)
        remainder = n_val % n_classes
        if remainder > 0:
            class_num_list_val[:remainder] += 1

        shuffled_indices = torch.tensor(shuffled_indices, dtype=torch.long)

        train_indices = []
        val_indices = []
        test_indices = []
        labeled_node = [[] for _ in range(n_classes)]

        for cls in range(n_classes):
            class_mask = (target_data.labels[shuffled_indices] == cls)
            class_indices = shuffled_indices[class_mask]
            num_samples_in_class = len(class_indices)
            n_train_samples = class_num_list_train[cls].item()
            n_train_samples = min(n_train_samples, num_samples_in_class)
            train_class_indices = class_indices[:n_train_samples]
            train_indices.extend(train_class_indices.tolist())
            labeled_node[cls].extend(train_class_indices.tolist())

            remaining_class_indices = class_indices[n_train_samples:]
            num_remaining = len(remaining_class_indices)

            n_val_samples = class_num_list_val[cls].item()
            n_val_samples = min(n_val_samples, num_remaining)
            val_class_indices = remaining_class_indices[:n_val_samples]
            val_indices.extend(val_class_indices.tolist())

            test_class_indices = remaining_class_indices[n_val_samples:]
            test_indices.extend(test_class_indices.tolist())

        train_indices = torch.tensor(train_indices, dtype=torch.long)
        val_indices = torch.tensor(val_indices, dtype=torch.long)
        test_indices = torch.tensor(test_indices, dtype=torch.long)

        target_data.labeled_nodes = labeled_node
        target_data.train_idx = train_indices
        target_data.val_idx = val_indices
        target_data.test_idx = test_indices

        return target_data  
    
    def to(self, device): 
        # self.edge_index = self.edge_index.to(device)
        self.adj_norm = self.adj_norm.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.adj = self.adj.to(device)
        return self    

def load_processed_data(args):
    data_name = args.dataset
    data_path = "./dataset/{}".format(data_name)
    data_dict = {'Cora':'planetoid','CiteSeer':'planetoid','PubMed':'planetoid',
                'Photo':'amazon','Computers':'amazon','Actor':'Actor',
                'Chameleon':'WikipediaNetwork','Squirrel':'WikipediaNetwork','arxiv':'ogbn'}    
    target_type = data_dict[data_name]
    if target_type == 'amazon':
        target_dataset = Amazon(data_path, name=data_name)
    elif target_type == 'planetoid':
        target_dataset = Planetoid(data_path, name=data_name)
    elif target_type == 'WikipediaNetwork':
         target_dataset = WikipediaNetwork(root=data_path, name=data_name, geom_gcn_preprocess=True)    
    elif target_type == 'Actor':
        target_dataset = Actor(data_path)
    elif data_name == 'arxiv':
        target_dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    
    target_data=target_dataset[0]
    features = target_data.x
    if data_name in ['Cora',"CiteSeer"]:
        features = normalize_features(features)
        features = torch.FloatTensor(np.array(features))
    
    labels = target_data.y
    adj = index2dense(target_data.edge_index,target_data.num_nodes)
    adj = nx.adjacency_matrix(nx.from_numpy_array(adj))
    adj = adj + sp.eye(adj.shape[0])
    adj_norm = normalize_sparse_adj(adj)
    adj_norm = torch.Tensor(adj_norm.todense())
    adj = torch.Tensor(adj.todense())

    data = Data(adj,adj_norm, features, labels, target_data.edge_index,\
                target_data.num_nodes,args)
    
    if(args.imbtype=='topology'):
        data = data.get_topo_imb_split(data,args.shuffle_seed)
    elif(args.imbtype=='classnum'):
        data = data.get_classnum_imb_split(data,args.imb_ratio,args.shuffle_seed)
        
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[data.train_idx] = True
    
    data.data_name = data_name
        
    return data

def index2dense(edge_index, nnode):
    idx = edge_index.numpy()
    adj = np.zeros((nnode,nnode))
    adj[(idx[0], idx[1])] = 1
    sum = np.sum(adj)

    return adj

def normalize_sparse_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx

def sort(data, data_mask=None):
    if data_mask is None:
        y = data.labels
    else:
        y = data.labels[data_mask]
    
    n_cls = data.labels.max().item() + 1

    class_num_list = []
    for i in range(n_cls):
        class_num_list.append(int((y == i).sum().item()))
    
    class_num_list_tensor = torch.tensor(class_num_list)
    class_num_list_sorted_tensor, indices = torch.sort(class_num_list_tensor, descending=True)
    inv_indices = torch.zeros(n_cls, dtype=indices.dtype, device=indices.device)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i

    assert torch.equal(class_num_list_sorted_tensor, class_num_list_tensor[indices])
    assert torch.equal(class_num_list_tensor, class_num_list_sorted_tensor[inv_indices])
    return class_num_list_tensor, indices, inv_indices

def choose(class_num_list_train, data, keep=0):
    node_mask = torch.zeros(data.labels.shape[0], dtype=torch.bool, device=data.labels.device)    
    classes = torch.unique(data.labels)
    
    for i in classes:
        idx = torch.nonzero(data.labels == i, as_tuple=False).squeeze()
        idx = idx[torch.randperm(len(idx))]
        n_samples = max(class_num_list_train[i], keep)
        n_samples = min(n_samples, len(idx))
        
        selected_idx = idx[:n_samples]        
        node_mask[selected_idx] = True
    
    return node_mask

def split_lt(class_num_list, indices, inv_indices, imb_ratio, n_cls, n, keep=0):
    class_num_list = class_num_list[indices]  # sort
    mu = np.power(imb_ratio, 1 / (n_cls - 1))
    _mu = 1 / mu
    if imb_ratio == 1:
        n_max = n / n_cls
    else:
        n_max = n / (imb_ratio * mu - 1) * (mu - 1) * imb_ratio
    class_num_list_lt = []
    for i in range(n_cls):
        class_num_list_lt.append(round(min(max(n_max * np.power(_mu, i), 1), class_num_list[i].item() - keep)))
    class_num_list_lt = torch.tensor(class_num_list_lt)
    return class_num_list_lt[inv_indices]  # unsort


def split_same(class_num_list,n_cls, n, keep=0):
    class_num_list_lt = []
    for i in range(n_cls):
        class_num_list_lt.append(min(round(n / n_cls), class_num_list[i].item() - keep))
    class_num_list_lt = torch.tensor(class_num_list_lt)
    return class_num_list_lt
