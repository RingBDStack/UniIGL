import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


#GCN layer
class GraphConvolution(Module):
    """
    refer to GraphSmote
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class GCN_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x
    
class Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Classifier, self).__init__()

        # self.mlp1 = nn.Linear(nhid, nhid)
        self.mlp2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp2.weight,std=0.05)

    def forward(self, x, adj):
        x = self.mlp2(x)

        return x
    
class Decoder(Module):
    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)


    def forward(self, node_embed):
        
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1,-2)))

        return adj_out

    
class GCN_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x
    
class SemanticLayer(nn.Module):

    def __init__(self, in_features, out_features, nheads, graph_mode=1):
        super(SemanticLayer, self).__init__()
        
        self.nheads = nheads
        self.graph_mode = graph_mode

        self.linear = nn.Linear(in_features, out_features, bias=False)
        if self.graph_mode == 0:
            self.classifier = nn.Linear(out_features, nheads, bias=False)
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.graph_mode == 1:
            self.classifier = nn.Linear(out_features, nheads, bias=False)
        else:
            self.classifier = nn.Linear(out_features, 1, bias=False)
            self.loss_fn = nn.MSELoss()

        self.layers = nn.ModuleList()
        self.atts = nn.ModuleList()
        self.convs_1 = nn.ModuleList()
        self.convs_2 = nn.ModuleList()

        for _ in range(nheads):
            self.layers.append(nn.Linear(in_features, int(out_features / nheads), bias=False))
            self.atts.append(nn.Linear(out_features*2, 1, bias=False))
            self.convs_1.append(GraphConvolution(out_features, int(out_features / 2)))
            self.convs_2.append(GraphConvolution(int(out_features / 2), out_features))

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)
        nn.init.normal_(self.classifier.weight)
        for linear in self.layers:
            nn.init.xavier_uniform_(linear.weight, gain=1.414)
        for att in self.atts:
            nn.init.xavier_uniform_(att.weight, gain=1.414)

    def forward(self, x, adj):

        h = self.linear(x)
        edge_list = adj.nonzero().t()
        edge_h = torch.cat((h[edge_list[0, :], :], h[edge_list[1, :], :]), dim=1)

        out_features = []
        self.out_descriptor = []

        for i in range(self.nheads):
            e = torch.sigmoid(self.atts[i](edge_h).squeeze(1))
            new_adj = torch.sparse.FloatTensor(edge_list, e, torch.Size([h.shape[0], h.shape[0]])).to_dense()

            features = torch.matmul(new_adj, x)
            out = self.layers[i](features)
            out_features.append(out)
            
            descriptor = torch.tanh(self.convs_1[i](h.detach(), new_adj))
            descriptor = torch.tanh(self.convs_2[i](descriptor, new_adj))
            self.out_descriptor.append(torch.mean(descriptor, dim=0, keepdim=True))

        out = torch.cat(tuple([rst for rst in out_features]), -1)

        return out
    
    def compute_semantic_loss(self):

        labels = [torch.ones(1)*i for i in range(self.nheads)]
        labels = torch.cat(tuple(labels), 0).long().to('cuda')

        factors_feature = torch.cat(tuple(self.out_descriptor), 0)
        pred = self.classifier(factors_feature)

        if self.graph_mode == 0:
            pred = nn.Softmax(dim=1)(pred)
            loss_sem = self.loss_fn(pred, labels)
        elif self.graph_mode == 1:
            loss_sem_list = []
            for i in range(self.nheads-1):
                for j in range(i+1, self.nheads):
                    loss_sem_list.append(torch.cosine_similarity(pred[i], pred[j], dim=0))
            loss_sem_list = torch.stack(loss_sem_list)
            loss_sem = torch.mean(loss_sem_list)
        else:
            loss_sem_list = []
            for i in range(self.nheads-1):
                for j in range(i+1, self.nheads):
                    loss_sem_list.append(self.loss_fn(pred[i], pred[j]))
            loss_sem_list = torch.stack(loss_sem_list)
            loss_sem = - torch.mean(loss_sem_list)    

        return loss_sem

class SEM_En2(nn.Module):
    
    def __init__(self, nfeat, nhid, nembed, dropout, nheads=4, graph_mode=1):
        super(SEM_En2, self).__init__()

        self.sem1 = SemanticLayer(nfeat, nhid, nheads=nheads, graph_mode=graph_mode)
        self.sem2 = SemanticLayer(nhid, nembed, nheads=nheads, graph_mode=graph_mode)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sem1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sem2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        sem_loss = self.sem1.compute_semantic_loss()
        sem_loss += self.sem2.compute_semantic_loss()

        return x, sem_loss

