#!/usr/bin/env python
"""
# Author: Chang Xu
# Created Time : Mon 23 Apr 2021 08:26:32 PM CST
# File Name: STMAP_train.py
# Description:`
"""

"""
Test:

from cal_graph import graph
from GCN import STMAP_model
import scanpy as sc
import torch 

data_path = "/home/xuchang/Project/STMAP/Human_breast/output/Breast_data/STMAP_Breast_15.h5ad"
adata = sc.read(data_path)
graph_dict = graph(adata.obsm['spatial'], distType='euclidean', k=10).main()
sc.pp.filter_genes(adata, min_cells=5)
adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
adata_X = sc.pp.scale(adata_X)
adata_X = sc.pp.pca(adata_X, n_comps=200)
stmap = STMAP_model(input_dim = adata_X.shape[1], 
                        Conv_type='GCNConv',
                        linear_encoder_hidden=[50,12],
                        linear_decoder_hidden=[50,70],
                        conv_hidden=[32,16],
                        p_drop=0.1,
                        dec_cluster_n=20,)
adata_X = torch.FloatTensor(adata_X.copy())
adj = graph_dict['adj_norm']
z, mu, logvar, de_feat, q, feat_x, gnn_z = stmap(adata_X, adj)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import Sequential

class STMAP_model(nn.Module):
    def __init__(self, 
                input_dim, 
                Conv_type='GCNConv',
                linear_encoder_hidden=[50,20],
                linear_decoder_hidden=[50,60],
                conv_hidden=[32,8],
                p_drop=0,
                dec_cluster_n=20,
                activate="relu",
                ):
        super(STMAP_model, self).__init__()
        self.input_dim = input_dim
        self.Conv_type = Conv_type
        self.alpha = 0.8
        self.conv_hidden = conv_hidden
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop
        self.dec_cluster_n = dec_cluster_n

        current_encoder_dim = self.input_dim
        ### a deep autoencoder network
        self.encoder = nn.Sequential()
        for le in range(len(linear_encoder_hidden)):
            self.encoder.add_module(f'encoder_L{le}', 
                                    buildNetwork(current_encoder_dim, self.linear_encoder_hidden[le], self.activate, self.p_drop))
            current_encoder_dim=linear_encoder_hidden[le]
        current_decoder_dim = linear_encoder_hidden[-1] + conv_hidden[-1]

        # self.combined_encoder = nn.Sequential()
        # self.combined_encoder.add_module("c_e_1", buildNetwork(current_decoder_dim, current_decoder_dim, self.activate, self.p_drop))
        # self.combined_encoder.add_module("c_e_2", buildNetwork(current_decoder_dim, current_decoder_dim, self.activate, self.p_drop))
        # self.combined_encoder.add_module("c_e_3", buildNetwork(current_decoder_dim, current_decoder_dim, self.activate, self.p_drop))      

        self.decoder = nn.Sequential()
        for ld in range(len(linear_decoder_hidden)):
            self.decoder.add_module(f'decoder_L{ld}',
                                    buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate, self.p_drop))
            current_decoder_dim= self.linear_decoder_hidden[ld]
        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}',buildNetwork(self.linear_decoder_hidden[-1], 
                                self.input_dim, "sigmoid", p_drop))

        #### a variational graph autoencoder based on pytorch geometric
        '''https://pytorch-geometric.readthedocs.io/en/latest/index.html'''

        # GCN layers
        if self.Conv_type == "GCNConv":
            '''https://arxiv.org/abs/1609.02907'''
            from torch_geometric.nn import GCNConv
            self.conv = Sequential('x, edge_index', [
                        (GCNConv(linear_encoder_hidden[-1], conv_hidden[0]), 'x, edge_index -> x1'),
                        nn.ReLU(inplace=True), 
                        (GCNConv(conv_hidden[0], conv_hidden[0]), 'x1, edge_index -> x2'),
                        nn.ReLU(inplace=True),
                        (GCNConv(conv_hidden[0], conv_hidden[0]), 'x2, edge_index -> x3'),
                        nn.ReLU(inplace=True),
                        (GCNConv(conv_hidden[0], conv_hidden[0]), 'x3, edge_index -> x4'),
                        nn.ReLU(inplace=True),])
            self.conv_mean = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden[0], conv_hidden[-1]), 'x, edge_index -> x1'),
                        nn.ReLU(inplace=True), 
                        (GCNConv(conv_hidden[-1], conv_hidden[-1]), 'x1, edge_index -> x2'),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden[0], conv_hidden[-1]), 'x, edge_index -> x1'),
                        nn.ReLU(inplace=True), 
                        (GCNConv(conv_hidden[-1], conv_hidden[-1]), 'x1, edge_index -> x2'),
                        nn.ReLU(inplace=True), 
                        ])
   
        elif self.Conv_type == "GMMConv":
            from torch_geometric.nn import GMMConv
            self.conv = Sequential('x, edge_index', [
                        (GMMConv(linear_encoder_hidden[-1], conv_hidden[0], dim=20, kernel_size=15), 'x, edge_index -> x'),
                        nn.ReLU(inplace=True), 
                        (GMMConv(conv_hidden[0], conv_hidden[0], dim=20, kernel_size=15), 'x, edge_index -> x'),
                        nn.ReLU(inplace=True),
                        (GMMConv(conv_hidden[0], conv_hidden[0], dim=20, kernel_size=15), 'x, edge_index -> x'),
                        nn.ReLU(inplace=True),
                        (GMMConv(conv_hidden[0], conv_hidden[0], dim=20, kernel_size=15), 'x, edge_index -> x'),
                        nn.ReLU(inplace=True),])
            self.conv_mean = GMMConv(conv_hidden[0], conv_hidden[-1], dim=20, kernel_size=15)
            self.conv_logvar = GMMConv(conv_hidden[0], conv_hidden[-1], dim=20, kernel_size=15)
            # self.conv = Sequential('x, edge_index, batch', [
            #             (Dropout(p=0.5), 'x -> x'),
            #             (GCNConv(dataset.num_features, 64), 'x, edge_index -> x1'),
            #             ReLU(inplace=True),
            #             (GCNConv(64, 64), 'x1, edge_index -> x2'),
            #             ReLU(inplace=True),
            #             (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
            #             (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
            #             (global_mean_pool, 'x, batch -> x'),
            #             Linear(2 * 64, dataset.num_classes),])
            # self.conv = GCNConv(linear_encoder_hidden[-1], conv_hidden[0])


            # self.conv = GraphConvolution(linear_encoder_hidden[-1], conv_hidden[0], self.p_drop, act=F.relu)
            # self.conv_mean = GraphConvolution(conv_hidden[0], conv_hidden[-1], self.p_drop, act=lambda x: x)
            # self.conv_logvar = GraphConvolution(conv_hidden[0], conv_hidden[-1], self.p_drop, act=lambda x: x)
            
        elif self.Conv_type == "GATConv":
            from torch_geometric.nn import GATConv
            self.conv = GATConv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = GATConv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = GATConv(conv_hidden[0], conv_hidden[-1])

        elif self.Conv_type == "GATv2Conv":
            from torch_geometric.nn import GATv2Conv
            self.conv = GATv2Conv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = GATv2Conv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = GATv2Conv(conv_hidden[0], conv_hidden[-1]) 

        elif self.Conv_type == "SAGEConv":
            from torch_geometric.nn import SAGEConv
            self.conv = SAGEConv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = SAGEConv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = SAGEConv(conv_hidden[0], conv_hidden[-1]) 

        elif self.Conv_type == "GraphConv":
            from torch_geometric.nn import GraphConv
            self.conv = GraphConv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = GraphConv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = GraphConv(conv_hidden[0], conv_hidden[-1])

        elif self.Conv_type == "GatedGraphConv":
            from torch_geometric.nn import GatedGraphConv
            self.conv = GatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = GatedGraphConv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = GatedGraphConv(conv_hidden[0], conv_hidden[-1])

        elif self.Conv_type == "ResGatedGraphConv":
            from torch_geometric.nn import ResGatedGraphConv
            self.conv = ResGatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = ResGatedGraphConv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = ResGatedGraphConv(conv_hidden[0], conv_hidden[-1])

        elif self.Conv_type == "TransformerConv":
            from torch_geometric.nn import TransformerConv
            self.conv = TransformerConv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = TransformerConv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = TransformerConv(conv_hidden[0], conv_hidden[-1])

        elif self.Conv_type == "TAGConv":
            from torch_geometric.nn import TAGConv
            self.conv = TAGConv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = TAGConv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = TAGConv(conv_hidden[0], conv_hidden[-1])

        elif self.Conv_type == "ARMAConv":
            from torch_geometric.nn import ARMAConv
            self.conv = ARMAConv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = ARMAConv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = ARMAConv(conv_hidden[0], conv_hidden[-1])  
 
        elif self.Conv_type == "SGConv":
            from torch_geometric.nn import SGConv
            self.conv = SGConv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = SGConv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = SGConv(conv_hidden[0], conv_hidden[-1])       

        elif self.Conv_type == "MFConv":
            from torch_geometric.nn import MFConv
            self.conv = MFConv(linear_encoder_hidden[-1], conv_hidden[0])
            self.conv_mean = MFConv(conv_hidden[0], conv_hidden[-1])
            self.conv_logvar = MFConv(conv_hidden[0], conv_hidden[-1]) 


        self.dc = InnerProductDecoder(p_drop)

        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.linear_encoder_hidden[-1]+self.conv_hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        conv_x = self.conv(feat_x, adj)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj), feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def target_distribution(self, target):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def stmap_loss(self, decoded, x, preds, 
                    labels, mu, logvar, n_nodes, 
                    norm, mask=None, mse_weight=10, bce_kld_weight=0.1):
        mse_fun = torch.nn.MSELoss()
        mse_loss = mse_fun(decoded, x)

        if mask is not None:
            preds = preds * mask
            labels = labels * mask

        bce_logits_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
              1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return mse_weight * mse_loss + bce_kld_weight* (bce_logits_loss + KLD)

    def forward(self, x, adj):
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z= torch.cat((feat_x, gnn_z), 1)
        # z = self.combined_encoder(combined_z)
        de_feat = self.decoder(z)

        # DEC clustering
        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z
            

def buildNetwork(in_features, out_features, activate="relu", p_drop=0.0):
    net = []
    net.append(nn.Linear(in_features, out_features))
    net.append(nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001))
    if activate=="relu":
        net.append(nn.ELU())
    elif activate=="sigmoid":
        net.append(nn.Sigmoid())
    if p_drop > 0:
        net.append(nn.Dropout(p_drop))
    return nn.Sequential(*net) 


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t())) 
        return adj 



