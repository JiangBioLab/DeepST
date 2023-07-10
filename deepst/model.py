#!/usr/bin/env python
"""
# Author: ChangXu
# Created Time : Mon 23 Apr 2021 08:26:32 PM CST
# File Name: model.py
# Description:`
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import Sequential, BatchNorm, InstanceNorm
from typing import Callable, Iterable, Union, Tuple, Optional
import logging

class DeepST_model(nn.Module):
    def __init__(self, 
                input_dim, 
                Conv_type = 'GCNConv',
                linear_encoder_hidden = [32,20],
                linear_decoder_hidden = [32],
                conv_hidden = [32,8],
                p_drop = 0.01,
                dec_cluster_n = 15,
                alpha = 0.9,
                activate="relu",
                ):
        super(DeepST_model, self).__init__()
        self.input_dim = input_dim
        self.Conv_type = Conv_type
        self.alpha = alpha
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
            current_encoder_dim = linear_encoder_hidden[le]
        current_decoder_dim = linear_encoder_hidden[-1] + conv_hidden[-1]

        self.decoder = nn.Sequential()
        for ld in range(len(linear_decoder_hidden)):
            self.decoder.add_module(f'decoder_L{ld}',
                                    buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate, self.p_drop))
            current_decoder_dim= self.linear_decoder_hidden[ld]
        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}', nn.Linear(self.linear_decoder_hidden[-1], 
                                self.input_dim))

        #### a variational graph autoencoder based on pytorch geometric
        '''https://pytorch-geometric.readthedocs.io/en/latest/index.html'''

        # GCN layers
        if self.Conv_type == "GCNConv":
            '''https://arxiv.org/abs/1609.02907'''
            from torch_geometric.nn import GCNConv
            self.conv = Sequential('x, edge_index', [
                        (GCNConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "SAGEConv":
            from torch_geometric.nn import SAGEConv
            self.conv = Sequential('x, edge_index', [
                        (SAGEConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (SAGEConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (SAGEConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "GraphConv":
            from torch_geometric.nn import GraphConv
            self.conv = Sequential('x, edge_index', [
                        (GraphConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (GraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (GraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "GatedGraphConv":
            from torch_geometric.nn import GatedGraphConv
            self.conv = Sequential('x, edge_index', [
                        (GatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (GatedGraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (GatedGraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "ResGatedGraphConv":
            from torch_geometric.nn import ResGatedGraphConv
            self.conv = Sequential('x, edge_index', [
                        (ResGatedGraphConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "TransformerConv":
            from torch_geometric.nn import TransformerConv
            self.conv = Sequential('x, edge_index', [
                        (TransformerConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (TransformerConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (TransformerConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "TAGConv":
            from torch_geometric.nn import TAGConv
            self.conv = Sequential('x, edge_index', [
                        (TAGConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (TAGConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (TAGConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "ARMAConv":
            from torch_geometric.nn import ARMAConv
            self.conv = Sequential('x, edge_index', [
                        (ARMAConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (ARMAConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (ARMAConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
        elif self.Conv_type == "SGConv":
            from torch_geometric.nn import SGConv
            self.conv = Sequential('x, edge_index', [
                        (SGConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (SGConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (SGConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])     
        elif self.Conv_type == "MFConv":
            from torch_geometric.nn import MFConv
            self.conv = Sequential('x, edge_index', [
                        (MFConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (MFConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (MFConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ]) 
        elif self.Conv_type == "RGCNConv":
            from torch_geometric.nn import RGCNConv
            self.conv = Sequential('x, edge_index', [
                        (RGCNConv(linear_encoder_hidden[-1], conv_hidden[0]* 2, num_relations=3), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (RGCNConv(conv_hidden[0]* 2, conv_hidden[-1], num_relations=3), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (RGCNConv(conv_hidden[0]* 2, conv_hidden[-1], num_relations=3), 'x, edge_index -> x1'),
                        ]) 
        elif self.Conv_type == "FeaStConv":
            from torch_geometric.nn import FeaStConv
            self.conv = Sequential('x, edge_index', [
                        (FeaStConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (FeaStConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (FeaStConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ]) 
        elif self.Conv_type == "LEConv":
            from torch_geometric.nn import LEConv
            self.conv = Sequential('x, edge_index', [
                        (LEConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (LEConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (LEConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ]) 
        elif self.Conv_type == "ClusterGCNConv":
            from torch_geometric.nn import ClusterGCNConv
            self.conv = Sequential('x, edge_index', [
                        (ClusterGCNConv(linear_encoder_hidden[-1], conv_hidden[0]* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden[0]* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.conv_mean = Sequential('x, edge_index', [
                        (ClusterGCNConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ])
            self.conv_logvar = Sequential('x, edge_index', [
                        (ClusterGCNConv(conv_hidden[0]* 2, conv_hidden[-1]), 'x, edge_index -> x1'),
                        ]) 
        self.dc = InnerProductDecoder(p_drop)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.linear_encoder_hidden[-1] + self.conv_hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(
        self, 
        x, 
        adj,
        ):
        feat_x = self.encoder(x)
        conv_x = self.conv(feat_x, adj)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj), feat_x

    def reparameterize(
        self, 
        mu, 
        logvar,
        ):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def target_distribution(
        self, 
        target
        ):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def deepst_loss(
        self, 
        decoded, 
        x, 
        preds, 
        labels, 
        mu, 
        logvar, 
        n_nodes, 
        norm, 
        mask=None, 
        mse_weight=10, 
        bce_kld_weight=0.1,
        ):
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

    def forward(
        self, 
        x, 
        adj,
        ):
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)

        #
        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z
            

def buildNetwork(
    in_features, 
    out_features, 
    activate = "relu", 
    p_drop = 0.0
    ):
    net = []
    net.append(nn.Linear(in_features, out_features))
    net.append(BatchNorm(out_features, momentum=0.01, eps=0.001))
    if activate=="relu":
        net.append(nn.ELU())
    elif activate=="sigmoid":
        net.append(nn.Sigmoid())
    if p_drop > 0:
        net.append(nn.Dropout(p_drop))
    return nn.Sequential(*net) 


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(
        self, 
        dropout, 
        act=torch.sigmoid,
        ):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(
        self, 
        z,
        ):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t())) 
        return adj 

class GradientReverseLayer(torch.autograd.Function):
    """Layer that reverses and scales gradients before
    passing them up to earlier ops in the computation graph
    during backpropogation.
    """

    @staticmethod
    def forward(ctx, x, weight):
        """
        Perform a no-op forward pass that stores a weight for later
        gradient scaling during backprop.
        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, Features]
        weight : float
            weight for scaling gradients during backpropogation.
            stored in the "context" ctx variable.
        Notes
        -----
        We subclass `Function` and use only @staticmethod as specified
        in the newstyle pytorch autograd functions.
        https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
        We define a "context" ctx of the class that will hold any values
        passed during forward for use in the backward pass.
        `x.view_as(x)` and `*1` are necessary so that `GradReverse`
        is actually called
        `torch.autograd` tries to optimize backprop and
        excludes no-ops, so we have to trick it :)
        """
        # store the weight we'll use in backward in the context
        ctx.weight = weight
        return x.view_as(x) * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        """Return gradients
        Returns
        -------
        rev_grad : torch.FloatTensor
            reversed gradients scaled by `weight` passed in `.forward()`
        None : None
            a dummy "gradient" required since we passed a weight float
            in `.forward()`.
        """
        # here scale the gradient and multiply by -1
        # to reverse the gradients
        return (grad_output * -1 * ctx.weight), None


class AdversarialNetwork(nn.Module):
    """Build a Graph Convolutional Adversarial Network 
       for semi-supervised Domain Adaptation. """

    def __init__(
        self,
        model,
        n_domains: int = 2,
        weight: float = 1,
        n_layers: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        model : ExtractDEF
            cell type classification model.
        n_domains : int
            number of domains to adapt.
        weight : float
            weight for reversed gradients.
        n_layers : int
            number of hidden layers in the network.

        Returns
        -------
        None.
        """
        super(AdversarialNetwork, self).__init__()
        self.model = model
        self.n_domains = n_domains
        self.n_layers = n_layers
        self.weight = weight

        hidden_layers = [
            nn.Linear(self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1], 
                        self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1]),
            nn.ReLU(),
        ] * n_layers

        self.domain_clf = nn.Sequential(
            *hidden_layers,
            nn.Linear(self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1], self.n_domains),
        )

        return

    def set_rev_grad_weight(
        self,
        weight: float,
    ) -> None:
        """Set the weight term used after reversing gradients"""
        self.weight = weight
        return
        
    def target_distribution(
        self, 
        target
        ):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def deepst_loss(
        self, 
        decoded, 
        x, 
        preds, 
        labels, 
        mu, 
        logvar, 
        n_nodes, 
        norm, 
        mask=None, 
        mse_weight=10, 
        bce_kld_weight=0.1,
        ):
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
        return mse_weight * mse_loss + bce_logits_loss + bce_kld_weight*KLD

    def forward(
        self,
        x: torch.FloatTensor,
        edge_index,
    ) -> torch.FloatTensor:
        """Perform a forward pass.

        Parameters
        ----------
        x : torch.FloatTensor
            [Batch, Features] input.

        Returns
        -------
        domain_pred : torch.FloatTensor
            [Batch, n_domains] logits.
        x_embed : torch.FloatTensor
            [Batch, n_hidden]
        """
        # reverse gradients and scale by a weight
        # domain_pred -> x_rev -> GradientReverseLayer -> x_embed
        #      d+     ->  d+   ->     d-      ->   d-
        z, mu, logvar, de_feat, q, feat_x, gnn_z = self.model(x, edge_index)
        x_rev = GradientReverseLayer.apply(
            z,
            self.weight,
        )
        # classify the domains
        domain_pred = self.domain_clf(x_rev)
        return z, mu, logvar, de_feat, q, feat_x, gnn_z, domain_pred

