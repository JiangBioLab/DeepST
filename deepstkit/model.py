#!/usr/bin/env python
"""
Deep Spatial Transcriptomics Model (DeepST)
Author: ChangXu
Created Time: Mon 23 Apr 2021 08:26:32 PM CST
Description: A deep learning model for spatial transcriptomics data analysis combining graph neural networks and autoencoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import Sequential, BatchNorm
from typing import Optional

class DeepST_model(nn.Module):
    def __init__(self, 
                input_dim: int,
                Conv_type: str = 'GATConv',
                linear_encoder_hidden: list = [32, 20],
                linear_decoder_hidden: list = [32],
                conv_hidden: list = [32, 8],
                p_drop: float = 0.01,
                dec_cluster_n: int = 15,
                alpha: float = 0.9,
                activate: str = "relu"):
        """
        DeepST Model Initialization
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features (number of genes)
        Conv_type : str
            Type of graph convolutional layer (default: 'GCNConv')
            Options: ['GCNConv', 'SAGEConv', 'GraphConv', 'GatedGraphConv', 
                     'ResGatedGraphConv', 'TransformerConv', 'TAGConv',
                     'ARMAConv', 'SGConv', 'MFConv', 'RGCNConv',
                     'FeaStConv', 'LEConv', 'ClusterGCNConv']
        linear_encoder_hidden : list
            List of hidden layer sizes for the encoder (default: [32, 20])
        linear_decoder_hidden : list
            List of hidden layer sizes for the decoder (default: [32])
        conv_hidden : list
            List of hidden layer sizes for the graph convolutional layers (default: [32, 8])
        p_drop : float
            Dropout probability (default: 0.01)
        dec_cluster_n : int
            Number of clusters for DEC (default: 15)
        alpha : float
            Parameter for student's t-distribution (default: 0.9)
        activate : str
            Activation function (default: 'relu')
            Options: ['relu', 'sigmoid']
        """
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

        # Build encoder network
        current_dim = self.input_dim
        self.encoder = nn.Sequential()
        for i, hidden_size in enumerate(linear_encoder_hidden):
            self.encoder.add_module(
                f'encoder_L{i}', 
                self._build_network(current_dim, hidden_size)
            )
            current_dim = hidden_size

        # Build decoder network
        current_dim = linear_encoder_hidden[-1] + conv_hidden[-1]
        self.decoder = nn.Sequential()
        for i, hidden_size in enumerate(linear_decoder_hidden):
            self.decoder.add_module(
                f'decoder_L{i}',
                self._build_network(current_dim, hidden_size)
            )
            current_dim = hidden_size
        
        # Final decoder layer
        self.decoder.add_module(
            'decoder_out', 
            nn.Linear(current_dim, self.input_dim)
        )

        # Build graph convolutional layers
        self._build_graph_conv_layers()
        
        # Inner product decoder for adjacency matrix
        self.dc = InnerProductDecoder(p_drop)
        
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(
            self.dec_cluster_n, 
            self.linear_encoder_hidden[-1] + self.conv_hidden[-1]
        ))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def _build_network(self, in_features: int, out_features: int) -> nn.Sequential:
        """Build a basic network block with linear, batchnorm, activation and dropout"""
        layers = [
            nn.Linear(in_features, out_features),
            BatchNorm(out_features, momentum=0.01, eps=0.001)
        ]
        
        if self.activate == "relu":
            layers.append(nn.ELU())
        elif self.activate == "sigmoid":
            layers.append(nn.Sigmoid())
            
        if self.p_drop > 0:
            layers.append(nn.Dropout(self.p_drop))
            
        return nn.Sequential(*layers)

    def _build_graph_conv_layers(self):
        """Build graph convolutional layers based on specified type"""
        conv_class = self._get_conv_class()
        
        # Shared initial graph convolution
        self.conv = Sequential('x, edge_index', [
            (conv_class(self.linear_encoder_hidden[-1], self.conv_hidden[0]*2), 
            'x, edge_index -> x'),
            BatchNorm(self.conv_hidden[0]*2),
            nn.ReLU(inplace=True), 
        ])
        
        # Mean and logvar branches
        self.conv_mean = Sequential('x, edge_index', [
            (conv_class(self.conv_hidden[0]*2, self.conv_hidden[-1]), 
            'x, edge_index -> x')
        ])
        
        self.conv_logvar = Sequential('x, edge_index', [
            (conv_class(self.conv_hidden[0]*2, self.conv_hidden[-1]), 
            'x, edge_index -> x')
        ])

    def _get_conv_class(self):
        """Get the appropriate graph convolution class"""
        conv_classes = {
            "GCNConv": self._import_conv_class("GCNConv"),
            "GATConv": self._import_conv_class("GATConv"),
            "SAGEConv": self._import_conv_class("SAGEConv"),
            "GraphConv": self._import_conv_class("GraphConv"),
            "GatedGraphConv": self._import_conv_class("GatedGraphConv"),
            "ResGatedGraphConv": self._import_conv_class("ResGatedGraphConv"),
            "TransformerConv": self._import_conv_class("TransformerConv"),
            "TAGConv": self._import_conv_class("TAGConv"),
            "ARMAConv": self._import_conv_class("ARMAConv"),
            "SGConv": self._import_conv_class("SGConv"),
            "MFConv": self._import_conv_class("MFConv"),
            "RGCNConv": self._import_conv_class("RGCNConv"),
            "FeaStConv": self._import_conv_class("FeaStConv"),
            "LEConv": self._import_conv_class("LEConv"),
            "ClusterGCNConv": self._import_conv_class("ClusterGCNConv")
        }
        return conv_classes[self.Conv_type]
    
    def _import_conv_class(self, class_name: str):
        """Dynamically import the required graph convolution class"""
        from torch_geometric.nn import __dict__ as geom_nn_dict
        return geom_nn_dict[class_name]

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        Encode input data through encoder and graph convolutional layers
        
        Parameters:
        -----------
        x : torch.Tensor
            Input feature matrix [n_nodes, input_dim]
        adj : torch.Tensor
            Adjacency matrix or edge index [2, n_edges]
            
        Returns:
        --------
        tuple: (mu, logvar, feat_x)
            mu : torch.Tensor
                Mean of latent distribution [n_nodes, conv_hidden[-1]]
            logvar : torch.Tensor
                Log variance of latent distribution [n_nodes, conv_hidden[-1]]
            feat_x : torch.Tensor
                Encoded features [n_nodes, linear_encoder_hidden[-1]]
        """
        feat_x = self.encoder(x)
        conv_x = self.conv(feat_x, adj)
        return self.conv_mean(conv_x, adj), self.conv_logvar(conv_x, adj), feat_x

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution
        
        Parameters:
        -----------
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
            
        Returns:
        --------
        torch.Tensor
            Sampled latent variables
        """
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu

    def target_distribution(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution for DEC clustering
        
        Parameters:
        -----------
        target : torch.Tensor
            Current soft cluster assignments
            
        Returns:
        --------
        torch.Tensor
            Target distribution
        """
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def deepst_loss(self, 
                   decoded: torch.Tensor, 
                   x: torch.Tensor, 
                   preds: torch.Tensor, 
                   labels: torch.Tensor, 
                   mu: torch.Tensor, 
                   logvar: torch.Tensor, 
                   n_nodes: int, 
                   norm: float, 
                   mask: Optional[torch.Tensor] = None, 
                   mse_weight: float = 10, 
                   bce_kld_weight: float = 0.1) -> torch.Tensor:
        """
        Compute DeepST loss function
        
        Parameters:
        -----------
        decoded : torch.Tensor
            Decoded/reconstructed features
        x : torch.Tensor
            Original input features
        preds : torch.Tensor
            Predicted adjacency matrix
        labels : torch.Tensor
            True adjacency matrix
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        n_nodes : int
            Number of nodes in graph
        norm : float
            Normalization factor
        mask : torch.Tensor, optional
            Mask for adjacency matrix
        mse_weight : float
            Weight for reconstruction loss
        bce_kld_weight : float
            Weight for BCE and KLD losses
            
        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        mse_loss = F.mse_loss(decoded, x)
        
        if mask is not None:
            preds = preds * mask
            labels = labels * mask

        bce_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)
        
        # KL divergence term
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        
        return mse_weight * mse_loss + bce_kld_weight * (bce_loss + KLD)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple:
        """
        Forward pass of DeepST model
        
        Parameters:
        -----------
        x : torch.Tensor
            Input feature matrix [n_nodes, input_dim]
        adj : torch.Tensor
            Adjacency matrix or edge index [2, n_edges]
            
        Returns:
        --------
        tuple: (z, mu, logvar, de_feat, q, feat_x, gnn_z)
            z : torch.Tensor
                Combined latent features [n_nodes, linear_encoder_hidden[-1] + conv_hidden[-1]]
            mu : torch.Tensor
                Mean of latent distribution
            logvar : torch.Tensor
                Log variance of latent distribution
            de_feat : torch.Tensor
                Decoded features [n_nodes, input_dim]
            q : torch.Tensor
                Soft cluster assignments [n_nodes, dec_cluster_n]
            feat_x : torch.Tensor
                Encoded features [n_nodes, linear_encoder_hidden[-1]]
            gnn_z : torch.Tensor
                Graph latent features [n_nodes, conv_hidden[-1]]
        """
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)

        # Compute soft cluster assignments
        q = 1.0 / ((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha) + 1e-8)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, mu, logvar, de_feat, q, feat_x, gnn_z


class InnerProductDecoder(nn.Module):
    """
    Inner Product Decoder for graph reconstruction
    
    Parameters:
    -----------
    dropout : float
        Dropout probability
    act : callable
        Activation function (default: torch.sigmoid)
    """
    def __init__(self, dropout: float, act: callable = torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct adjacency matrix from latent features
        
        Parameters:
        -----------
        z : torch.Tensor
            Latent features [n_nodes, n_features]
            
        Returns:
        --------
        torch.Tensor
            Reconstructed adjacency matrix [n_nodes, n_nodes]
        """
        z = F.dropout(z, self.dropout, training=self.training)
        return self.act(torch.mm(z, z.t()))


class GradientReverseLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training
    
    This layer reverses and scales gradients during backpropagation
    while performing a no-op during forward pass.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: float) -> torch.Tensor:
        """
        Forward pass (no operation)
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        weight : float
            Weight for gradient scaling
            
        Returns:
        --------
        torch.Tensor
            Same as input tensor
        """
        ctx.weight = weight
        return x.view_as(x) * 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass with gradient reversal
        
        Parameters:
        -----------
        grad_output : torch.Tensor
            Gradient from subsequent layers
            
        Returns:
        --------
        tuple: (rev_grad, None)
            rev_grad : torch.Tensor
                Reversed and scaled gradients
            None : 
                Placeholder for weight gradient
        """
        return (grad_output * -1 * ctx.weight), None


class AdversarialNetwork(nn.Module):
    """
    Adversarial Network for Domain Adaptation
    
    Parameters:
    -----------
    model : DeepST_model
        The base DeepST model
    n_domains : int
        Number of domains to adapt between (default: 2)
    weight : float
        Weight for gradient reversal (default: 1)
    n_layers : int
        Number of hidden layers (default: 2)
    """
    def __init__(self,
                 model: DeepST_model,
                 n_domains: int = 2,
                 weight: float = 1,
                 n_layers: int = 2):
        super(AdversarialNetwork, self).__init__()
        self.model = model
        self.n_domains = n_domains
        self.weight = weight
        self.n_layers = n_layers

        # Build hidden layers
        input_dim = self.model.linear_encoder_hidden[-1] + self.model.conv_hidden[-1]
        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.extend([
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
            ])

        # Build domain classifier
        self.domain_clf = nn.Sequential(
            *hidden_layers,
            nn.Linear(input_dim, self.n_domains),
        )

    def set_rev_grad_weight(self, weight: float) -> None:
        """Set the gradient reversal weight"""
        self.weight = weight

    def target_distribution(
        self, 
        target
        ):
        weight = (target ** 2) / torch.sum(target, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def deepst_loss(self, 
                   decoded: torch.Tensor, 
                   x: torch.Tensor, 
                   preds: torch.Tensor, 
                   labels: torch.Tensor, 
                   mu: torch.Tensor, 
                   logvar: torch.Tensor, 
                   n_nodes: int, 
                   norm: float, 
                   mask: Optional[torch.Tensor] = None, 
                   mse_weight: float = 10, 
                   bce_kld_weight: float = 0.1) -> torch.Tensor:
        """
        Compute DeepST loss function
        
        Parameters:
        -----------
        decoded : torch.Tensor
            Decoded/reconstructed features
        x : torch.Tensor
            Original input features
        preds : torch.Tensor
            Predicted adjacency matrix
        labels : torch.Tensor
            True adjacency matrix
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution
        n_nodes : int
            Number of nodes in graph
        norm : float
            Normalization factor
        mask : torch.Tensor, optional
            Mask for adjacency matrix
        mse_weight : float
            Weight for reconstruction loss
        bce_kld_weight : float
            Weight for BCE and KLD losses
            
        Returns:
        --------
        torch.Tensor
            Combined loss value
        """
        mse_loss = F.mse_loss(decoded, x)
        
        if mask is not None:
            preds = preds * mask
            labels = labels * mask

        bce_loss = norm * F.binary_cross_entropy_with_logits(preds, labels)
        
        # KL divergence term
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        
        return mse_weight * mse_loss + bce_kld_weight * (bce_loss + KLD)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        """
        Forward pass with domain classification
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features [n_nodes, input_dim]
        edge_index : torch.Tensor
            Graph edge indices [2, n_edges]
            
        Returns:
        --------
        tuple: (z, mu, logvar, de_feat, q, feat_x, gnn_z, domain_pred)
            All outputs from base model plus:
            domain_pred : torch.Tensor
                Domain classification logits [n_nodes, n_domains]
        """
        # Get base model outputs
        z, mu, logvar, de_feat, q, feat_x, gnn_z = self.model(x, edge_index)
        
        # Apply gradient reversal
        x_rev = GradientReverseLayer.apply(z, self.weight)
        
        # Domain classification
        domain_pred = self.domain_clf(x_rev)
        
        return z, mu, logvar, de_feat, q, feat_x, gnn_z, domain_pred