#!/usr/bin/env python
"""
DeepST Model Trainer
Author: ChangXu
Created Time: Mon 23 Apr 2021 08:26:32 PM CST
Description: Training module for Deep Spatial Transcriptomics model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.cluster import KMeans


class train():
    def __init__(self,
                 processed_data: np.ndarray,
                 graph_dict: dict,
                 model: nn.Module,
                 pre_epochs: int = 200,
                 epochs: int = 200,
                 corrupt: float = 0.001,
                 lr: float = 5e-4,
                 weight_decay: float = 1e-4,
                 domains: np.ndarray = None,
                 kl_weight: float = 100,
                 mse_weight: float = 10,
                 bce_kld_weight: float = 0.1,
                 domain_weight: float = 1,
                 use_gpu: bool = True
                ):
        """
        Initialize DeepST Trainer
        
        Parameters:
        -----------
        processed_data : np.ndarray
            Processed input data [n_spots, n_features]
        graph_dict : dict
            Dictionary containing graph information with keys:
            - 'adj_norm': Normalized adjacency matrix
            - 'adj_label': Original adjacency labels
            - 'norm_value': Normalization factor
        model : nn.Module
            DeepST model instance
        pre_epochs : int
            Number of pretraining epochs (default: 200)
        epochs : int
            Number of training epochs (default: 200)
        corrupt : float
            Corruption rate for masking noise (default: 0.001)
        lr : float
            Learning rate (default: 5e-4)
        weight_decay : float
            Weight decay for optimizer (default: 1e-4)
        domains : np.ndarray, optional
            Domain labels for adversarial training (default: None)
        kl_weight : float
            Weight for KL divergence loss (default: 100)
        mse_weight : float
            Weight for MSE reconstruction loss (default: 10)
        bce_kld_weight : float
            Weight for BCE and KLD losses (default: 0.1)
        domain_weight : float
            Weight for domain classification loss (default: 1)
        use_gpu : bool
            Whether to use GPU if available (default: True)
        """
        # Device setup
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Data setup
        self.processed_data = processed_data
        self.data = torch.FloatTensor(processed_data).to(self.device)
        self.adj = graph_dict['adj_norm'].to(self.device)
        self.adj_label = graph_dict['adj_label'].to(self.device)
        self.norm = graph_dict['norm_value']
        self.num_spots = self.data.shape[0]
        
        # Model setup
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Training parameters
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.corrupt = corrupt
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight
        self.bce_kld_weight = bce_kld_weight
        self.domain_weight = domain_weight
        
        # Clustering parameters
        self.dec_tol = 0  # Tolerance for cluster assignment change
        self.q_stride = 20  # Interval for updating target distribution
        
        # Domain adaptation
        self.domains = None
        if domains is not None:
            self.domains = torch.from_numpy(domains).to(self.device)

    def pretrain(self, grad_clip: float = 5.0) -> None:
        """
        Pretrain the model without clustering loss
        
        Parameters:
        -----------
        grad_clip : float
            Maximum gradient norm for clipping (default: 5.0)
        """
        with tqdm(total=self.pre_epochs, 
                 desc="Pretraining initial model",
                 bar_format="{l_bar}{bar} [ time left: {remaining} ]", ncols=80, dynamic_ncols=True, leave=True) as pbar:
            
            for epoch in range(self.pre_epochs):
                # Add masking noise for robustness
                inputs_corr = self._masking_noise(self.data, self.corrupt)
                
                # Forward pass
                self.model.train()
                self.optimizer.zero_grad()
                
                if self.domains is None:
                    z, mu, logvar, de_feat, _, _, _ = self.model(inputs_corr, self.adj)
                    preds = self.model.dc(z)
                else:
                    z, mu, logvar, de_feat, _, _, _, domain_pred = self.model(inputs_corr, self.adj)
                    preds = self.model.model.dc(z)
                
                # Compute loss
                loss = self.model.deepst_loss(
                            decoded=de_feat,
                            x=self.data,
                            preds=preds,
                            labels=self.adj_label,
                            mu=mu,
                            logvar=logvar,
                            n_nodes=self.num_spots,
                            norm=self.norm,
                            mask=self.adj_label,
                            mse_weight=self.mse_weight,
                            bce_kld_weight=self.bce_kld_weight
                            )
                
                # Add domain loss if applicable
                if self.domains is not None:
                    domain_loss = F.cross_entropy(domain_pred, self.domains)
                    loss += self.domain_weight * domain_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
                pbar.update(1)

    @torch.no_grad()
    def process(self) -> tuple:
        """
        Process data through trained model
        
        Returns:
        --------
        tuple: (z, q)
            z : np.ndarray
                Latent representation [n_spots, n_features]
            q : np.ndarray
                Soft cluster assignments [n_spots, n_clusters]
        """
        self.model.eval()
        if self.domains is None:
            z, _, _, _, q, _, _ = self.model(self.data, self.adj)
        else:
            z, _, _, _, q, _, _, _ = self.model(self.data, self.adj)
            
        return z.cpu().numpy(), q.cpu().numpy()

    def save_model(self, save_path: str) -> None:
        """
        Save model state to file
        
        Parameters:
        -----------
        save_path : str
            Path to save model state
        """
        torch.save({'state_dict': self.model.state_dict()}, save_path)
        print(f'Saved model to {save_path}')

    def load_model(self, load_path: str) -> None:
        """
        Load model state from file
        
        Parameters:
        -----------
        load_path : str
            Path to load model state from
        """
        state_dict = torch.load(load_path)
        self.model.load_state_dict(state_dict['state_dict'])
        print(f'Loaded model from {load_path}')

    def fit(self, 
            cluster_n: int = 20,
            cluster_type: str = 'Louvain',
            resolution: float = 1.0,
            pretrain: bool = True) -> np.ndarray:
        """
        Train the full DeepST model with clustering
        
        Parameters:
        -----------
        cluster_n : int
            Number of clusters (default: 20)
        cluster_type : str
            Clustering method ['Louvain', 'KMeans'] (default: 'Louvain')
        resolution : float
            Resolution parameter for Louvain clustering (default: 1.0)
        pretrain : bool
            Whether to run pretraining (default: True)
            
        Returns:
        --------
        np.ndarray
            Final cluster assignments [n_spots]
        """
        # Pretrain if specified
        if pretrain:
            self.pretrain()
            pre_z, _ = self.process()
            
        # Initialize cluster centers
        y_pred_last = self._initialize_clusters(pre_z, cluster_n, cluster_type, resolution)
        
        # Main training loop
        with tqdm(total=self.epochs,
                  desc="Training final model",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]", ncols=80, dynamic_ncols=True, leave=True) as pbar:
            
            for epoch in range(self.epochs):
                # Update target distribution periodically
                if epoch % self.q_stride == 0:
                    _, q = self.process()
                    q = self.model.target_distribution(torch.Tensor(q).to(self.device))
                    y_pred = q.cpu().numpy().argmax(1)
                    
                    # Check for convergence
                    delta_label = np.sum(y_pred != y_pred_last) / y_pred.shape[0]
                    y_pred_last = y_pred.copy()
                    
                    if epoch > 0 and delta_label < self.dec_tol:
                        print(f'delta_label {delta_label:.4f} < tol {self.dec_tol}')
                        print('Reached tolerance threshold. Stopping training.')
                        break
                
                # Training step
                self._train_step(q)
                pbar.update(1)
        
        # Return final cluster assignments
        _, q = self.process()
        return q.argmax(1)

    def _train_step(self, q: torch.Tensor) -> None:
        """Perform a single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        if self.domains is None:
            z, mu, logvar, de_feat, out_q, _, _ = self.model(self.data, self.adj)
            preds = self.model.dc(z)
        else:
            z, mu, logvar, de_feat, out_q, _, _, domain_pred = self.model(self.data, self.adj)
            preds = self.model.model.dc(z)
        
        # Compute losses
        loss_deepst = self.model.deepst_loss(
            decoded = de_feat,
            x = self.data,
            preds = preds,
            labels = self.adj_label,
            mu = mu,
            logvar = logvar,
            n_nodes = self.num_spots,
            norm = self.norm,
            mask = self.adj_label,
            mse_weight = self.mse_weight,
            bce_kld_weight = self.bce_kld_weight
        )
        
        loss_kl = F.kl_div(out_q.log(), q, reduction='batchmean')
        
        # Combine losses
        if self.domains is None:
            loss = self.kl_weight * loss_kl + loss_deepst
        else:
            domain_loss = F.cross_entropy(domain_pred, self.domains)
            loss = self.kl_weight * loss_kl + loss_deepst + self.domain_weight * domain_loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

    def _initialize_clusters(self, 
                           z: np.ndarray,
                           cluster_n: int,
                           cluster_type: str,
                           resolution: float) -> np.ndarray:
        """
        Initialize cluster centers using specified method
        
        Returns:
        --------
        np.ndarray
            Initial cluster assignments
        """
        if cluster_type == 'KMeans':
            kmeans = KMeans(n_clusters=cluster_n, n_init=cluster_n*2, random_state=88)
            y_pred = kmeans.fit_predict(z)
            centers = kmeans.cluster_centers_
        elif cluster_type == 'Louvain':
            cluster_data = sc.AnnData(z)
            sc.pp.neighbors(cluster_data, n_neighbors=cluster_n)
            sc.tl.louvain(cluster_data, resolution=resolution)
            y_pred = cluster_data.obs['louvain'].astype(int).to_numpy()
            
            # Calculate cluster centers
            features = pd.DataFrame(z, index=np.arange(z.shape[0]))
            groups = pd.Series(y_pred, index=np.arange(z.shape[0]), name="Group")
            merged = pd.concat([features, groups], axis=1)
            centers = np.asarray(merged.groupby("Group").mean())
        
        # Update cluster layer
        target_model = self.model if self.domains is None else self.model.model
        target_model.cluster_layer.data = torch.tensor(centers).to(self.device)
        
        return y_pred

    @staticmethod
    def _masking_noise(data: torch.Tensor, frac: float) -> torch.Tensor:
        """
        Apply masking noise to input data
        
        Parameters:
        -----------
        data : torch.Tensor
            Input data tensor
        frac : float
            Fraction of elements to mask
            
        Returns:
        --------
        torch.Tensor
            Noisy data tensor
        """
        data_noise = data.clone()
        mask = torch.rand(data.size()) < frac
        data_noise[mask.to(data.device)] = 0
        return data_noise