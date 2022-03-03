#!/usr/bin/env python
"""
# Author: *******
# Created Time : Mon 23 Apr 2021 08:26:32 PM CST
# File Name: DeepST_train.py
# Description:`

"""
"""
test:
from cal_graph import graph
import scanpy as sc
from GCN import STMAP_model
from STMAP_main import train
adata = sc.read(data_path)
graph_dict = graph(adata.obsm['spatial'], distType='euclidean', k=10).main()
sc.pp.filter_genes(adata, min_cells=5)
adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
adata_X = sc.pp.scale(adata_X)
adata_X = sc.pp.pca(adata_X, n_comps=200)
stmap = STMAP_model(input_dim = adata_X.shape[1], 
                        Conv_type='GCNConv',
                        linear_encoder_hidden=[100,20],
                        linear_decoder_hidden=[50,70],
                        conv_hidden=[32,16,8],
                        p_drop=0.1,
                        dec_cluster_n=20,)

train(adata_X, graph_dict, stmap, pre_epochs=200, epochs=200).fit()

"""

import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from torch.autograd import Variable
from sknetwork.clustering import Louvain
from sklearn.cluster import SpectralClustering, KMeans
from tqdm import tqdm


class train:
    def __init__(self,
                processed_data,
                graph_dict,
                model,
                pre_epochs,
                epochs,
                corrupt=0.01,
                ):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.processed_data = processed_data
        self.data = torch.FloatTensor(processed_data.copy()).to(self.device)
        self.adj = graph_dict['adj_norm'].to(self.device)
        self.adj_label = graph_dict['adj_label'].to(self.device)
        self.norm = graph_dict['norm_value']
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()),lr=0.0001, weight_decay=0.5)
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.num_spots = self.data.shape[0]
        self.dec_tol = 0
        self.kl_weight = 10
        self.q_stride = 10
        self.mse_weight = 10
        self.bce_kld_weight = 0.1
        self.corrupt = corrupt

    def pretrain(self, ):
        with tqdm(total=int(self.pre_epochs), 
                    desc="DeepST trains an initial model",
                        bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.pre_epochs):
                inputs_corr = masking_noise(self.data, self.corrupt)
                inputs_coor = inputs_corr.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                z, mu, logvar, de_feat, _, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
                loss = self.model.stmap_loss(decoded=de_feat, x=self.data, preds=self.model.dc(z), 
                                             labels=self.adj_label, mu=mu, logvar=logvar, n_nodes=self.num_spots, 
                                             norm=self.norm, mask=self.adj_label, mse_weight=self.mse_weight, 
                                             bce_kld_weight=self.bce_kld_weight)
                loss.backward()
                self.optimizer.step()
                pbar.update(1)

    def process(self,):
        self.model.eval()
        z, _, _, _, q, _, _ = self.model(self.data, self.adj)
        z = z.cpu().detach().numpy()
        q = q.cpu().detach().numpy()

        return z, q

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def fit(self, 
            cluster_n=20, 
            clusterType ='Louvain',
            res = 1.0):

        """
        load pretrain model for IDEC
        For specific methods, please refer to: https://github.com/IoannisStournaras/Deep-Learning-
                                                for-Deconvolution-of-scRNA-seq-Data    
        """
        self.pretrain()
        pre_z, _ = self.process()
        # z, _, _, _, _, _, _ = self.model(self.data, self.adj)
        # pre_z, _ = self.process()
        if clusterType == 'KMeans':
            cluster_method = KMeans(n_clusters= cluster_n, n_init= cluster_n, random_state=88)
            y_pred_last = np.copy(cluster_method.fit_predict(pre_z))
            self.model.cluster_layer.data = torch.tensor(cluster_method.cluster_centers_).to(self.device)
        elif clusterType == 'Louvain':
            cluster_data = sc.AnnData(pre_z)
            sc.pp.neighbors(cluster_data, n_neighbors=cluster_n)
            sc.tl.louvain(cluster_data, resolution = res)
            y_pred_last = cluster_data.obs['louvain'].astype(int).to_numpy()
            n_clusters = len(np.unique(y_pred_last))
            features = pd.DataFrame(pre_z,index=np.arange(0,pre_z.shape[0]))
            Group = pd.Series(y_pred_last,index=np.arange(0,features.shape[0]),name="Group")
            Mergefeature = pd.concat([features,Group],axis=1)
            cluster_centers_ = np.asarray(Mergefeature.groupby("Group").mean())
            self.model.cluster_layer.data = torch.tensor(cluster_centers_).to(self.device)
   
        self.model.train()
        with tqdm(total=int(self.pre_epochs), 
                    desc="DeepST trains a final model",
                        bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for epoch in range(self.epochs):

                if epoch % self.q_stride == 0:
                    _, q = self.process()
                    q = self.model.target_distribution(torch.Tensor(q).clone().detach())
                    y_pred = q.cpu().numpy().argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    self.model.train()
                    if epoch > 0 and delta_label < self.dec_tol:
                        print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                torch.set_grad_enabled(True)
                self.optimizer.zero_grad()
                inputs_corr = masking_noise(self.data, self.corrupt)
                inputs_coor = inputs_corr.to(self.device)
                z, mu, logvar, de_feat, out_q, feat_x, gnn_z = self.model(Variable(inputs_coor), self.adj)
                loss_stmap = self.model.stmap_loss(decoded=de_feat, x=self.data, preds=self.model.dc(z), 
                                                    labels=self.adj_label, mu=mu, logvar=logvar, n_nodes=self.num_spots, 
                                                    norm=self.norm, mask=self.adj_label, mse_weight=self.mse_weight, 
                                                    bce_kld_weight=self.bce_kld_weight)
                loss_kl = F.kl_div(out_q.log(), torch.tensor(q).to(self.device))
                loss = self.kl_weight * loss_kl + loss_stmap
                loss.backward()
                self.optimizer.step()
                pbar.update(1)
         
def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise
