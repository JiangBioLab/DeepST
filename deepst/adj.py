#!/usr/bin/env python
"""
# Author: ChangXu
# Created Time : Mon 23 Apr
# File Name: cal_graph.py
# Description:`

"""
"""
test:
    from cal_graph import graph, combine_graph_dict
    import scanpy as sc
    data_path = "/home/xuchang/Project/STMAP/Human_breast/output/Breast_data/STMAP_Breast_15.h5ad"
    adata = sc.read(data_path)
    graph_cons = graph(adata.obsm['spatial'], distType='euclidean', k=10)
    graph_dict = graph_cons.main()

"""

import os,sys
import numpy as np
import torch
from scipy import stats
import scipy.sparse as sp
from scipy.spatial import distance
from torch_sparse import SparseTensor
import networkx as nx


##### refer to https://github.com/mustafaCoskunAgu/SiGraC/blob/main/DGI/utils/process.py


# edgeList to edgeDict
class graph():
    def __init__(self, 
                 data, 
                 rad_cutoff,
                 k,  
                 distType='euclidean',):
        super(graph, self).__init__()
        self.data = data
        self.distType = distType
        self.k = k
        self.rad_cutoff = rad_cutoff
        self.num_cell = data.shape[0]

        
    def graph_computing(self):
        """
        Input: -adata.obsm['spatial']
               -distanceType:
                    -if get more information, https://docs.scipy.org/doc/scipy/reference/generated/scipy.
                     spatial.distance.cdist.html#scipy.spatial.distance.cdist
               -k: number of neighbors
        Return: graphList
        """
        dist_list = ["euclidean","braycurtis","canberra","mahalanobis","chebyshev","cosine",
                    "jensenshannon","mahalanobis","minkowski","seuclidean","sqeuclidean","hamming",
                    "jaccard", "jensenshannon", "kulsinski", "mahalanobis","matching", "minkowski", 
                    "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
                    "sqeuclidean", "wminkowski", "yule"]

        if self.distType == 'spearmanr':
            SpearA, _= stats.spearmanr(self.data, axis=1)
            graphList = []
            for node_idx in range(self.data.shape[0]):
                tmp = SpearA[node_idx, :].reshape(1, -1)
                res = tmp.argsort()[0][-(self.k+1):]
                for j in np.arange(0, self.k):
                    graphList.append((node_idx, res[j]))

        elif self.distType == "BallTree":
            from sklearn.neighbors import BallTree
            tree = BallTree(self.data)
            dist, ind = tree.query(self.data, k=self.k+1)
            indices = ind[:, 1:]
            graphList=[]
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))

        elif self.distType == "KDTree":
            from sklearn.neighbors import KDTree
            tree = KDTree(self.data)
            dist, ind = tree.query(self.data, k=self.k+1)
            indices = ind[:, 1:]
            graphList=[]
            for node_idx in range(self.data.shape[0]):
                for j in np.arange(0, indices.shape[1]):
                    graphList.append((node_idx, indices[node_idx][j]))

        elif self.distType == "kneighbors_graph":
            from sklearn.neighbors import kneighbors_graph
            A = kneighbors_graph(self.data, n_neighbors=self.k, mode='connectivity', include_self=False)
            A = A.toarray()
            graphList=[]
            for node_idx in range(self.data.shape[0]):
                indices = np.where(A[node_idx] == 1)[0]
                for j in np.arange(0, len(indices)):
                    graphList.append((node_idx, indices[j])) 

        elif self.distType == "Radius":
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(radius = self.rad_cutoff).fit(self.data)
            distances, indices = nbrs.radius_neighbors(self.data, return_distance=True)
            graphList=[]
            for node_idx in range(indices.shape[0]):
                for j in range(indices[node_idx].shape[0]):
                    if distances[node_idx][j] > 0 :
                        graphList.append((node_idx, indices[node_idx][j]))
            print('%.4f neighbors per cell on average.' %(len(graphList)/self.data.shape[0]))

        elif self.distType in dist_list:
            graphList = []
            for node_idx in range(self.data.shape[0]):
                tmp = self.data[node_idx, :].reshape(1, -1)
                distMat = distance.cdist(tmp, self.data, self.distType)
                res = distMat.argsort()[:self.k + 1]
                tmpdist = distMat[0, res[0][1:self.k + 1]]
                boundary = np.mean(tmpdist) + np.std(tmpdist)
                for j in np.arange(1, self.k+1):
                    if distMat[0, res[0][j]] <= boundary:
                        graphList.append((node_idx, res[0][j]))
                    else:
                        pass

        else: 
            raise ValueError(
                f"""\
                {self.distType!r} does not support. Disttype must in {dist_list} """)
        
        return graphList

    def List2Dict(self, graphList):
        """
        Return dict: eg {0: [0, 3542, 2329, 1059, 397, 2121, 485, 3099, 904, 3602],
                     1: [1, 692, 2334, 1617, 1502, 1885, 3106, 586, 3363, 101],
                     2: [2, 1849, 3024, 2280, 580, 1714, 3311, 255, 993, 2629],...}
        """
        graphdict = {}
        tdict = {}
        for graph in graphList:
            end1 = graph[0]
            end2 = graph[1]
            tdict[end1] = ""
            tdict[end2] = ""
            if end1 in graphdict:
                tmplist = graphdict[end1]
            else:
                tmplist = []
            tmplist.append(end2)
            graphdict[end1] = tmplist

        for i in range(self.num_cell):
            if i not in tdict:
                graphdict[i] = []

        return graphdict

    def mx2SparseTensor(self, mx):

        """Convert a scipy sparse matrix to a torch SparseTensor."""
        mx = mx.tocoo().astype(np.float32)
        row = torch.from_numpy(mx.row).to(torch.long)
        col = torch.from_numpy(mx.col).to(torch.long)
        values = torch.from_numpy(mx.data)
        adj = SparseTensor(row=row, col=col, \
                           value=values, sparse_sizes=mx.shape)
        adj_ = adj.t()
        return adj_

    def pre_graph(self, adj):
        
        """ Graph preprocessing."""
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
             
        return self.mx2SparseTensor(adj_normalized)

    def main(self):
        adj_mtx = self.graph_computing()
        graphdict = self.List2Dict(adj_mtx)
        adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

        """ Store original adjacency matrix (without diagonal entries) for later """
        adj_pre = adj_org
        adj_pre = adj_pre - sp.dia_matrix((adj_pre.diagonal()[np.newaxis, :], [0]), shape=adj_pre.shape)
        adj_pre.eliminate_zeros()

        """ Some preprocessing."""
        adj_norm = self.pre_graph(adj_pre)
        adj_label = adj_pre + sp.eye(adj_pre.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())
        norm = adj_pre.shape[0] * adj_pre.shape[0] / float((adj_pre.shape[0] * adj_pre.shape[0] - adj_pre.sum()) * 2)

        graph_dict = {
                     "adj_norm": adj_norm,
                     "adj_label": adj_label,
                     "norm_value": norm }

        return graph_dict


def combine_graph_dict(dict_1, dict_2):

    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
    
    graph_dict = {
        "adj_norm": SparseTensor.from_dense(tmp_adj_norm),
        "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label']),
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])}
    return graph_dict