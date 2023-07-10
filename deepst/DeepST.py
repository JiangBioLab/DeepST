#!/usr/bin/env python3

# @Author: ChangXu
# @E-mail: xuchang0214@163.com
# @Last Modified by:   ChangXu
# @Last Modified time: 2021-04-22 08:42:54 23:22:34
# -*- coding: utf-8 -*-


import os
import psutil
import time
import torch
import math
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata
from pathlib import Path
from sklearn.metrics import pairwise_distances, calinski_harabasz_score
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import distance

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import Union, Callable

from utils_func import *
from his_feat import image_feature, image_crop
from adj import graph, combine_graph_dict
from model import DeepST_model, AdversarialNetwork
from trainer import train

from augment import augment_adata


class run():
	def __init__(
		self,
		save_path="./",
		task = "Identify_Domain",
		pre_epochs=1000, 
		epochs=500,
		use_gpu = True,
		):
		self.save_path = save_path
		self.pre_epochs = pre_epochs
		self.epochs = epochs
		self.use_gpu = use_gpu
		self.task = task

	def _get_adata(
		self,
		platform, 
		data_path,
		data_name,
		verbose = True,
		):
		assert platform in ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
		if platform in ['Visium', 'ST']:
			if platform == 'Visium':
				adata = read_10X_Visium(os.path.join(data_path, data_name))
			else:
				adata = ReadOldST(os.path.join(data_path, data_name))
		elif platform == 'MERFISH':
			adata = read_merfish(os.path.join(data_path, data_name))
		elif platform == 'slideSeq':
			adata = read_SlideSeq(os.path.join(data_path, data_name))
		elif platform == 'seqFish':
			adata = read_seqfish(os.path.join(data_path, data_name))
		elif platform == 'stereoSeq':
			adata = read_stereoSeq(os.path.join(data_path, data_name))
		else:
			raise ValueError(
               				 f"""\
               				 {self.platform!r} does not support.
	                				""")
		if verbose:
			save_data_path = Path(os.path.join(self.save_path, "Data", data_name))
			save_data_path.mkdir(parents=True, exist_ok=True)
			adata.write(os.path.join(save_data_path, f'{data_name}_raw.h5ad'), compression="gzip")
		return adata

	def _get_image_crop(
		self,
		adata,
		data_name,
		cnnType = 'ResNet50',
		pca_n_comps = 50, 
		):
		save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', data_name))
		save_path_image_crop.mkdir(parents=True, exist_ok=True)
		adata = image_crop(adata, save_path=save_path_image_crop)
		adata = image_feature(adata, pca_components = pca_n_comps, cnnType = cnnType).extract_image_feat()
		return adata

	def _get_augment(
		self,
		adata,
		adjacent_weight = 0.3,
		neighbour_k = 4,
		spatial_k = 30,
		n_components = 100,
		md_dist_type="cosine",
		gb_dist_type="correlation",
		use_morphological = True,
		use_data = "raw",
		spatial_type = "KDTree"
		):
		adata = augment_adata(adata, 
				md_dist_type = md_dist_type,
				gb_dist_type = gb_dist_type,
				n_components = n_components,
				use_morphological = use_morphological,
				use_data = use_data,
				neighbour_k = neighbour_k,
				adjacent_weight = adjacent_weight,
				spatial_k = spatial_k,
				spatial_type = spatial_type
				)
		print("Step 1: Augment molecule expression is Done!")
		return adata

	def _get_graph(
		self,
		data,
		distType = "BallTree",
		k = 12,
		rad_cutoff = 150,
		):
		graph_dict = graph(data, distType = distType, k = k, rad_cutoff = rad_cutoff).main()
		print("Step 2: Graph computing is Done!")
		return graph_dict

	def _optimize_cluster(
		self,
		adata,
		resolution: list = list(np.arange(0.1, 2.5, 0.01)),
		):
		scores = []
		for r in resolution:
			sc.tl.leiden(adata, resolution=r)
			s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
			scores.append(s)
		cl_opt_df = pd.DataFrame({"resolution": resolution, "score": scores})
		best_idx = np.argmax(cl_opt_df["score"])
		res = cl_opt_df.iloc[best_idx, 0]
		print("Best resolution: ", res)
		return res

	def _priori_cluster(
		self,
		adata,
		n_domains = 7,
		):
		for res in sorted(list(np.arange(0.1, 2.5, 0.01)), reverse=True):
			sc.tl.leiden(adata, random_state=0, resolution=res)
			count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
			if count_unique_leiden == n_domains:
				break
		print("Best resolution: ", res)
		return res

	def _get_multiple_adata(
		self,
		adata_list,
		data_name_list,
		graph_list,
		):
		for i in range(len(data_name_list)):
			current_adata = adata_list[i]
			current_adata.obs['batch_name'] = data_name_list[i]
			current_adata.obs['batch_name'] = current_adata.obs['batch_name'].astype('category')
			current_graph = graph_list[i]
			if i == 0:
				multiple_adata = current_adata
				multiple_graph = current_graph
			else:
				var_names = multiple_adata.var_names.intersection(current_adata.var_names)
				multiple_adata = multiple_adata[:, var_names]
				current_adata = current_adata[:, var_names]
				multiple_adata = multiple_adata.concatenate(current_adata)
				multiple_graph = combine_graph_dict(multiple_graph, current_graph)

		multiple_adata.obs["batch"] = np.array(
            					pd.Categorical(
                					multiple_adata.obs['batch_name'],
                					categories=np.unique(multiple_adata.obs['batch_name'])).codes,
            						dtype=np.int64,
        						)

		return multiple_adata, multiple_graph

	def _data_process(self,
		adata,
		pca_n_comps = 200,
		):
		adata.raw = adata
		adata.X = adata.obsm["augment_gene_data"].astype(np.float64)
		data = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
		data = sc.pp.log1p(data)
		data = sc.pp.scale(data)
		data = sc.pp.pca(data, n_comps=pca_n_comps)
		return data

	def _fit(
		self,
		data,
		graph_dict,
		domains = None,
		n_domains = None,
		Conv_type = "GCNConv", 
		linear_encoder_hidden = [32, 20],
		linear_decoder_hidden = [32],
		conv_hidden = [32, 8], 
		p_drop = 0.01, 
		dec_cluster_n = 20, 
		kl_weight = 1,
		mse_weight = 1,
		bce_kld_weight = 1,
		domain_weight = 1,
		):
		print("Your task is in full swing, please wait")
		start_time = time.time()
		deepst_model = DeepST_model(
				input_dim = data.shape[1], 
                Conv_type = Conv_type,
				linear_encoder_hidden = linear_encoder_hidden,
				linear_decoder_hidden = linear_decoder_hidden,
				conv_hidden = conv_hidden,
				p_drop = p_drop,
				dec_cluster_n = dec_cluster_n,
				)
		if self.task == "Identify_Domain":
			deepst_training = train(
					data, 
					graph_dict, 
					deepst_model, 
					pre_epochs = self.pre_epochs, 
					epochs = self.epochs,
					kl_weight = kl_weight,
                			mse_weight = mse_weight, 
                			bce_kld_weight = bce_kld_weight,
                			domain_weight = domain_weight,
                			use_gpu = self.use_gpu
                			)
		elif self.task == "Integration":
			deepst_adversial_model = AdversarialNetwork(model = deepst_model, n_domains = n_domains)
			deepst_training = train(
					data, 
					graph_dict, 
					deepst_adversial_model,
					domains = domains,
					pre_epochs = self.pre_epochs, 
					epochs = self.epochs,
					kl_weight = kl_weight,
                			mse_weight = mse_weight, 
                			bce_kld_weight = bce_kld_weight,
                			domain_weight = domain_weight,
                			use_gpu = self.use_gpu
                			)
		else:
			print("There is no such function yet, looking forward to further development")
		deepst_training.fit()
		deepst_embed, _ = deepst_training.process()
		print("Step 3: DeepST training has been Done!")
		print(u'Current memory usage：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
		end_time = time.time()
		total_time = end_time - start_time
		print(f"Total time: {total_time / 60 :.2f} minutes")
		print("Your task has been completed, thank you")
		print("Of course, you can also perform downstream analysis on the processed data")

		return deepst_embed

	def _get_cluster_data(
		self,
		adata,
		n_domains,
		priori = True,
		):
		sc.pp.neighbors(adata, use_rep='DeepST_embed')
		if priori:
			res = self._priori_cluster(adata, n_domains = n_domains)
		else:
			res = self._optimize_cluster(adata)
		sc.tl.leiden(adata, key_added="DeepST_domain", resolution=res)
		######### Strengthen the distribution of points in the model
		adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
		refined_pred= refine(sample_id=adata.obs.index.tolist(), 
							 pred=adata.obs["DeepST_domain"].tolist(), dis=adj_2d, shape="hexagon")
		adata.obs["DeepST_refine_domain"]= refined_pred
		# save_data_path = Path(os.path.join(self.save_path, 'Data'))
		# save_data_path.mkdir(parents=True, exist_ok=True)
		# adata.write(os.path.join(save_data_path, 'DeepST_processed.h5ad'), compression="gzip")
		return adata
