#!/usr/bin/env python3

# @Author: ChangXu
# @E-mail: xuchang0214@163.com
# @Last Modified by:   ChangXu
# @Last Modified time: 2021-04-22 08:42:54 23:22:34
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm



def cal_spatial_weight(
	data,
	spatial_k = 50,
	spatial_type = "BallTree",
	):
	from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
	if spatial_type == "NearestNeighbors":
		nbrs = NearestNeighbors(n_neighbors = spatial_k+1, algorithm ='ball_tree').fit(data)
		_, indices = nbrs.kneighbors(data)
	elif spatial_type == "KDTree":
		tree = KDTree(data, leaf_size=2) 
		_, indices = tree.query(data, k = spatial_k+1)
	elif spatial_type == "BallTree":
		tree = BallTree(data, leaf_size=2)
		_, indices = tree.query(data, k = spatial_k+1)
	indices = indices[:, 1:]
	spatial_weight = np.zeros((data.shape[0], data.shape[0]))
	for i in range(indices.shape[0]):
		ind = indices[i]
		for j in ind:
			spatial_weight[i][j] = 1
	return spatial_weight

def cal_gene_weight(
	data,
	n_components = 50,
	gene_dist_type = "cosine",
	):
	if isinstance(data, csr_matrix):
		data = data.toarray()
	if data.shape[1] > 500:
		pca = PCA(n_components = n_components)
		data = pca.fit_transform(data)
		gene_correlation = 1 - pairwise_distances(data, metric = gene_dist_type)
	else:
		gene_correlation = 1 - pairwise_distances(data, metric = gene_dist_type)
	return gene_correlation


def cal_weight_matrix(
		adata,
		md_dist_type="cosine",
		gb_dist_type="correlation",
		n_components = 50,
		use_morphological = True,
		spatial_k = 30,
		spatial_type = "BallTree",
		verbose = False,
		):
	if use_morphological:
		if spatial_type == "LinearRegress":
			img_row = adata.obs["imagerow"]
			img_col = adata.obs["imagecol"]
			array_row = adata.obs["array_row"]
			array_col = adata.obs["array_col"]
			rate = 3
			reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
			reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
			physical_distance = pairwise_distances(
									adata.obs[["imagecol", "imagerow"]], 
								  	metric="euclidean")
			unit = math.sqrt(reg_row.coef_ ** 2 + reg_col.coef_ ** 2)
			physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
		else:
			physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k = spatial_k, spatial_type = spatial_type)
	else:
		physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k = spatial_k, spatial_type = spatial_type)
	print("Physical distance calculting Done!")
	print("The number of nearest tie neighbors in physical distance is: {}".format(physical_distance.sum()/adata.shape[0]))
	
	########### gene_expression weight
	gene_correlation = cal_gene_weight(data = adata.X.copy(), 
											gene_dist_type = gb_dist_type, 
											n_components = n_components)
	# gene_correlation[gene_correlation < 0 ] = 0
	print("Gene correlation calculting Done!")
	if verbose:
		adata.obsm["gene_correlation"] = gene_correlation
		adata.obsm["physical_distance"] = physical_distance

	###### calculated image similarity
	if use_morphological: 
		morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric = md_dist_type)
		morphological_similarity[morphological_similarity < 0] = 0
		print("Morphological similarity calculting Done!")
		if verbose:
			adata.obsm["morphological_similarity"] = morphological_similarity	
		adata.obsm["weights_matrix_all"] = (physical_distance
												*gene_correlation
												*morphological_similarity)
		print("The weight result of image feature is added to adata.obsm['weights_matrix_all'] !")						
	else:
		adata.obsm["weights_matrix_all"] = (gene_correlation
												* physical_distance)
		print("The weight result of image feature is added to adata.obsm['weights_matrix_all'] !")
	return adata

def find_adjacent_spot(
	adata,
	use_data = "raw",
	neighbour_k = 4,
	verbose = False,
	):
	if use_data == "raw":
		if isinstance(adata.X, csr_matrix):
			gene_matrix = adata.X.toarray()
		elif isinstance(adata.X, np.ndarray):
			gene_matrix = adata.X
		elif isinstance(adata.X, pd.Dataframe):
			gene_matrix = adata.X.values
		else:
			raise ValueError(f"""{type(adata.X)} is not a valid type.""")
	else:
		gene_matrix = adata.obsm[use_data]
	weights_list = []
	final_coordinates = []
	with tqdm(total=len(adata), desc="Find adjacent spots of each spot",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
		for i in range(adata.shape[0]):
			current_spot = adata.obsm['weights_matrix_all'][i].argsort()[-neighbour_k:][:neighbour_k-1]
			spot_weight = adata.obsm['weights_matrix_all'][i][current_spot]
			spot_matrix = gene_matrix[current_spot]
			if spot_weight.sum() > 0:
				spot_weight_scaled = (spot_weight / spot_weight.sum())
				weights_list.append(spot_weight_scaled)
				spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1,1), spot_matrix)
				spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
			else:
				spot_matrix_final = np.zeros(gene_matrix.shape[1])
				weights_list.append(np.zeros(len(current_spot)))
			final_coordinates.append(spot_matrix_final)
			pbar.update(1)
		adata.obsm['adjacent_data'] = np.array(final_coordinates)
		if verbose:
			adata.obsm['adjacent_weight'] = np.array(weights_list)
		return adata


def augment_gene_data(
	adata,
	adjacent_weight = 0.2,
	):
	if isinstance(adata.X, np.ndarray):
		augement_gene_matrix =  adata.X + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
	elif isinstance(adata.X, csr_matrix):
		augement_gene_matrix = adata.X.toarray() + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
	adata.obsm["augment_gene_data"] = augement_gene_matrix
	return adata

def augment_adata(
	adata,
	md_dist_type="cosine",
	gb_dist_type="correlation",
	n_components = 50,
	use_morphological = True,
	use_data = "raw",
	neighbour_k = 4,
	adjacent_weight = 0.2,
	spatial_k = 30,
	spatial_type = "KDTree"
	):
	adata = cal_weight_matrix(
				adata,
				md_dist_type = md_dist_type,
				gb_dist_type = gb_dist_type,
				n_components = n_components,
				use_morphological = use_morphological,
				spatial_k = spatial_k,
				spatial_type = spatial_type,
				)
	adata = find_adjacent_spot(adata,
				use_data = use_data,
				neighbour_k = neighbour_k)
	adata = augment_gene_data(adata,
				adjacent_weight = adjacent_weight)
	return adata





