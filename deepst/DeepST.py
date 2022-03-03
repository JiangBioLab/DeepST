#!/usr/bin/env python3

# @Author: *******
# @E-mail: ********
# @Last Modified by:   ********
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
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.spatial import distance
from tqdm import tqdm

from utils_func import *
from integrated_feat import image_feature, image_crop
from adj import graph, combine_graph_dict
from model import DeepST_model
from main import train

"""
Test:

"""

class run:
	def __init__(self,
			data_path,
			data_name,
			save_path,
			eval_cluster_n=7,
			pre_epochs=500, 
			epochs=1000,
			distType='euclidean',
			concat_pca_dim=40,
			linear_encoder_hidden=[100,20],
			linear_decoder_hidden=[30],
			conv_hidden=[32,8],
			k=25,
			verbose=False,
			platform='Visium',
			pca_n_comps=50,
			cnnType='ResNet50',
			save=True,
			eclidean_k = 60,
			pd_dist_type="euclidean",
			md_dist_type="correlation",
			gb_dist_type="cosine",
			use_data='raw', 
			neighbour_k=3,
			weights='weights_matrix_all', 
			Conv_type='GCNConv',		
			p_drop=0,
			dec_cluster_n=20,
			n_neighbors=15,
			min_cells = 5,
			Streamline_save = False,
			batchType = 'Harmony',
			):
		self.data_path = data_path
		self.data_name = data_name
		self.save_path = save_path
		self.verbose =  verbose
		self.platform = platform
		self.distType = distType
		self.pca_n_comps = pca_n_comps
		self.cnnType = cnnType
		self.save = save
		self.concat_pca_dim = concat_pca_dim
		self.k = k
		self.pd_dist_type = pd_dist_type
		self.md_dist_type = md_dist_type
		self.gb_dist_type = gb_dist_type
		self.use_data = use_data
		self.neighbour_k = neighbour_k
		self.weights = weights
		self.Conv_type = Conv_type
		self.linear_encoder_hidden = linear_encoder_hidden
		self.linear_decoder_hidden = linear_decoder_hidden
		self.conv_hidden = conv_hidden
		self.p_drop = p_drop
		self.dec_cluster_n = dec_cluster_n
		self.pre_epochs = pre_epochs
		self.epochs = epochs
		self.eval_cluster_n = eval_cluster_n
		self.n_neighbors = n_neighbors
		self.min_cells = min_cells
		self.Streamline_save = Streamline_save
		self.batchType = batchType
		self.eclidean_k = eclidean_k

	def load_adata(self,):
		if isinstance(self.data_name, str):
			if self.platform in ['Visium', 'ST']:
				if self.platform == 'Visium':
					adata = read_10X_Visium(os.path.join(self.data_path, self.data_name))
				else:
					adata = ReadOldST(os.path.join(self.data_path, self.data_name))
				save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', f'{self.data_name}'))
				save_path_image_crop.mkdir(parents=True, exist_ok=True)
				adata = image_crop(adata, save_path_image_crop)
				adata = image_feature(adata, pca_components=self.pca_n_comps, cnnType=self.cnnType).extract_image_feat()
			elif self.platform == 'MERFISH':
				adata = read_merfish(os.path.join(self.data_path, self.data_name))
			elif self.platform == 'SlideSeq':
				adata = read_SlideSeq(os.path.join(self.data_path, self.data_name))
			elif self.platform == 'SeqFish':
				adata = read_seqfish(os.path.join(self.data_path, self.data_name))
			elif self.platform == 'StereoSeq':
				adata = read_stereoSeq(os.path.join(self.data_path, self.data_name))
			else:
				raise ValueError(
               				 f"""\
               				 {self.platform!r} does not support.
                				""")
			graph_dict = graph(np.array(adata.obsm["spatial"]), distType=self.distType, k=self.k).main()

		elif isinstance(self.data_name, list):
			for idx in range(len(self.data_name)):
				if self.platform in ['Visium', 'ST']:
					if self.platform == 'Visium':
						current_data = read_10X_Visium(os.path.join(self.data_path, self.data_name[idx]))
					else:
						current_data = ReadOldST(os.path.join(self.data_path, self.data_name[idx]))
					save_path_image_crop = Path(os.path.join(self.save_path, 'Image_crop', f'{self.data_name[idx]}'))
					save_path_image_crop.mkdir(parents=True, exist_ok=True)
					current_data = image_crop(current_data, save_path_image_crop)
					current_data = image_feature(current_data, pca_components=self.pca_n_comps, cnnType=self.cnnType).extract_image_feat()

				elif self.platform == 'MERFISH':
					current_data = read_merfish(os.path.join(self.data_path, self.data_name[idx]))
				elif self.platform == 'SlideSeq':
					current_data = read_SlideSeq(os.path.join(self.data_path, self.data_name[idx]))
				elif self.platform == 'SeqFish':
					current_data = read_seqfish(os.path.join(self.data_path, self.data_name[idx]))
				elif self.platform == 'StereoSeq':
					current_adata = read_stereoSeq(os.path.join(self.data_path, self.data_name[idx]))
				else:
					raise ValueError(
                				f"""\
                				{self.platform!r} does not support.
                				""")

				current_data.obs['batch_label'] = np.ones(current_data.shape[0]) * idx
				current_data.obs['batch_label'] = current_data.obs['batch_label'].astype('category')
				current_data.obs['batch_name'] = self.data_name[idx]
				current_data.obs['batch_name'] = current_data.obs['batch_name'].astype('category')
				current_graph_dict = graph(current_data.obsm['spatial'], distType=self.distType, k=self.k).main()

				if idx == 0:
					adata = current_data
					graph_dict = current_graph_dict
				else:
					var_names = adata.var_names.intersection(current_data.var_names)
					adata = adata[:, var_names]
					current_data = current_data[:, var_names]
					adata = adata.concatenate(current_data)
					graph_dict = combine_graph_dict(graph_dict, current_graph_dict)
		else:
			raise ValueError("{} Must have data_name".format(self.data_name))

		self.num_cell = adata.shape[0]

		return adata, graph_dict

	def cal_spatial_weight(self, 
				data,
				spatial_k = 50,
				spatial_type = "NearestNeighbors",
				):
		from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
		if spatial_type == "NearestNeighbors":
			nbrs = NearestNeighbors(n_neighbors=spatial_k+1, algorithm='ball_tree').fit(data)
			_, indices = nbrs.kneighbors(data)
		elif spatial_type == "KDTree":
			tree = KDTree(data, leaf_size=2) 
			_, indices = tree.query(data, k=spatial_k+1)
		elif spatial_type == "BallTree":
			tree = BallTree(data, leaf_size=2)
			_, indices = tree.query(data, k=spatial_k+1)
		indices = indices[:, 1:]
		spatial_weight = np.zeros((data.shape[0], data.shape[0]))
		for i in range(indices.shape[0]):
			ind = indices[i]
			for j in ind:
				spatial_weight[i][j] = 1
		return spatial_weight

	def ingrated_adata(self,
			highly_variable = False,
			n_top_genes = 3000,):
		adata_all, graph_dict = self.load_adata()
		if isinstance(adata_all.X, csr_matrix):
			adata_all.X = adata_all.X.toarray()
		else:
			pass
		sc.pp.filter_genes(adata_all, min_cells=self.min_cells)
		sc.pp.normalize_total(adata_all)
		sc.pp.log1p(adata_all)
		if highly_variable:
			sc.pp.highly_variable_genes(adata_all, flavor="seurat_v3", n_top_genes=n_top_genes)
			adata_all =  adata_all[:, adata_all.var['highly_variable']]
		else:
			pass
		# sc.pp.scale(adata_all)
		sc.pp.pca(adata_all, n_comps=self.pca_n_comps)
		
		if isinstance(self.data_name, str):		
				adata = self.cal_weight_matrix(adata_all, 
							       pd_dist_type = self.pd_dist_type,
							       md_dist_type = self.md_dist_type,
							       gb_dist_type = self.gb_dist_type,)
				adata = self.find_adjacent_spot(adata, neighbour_k = self.neighbour_k)
		elif isinstance(self.data_name, list):	
			for idx in range(len(self.data_name)):
				current_adata = adata_all[adata_all.obs['batch_name'] == self.data_name[idx], :]
				current_adata = self.cal_weight_matrix(current_adata, 
								       pd_dist_type = self.pd_dist_type,
								       md_dist_type = self.md_dist_type,
								       gb_dist_type = self.gb_dist_type,)
				current_adata = self.find_adjacent_spot(current_adata, neighbour_k = self.neighbour_k)

				if idx == 0:
					adata = current_adata
				else:
					adata = adata.concatenate(current_adata)
		
		del adata_all

		return adata, graph_dict

	def search_cluster(self,
			   adata,
			   eval_cluster_n=7,):
		for res in sorted(list(np.arange(0.1, 2.5, 0.01)), reverse=True):
			sc.tl.leiden(adata, random_state=0, resolution=res)
			count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
			if count_unique_leiden == eval_cluster_n:
				break
		return res

	def cal_weight_matrix(self, 
			      adata,
			      pd_dist_type="euclidean",
			      md_dist_type="cosine",
			      gb_dist_type="correlation",
			      eclidean_k = 60):

		if self.platform == "Visium":
			img_row = adata.obs["imagerow"]
			img_col = adata.obs["imagecol"]
			array_row = adata.obs["array_row"]
			array_col = adata.obs["array_col"]
			rate = 3
		elif self.platform == "ST":
			img_row = adata.obs["imagerow"]
			img_col = adata.obs["imagecol"]
			array_row = adata.obs_names.map(lambda x: x.split("x")[1])
			array_col = adata.obs_names.map(lambda x: x.split("x")[0])
			rate = 1.5
		else:
			raise ValueError(
                f"""\
                {platform!r} does not support.
                """)

		reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
		reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)

		pd_1 = pairwise_distances(adata.obs[["imagecol", "imagerow"]], 
								  metric=pd_dist_type)
		unit = math.sqrt(reg_row.coef_ ** 2 + reg_col.coef_ ** 2)
		pd_norm_Re = np.where(pd_1 >= rate * unit, 0, 1)
		
		########## cal spatial weight
		from sklearn.neighbors import kneighbors_graph
		pd_norm_kneighbors_graph = kneighbors_graph(adata.obsm['spatial'], self.eclidean_k, 
							    mode='connectivity', include_self=False)
		pd_norm_kneighbors_graph = pd_norm_kneighbors_graph.toarray()

		pd_list = []
		pd = pairwise_distances(np.array(adata.obsm['spatial']), metric=pd_dist_type)
		for node_idx in range(adata.shape[0]):		
			tmp_pd = pd[node_idx, :]
			res = tmp_pd.argsort()[1:eclidean_k]
			tmpdist = tmp_pd[res]
			boundary = np.mean(tmpdist) + np.std(tmpdist)
			pd_array = np.where(tmp_pd >= boundary, 0, 1)
			pd_list.append(pd_array)
		pd_norm_eu = np.array(pd_list)


		######### cal "NearestNeighbors"
		pd_norm_NearestNeighbors = self.cal_spatial_weight(adata.obsm['spatial'], 
								   spatial_k = self.eclidean_k, spatial_type = "NearestNeighbors")

		pd_norm_KDTree = self.cal_spatial_weight(adata.obsm['spatial'], 
							 spatial_k = self.eclidean_k, spatial_type = "KDTree")

		pd_norm_BallTree = self.cal_spatial_weight(adata.obsm['spatial'], 
							   spatial_k = self.eclidean_k, spatial_type = "BallTree")
		pd_norm =  pd_norm_Re * pd_norm_kneighbors_graph * pd_norm_eu

		Average_pd = pd_norm.sum()/adata.shape[0]

		# print(Average_pd)

		gd = 1 - pairwise_distances(np.array(adata.obsm["X_pca"]), metric=gb_dist_type)
		# gd[gd < 0] = 0
		adata.obsm["gene_correlation"] = gd
		adata.obsm["physical_distance"] = pd_norm

		if self.platform in ['Visium', 'ST']: 
			md = 1 - pairwise_distances(np.array(adata.obsm["image_feat_pca"]), metric=md_dist_type)
			md[md < 0] = 0
			adata.obsm["morphological_distance"] = md	
			adata.obsm["weights_matrix_all"] = (adata.obsm["physical_distance"]
							    *adata.obsm["gene_correlation"]
							    *adata.obsm["morphological_distance"])
			adata.obsm["weights_matrix_nomd"] = (adata.obsm["gene_correlation"]
							     * adata.obsm["physical_distance"])							
		else:
			adata.obsm["weights_matrix_all"] = (adata.obsm["gene_correlation"]
							    * adata.obsm["physical_distance"])
		if self.verbose:
			print("The weight result of image feature is added to adata.obsm['weights_matrix_all'] !")

		return adata

	def find_adjacent_spot(self, 
			       adata, 
			       use_data='raw', 
			       neighbour_k=3,
			       weights='weights_matrix_all'):

		if use_data == "raw":
			if isinstance(adata.X, csr_matrix):
				gene_matrix = adata.X.toarray()
			elif isinstance(adata.X, np.ndarray):
				gene_matrix = adata.X
			elif isinstance(adata.X, pd.Dataframe):
				gene_matrix = adata.X.values
			else:
				raise ValueError(
                    f"""\
                        {type(adata.X)} is not a valid type.
                        """)
		else:
			gene_matrix = adata.obsm[use_data]

		weights_matrix = adata.obsm[weights]
		weights_list = []
		final_coordinates = []

		with tqdm(total=len(adata), desc="Find adjacent spots of each spot",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]",) as pbar:
			for i in range(adata.shape[0]):

				if weights == "physical_distance":
					current_spot = adata.obsm[weights][i].argsort()[-(neighbour_k+3):]
				else:
					current_spot = adata.obsm[weights][i].argsort()[-neighbour_k:]
				spot_weight = adata.obsm[weights][i][current_spot]
				spot_weight = adata.obsm["weights_matrix_nomd"][i][current_spot]
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

			integrated_adata = np.array(final_coordinates)
			adata.obsm['estimate_data'] = integrated_adata
			adata.obsm['estimate_weight'] = np.array(weights_list)
			return adata

	def fit(self,):
		print("Your task is in full swing, please wait")
		start_time = time.time()
		adata, graph_dict = self.ingrated_adata()
		if self.verbose:
			print("Step 1: integrated_representation adata is completed")

		imputed_data = adata.obsm["estimate_data"].astype(float)
		imputed_data[imputed_data == 0] = np.nan
		adjusted_count_matrix = np.nanmean(np.array([adata.X, imputed_data]), axis=0)
		adata.obsm["Integrated_data"] = adjusted_count_matrix
		concat_X = sc.pp.scale(adjusted_count_matrix)
		concat_X = sc.pp.pca(concat_X, n_comps=self.concat_pca_dim)			
		# graph_dict = graph(np.array(concat_X), distType=self.distType, k=self.k).main()
		stmap_model = STMAP_model(input_dim = concat_X.shape[1], 
                        		Conv_type = self.Conv_type,
                        		linear_encoder_hidden= self.linear_encoder_hidden,
					linear_decoder_hidden= self.linear_decoder_hidden,
					conv_hidden= self.conv_hidden,
					p_drop=self.p_drop,
					dec_cluster_n=self.dec_cluster_n,)
		stmap = train(concat_X, graph_dict, stmap_model, pre_epochs=self.pre_epochs, epochs=self.epochs)
		stmap.fit()
		stmap_feat, _ = stmap.process()
		if self.verbose:
			print("Step 2: model training has been completed")
		##### clustering
		del concat_X		
		cluster_adata = anndata.AnnData(stmap_feat)
		cluster_adata.obs_names = adata.obs_names
		if isinstance(self.data_name, list):
			cluster_adata.obs['batch_name'] = adata.obs['batch_name']
			cluster_adata.obsm['X_pca'] = stmap_feat
			if self.batchType == 'Harmony':
				sce.pp.harmony_integrate(cluster_adata, 'batch_name')
				cluster_adata.X = cluster_adata.obsm['X_pca_harmony']
			elif self.batchType == 'BBKNN':
				sce.pp.bbknn(cluster_adata, 'batch_name')
		else:
			pass
		adata.obsm['DeepST_feat'] = cluster_adata.X
		cluster_adata.obsm['X_pca'] = cluster_adata.X
		sc.pp.neighbors(cluster_adata, n_neighbors = self.n_neighbors)
		eval_resolution = self.search_cluster(cluster_adata, eval_cluster_n=self.eval_cluster_n)
		print(eval_resolution)
		sc.tl.leiden(cluster_adata, key_added="DeepST", resolution=eval_resolution)
		adata.obs['DeepST'] = cluster_adata.obs['DeepST']
		####### refine DeepST results
		adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
		refined_pred= refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["DeepST"].tolist(), dis=adj_2d, shape="hexagon")
		adata.obs["DeepST_refine"]= refined_pred
		if self.save:
			if type(self.data_name) == type('str'):
				save_path_data = Path(os.path.join(self.save_path, "Data", self.data_name))
			else:
				save_path_data = Path(os.path.join(self.save_path, "Data", '_'.join(self.data_name)))
			save_path_data.mkdir(parents=True, exist_ok=True)
			adata.X = sparse.csr_matrix(adata.X)
			adata.obsm["Integrated_data"] = sparse.csr_matrix(adata.obsm["Integrated_data"])
			if self.Streamline_save:
				del adata.obsm['gene_correlation']
				del adata.obsm['weights_matrix_all']
				del adata.obsm['estimate_weight']
				if self.platform in ['Visium', 'ST']:
					del adata.obsm['morphological_distance']
			if type(self.data_name) == type('str'):
				adata.write(os.path.join(save_path_data, f"{self.data_name}.h5ad"), compression="gzip")
			else:
				adata.write(os.path.join(save_path_data, f"{'_'.join(self.data_name)}.h5ad"), compression="gzip")
		print(u'Current memory usage：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
		end_time = time.time()
		total_time = end_time - start_time
		print(f"Total time: {total_time / 60 :.2f} minutes")
		print("Your task has been completed, thank you")
		print("Of course, you can also perform downstream analysis on the processed data")
		return adata, stmap_feat

	def plot_clustering(self, 
			    adata, 
			    img_key=None, 
			    color='DeepST',
			    show=False,
			    legend_loc='right margin',
			    legend_fontsize='x-large',
			    size=1.6,
			    dpi=300):
		if isinstance(self.data_name, str):
			sc.pl.spatial(adata, img_key=img_key, color=color, show=show, 
    					 legend_loc=legend_loc, legend_fontsize=legend_fontsize, size=size)
			save_path_figure = Path(os.path.join(self.save_path, "Figure", self.data_name))
			save_path_figure.mkdir(parents=True, exist_ok=True)
			plt.savefig(os.path.join(save_path_figure,f'{self.data_name}_{self.Conv_type}_{self.k}_{self.distType}_{self.eval_cluster_n}_{self.eclidean_k}_{self.neighbour_k}_{self.concat_pca_dim}_clustering.pdf'), bbox_inches='tight', dpi=dpi)
		else:
			pass

	def plot_umap(self, 
		      adata,
		      color='batch_name', 
		      legend_loc=None,
		      legend_fontsize=12,
		      legend_fontoutline=2,
		      frameon=False,
		      add_outline=True,
		      dpi=300,
		     ):
		umap_adata = anndata.AnnData(adata.obsm["DeepST_feat"])
		umap_adata.obs_names = adata.obs_names
		umap_adata.obs = adata.obs
		sc.pp.neighbors(umap_adata, n_neighbors = self.n_neighbors)
		sc.tl.umap(umap_adata)
		sc.pl.umap(umap_adata, color=color, add_outline=add_outline, legend_loc=legend_loc,
            		legend_fontsize=legend_fontsize, legend_fontoutline=legend_fontoutline, 
            		frameon=frameon)
		if isinstance(self.data_name, list):		
			save_path_figure = Path(os.path.join(self.save_path, "Figure", '_'.join(self.data_name)))
			save_path_figure.mkdir(parents=True, exist_ok=True)
			plt.savefig(os.path.join(save_path_figure, f"{'_'.join(self.data_name)}_umap.pdf"), bbox_inches='tight', dpi=dpi)
		else:
			save_path_figure = Path(os.path.join(self.save_path, "Figure", self.data_name))
			save_path_figure.mkdir(parents=True, exist_ok=True)
			plt.savefig(os.path.join(save_path_figure, f"{self.data_name}_umap.pdf"), bbox_inches='tight', dpi=dpi)
			

def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred



# efined_pred= refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["STMAP"].tolist(), dis=adj_2d, shape="hexagon")
# adata.obs["STMAP_refine"]= efined_pred
# sc.pl.spatial(adata, img_key="hires", color="STMAP_refine", show=False, size=1.6)
# plt.savefig("/home/xuchang/Project/STMAP_Final/data/OFC4/spatial/cluster_refine.png", bbox_inches='tight',dpi=300)

 


# adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
# refined_pred= refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
# adata.obs["refined_pred"]=refined_pred
# adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')       	



