#!/usr/bin/env python3
"""
DeepST Pipeline Runner
Author: ChangXu
Description: Main pipeline for Deep Spatial Transcriptomics analysis
"""

import os
import time
import psutil
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from pathlib import Path
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial import distance
from typing import Optional, List, Union
import anndata

from .utils_func import read_10X_Visium, read_merfish, read_SlideSeq, read_seqfish, read_stereoSeq, refine
from .his_feat import image_feature, image_crop
from .adj import graph, combine_graph_dict
from .model import DeepST_model, AdversarialNetwork
from .trainer import train
from .augment import augment_adata


class run():
    def __init__(
        self,
        save_path: str = "./",
        task: str = "Identify_Domain",
        pre_epochs: int = 1000,
        epochs: int = 500,
        use_gpu: bool = True
    ):
        """
        Initialize DeepST pipeline runner
        
        Parameters:
        -----------
        save_path : str
            Path to save results (default: "./")
        task : str
            Analysis task type (default: "Identify_Domain")
            Options: ["Identify_Domain", "Integration"]
        pre_epochs : int
            Number of pretraining epochs (default: 1000)
        epochs : int
            Number of training epochs (default: 500)
        use_gpu : bool
            Whether to use GPU if available (default: True)
        """
        self.save_path = Path(save_path)
        self.task = task
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.use_gpu = use_gpu

    def _get_adata(
        self,
        platform: str,
        data_path: str,
        data_name: str,
        verbose: bool = True
    ) -> sc.AnnData:
        """
        Load spatial transcriptomics data
        
        Parameters:
        -----------
        platform : str
            Technology platform 
            Options: ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
        data_path : str
            Path to data directory
        data_name : str
            Name of dataset
        verbose : bool
            Whether to save raw data (default: True)
            
        Returns:
        --------
        sc.AnnData
            Loaded spatial transcriptomics data
        """
        platform_options = ['Visium', 'ST', 'MERFISH', 'slideSeq', 'stereoSeq']
        if platform not in platform_options:
            raise ValueError(f"Platform must be one of {platform_options}")
            
        reader_map = {
            'Visium': read_10X_Visium,
            'MERFISH': read_merfish,
            'slideSeq': read_SlideSeq,
            'stereoSeq': read_stereoSeq
        }
        
        adata = reader_map[platform](os.path.join(data_path, data_name))
        
        if verbose:
            save_dir = self.save_path / "Data" / data_name
            save_dir.mkdir(parents=True, exist_ok=True)
            adata.write(save_dir / f'{data_name}_raw.h5ad', compression="gzip")
            
        return adata

    def _get_image_crop(
        self,
        adata: sc.AnnData,
        data_name: str,
        cnn_type: str = 'ResNet50',
        pca_n_comps: int = 50
    ) -> sc.AnnData:
        """
        Extract image features from spatial data
        
        Parameters:
        -----------
        adata : sc.AnnData
            Spatial transcriptomics data
        data_name : str
            Name of dataset
        cnn_type : str
            CNN model type for feature extraction (default: 'ResNet50')
        pca_n_comps : int
            Number of PCA components (default: 50)
            
        Returns:
        --------
        sc.AnnData
            Data with extracted image features
        """
        save_dir = self.save_path / 'Image_crop' / data_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        adata = image_crop(adata, save_path=save_dir)
        adata = image_feature(
            adata, 
            pca_components=pca_n_comps, 
            cnnType=cnn_type
        ).extract_image_feat()
        
        return adata

    def _get_augment(
        self,
        adata: sc.AnnData,
        adjacent_weight: float = 0.3,
        neighbour_k: int = 4,
        spatial_k: int = 30,
        n_components: int = 100,
        md_dist_type: str = "cosine",
        gb_dist_type: str = "correlation",
        use_morphological: bool = True,
        use_data: str = "raw",
        spatial_type: str = "KDTree"
    ) -> sc.AnnData:
        """
        Augment spatial transcriptomics data
        
        Parameters:
        -----------
        adata : sc.AnnData
            Input spatial data
        adjacent_weight : float
            Weight for adjacent spots (default: 0.3)
        neighbour_k : int
            Number of neighbors (default: 4)
        spatial_k : int
            Spatial neighbors (default: 30)
        n_components : int
            Number of components (default: 100)
        md_dist_type : str
            Distance type (default: "cosine")
        gb_dist_type : str
            Distance type (default: "correlation")
        use_morphological : bool
            Use morphological features (default: True)
        use_data : str
            Data type to use (default: "raw")
        spatial_type : str
            Spatial neighbor type (default: "KDTree")
            
        Returns:
        --------
        sc.AnnData
            Augmented spatial data
        """
        adata = augment_adata(
            adata,
            md_dist_type=md_dist_type,
            gb_dist_type=gb_dist_type,
            n_components=n_components,
            use_morphological=use_morphological,
            use_data=use_data,
            neighbour_k=neighbour_k,
            adjacent_weight=adjacent_weight,
            spatial_k=spatial_k,
            spatial_type=spatial_type
        )
        print("Step 1: Data augmentation completed")
        return adata

    def _get_graph(
        self,
        spatial_coords: np.ndarray,
        distType: str = "BallTree",
        k: int = 12,
        rad_cutoff: float = 150
    ) -> dict:
        """
        Compute spatial graph
        
        Parameters:
        -----------
        spatial_coords : np.ndarray
            Spatial coordinates [n_spots, 2]
        distType : str
            Distance computation method (default: "BallTree")
        k : int
            Number of neighbors (default: 12)
        rad_cutoff : float
            Radius cutoff (default: 150)
            
        Returns:
        --------
        dict
            Graph dictionary containing:
            - adj_norm: Normalized adjacency matrix
            - adj_label: Original adjacency labels
            - norm_value: Normalization factor
        """
        graph_dict = graph(
            spatial_coords,
            distType=distType,
            k=k,
            rad_cutoff=rad_cutoff
        ).main()
        print("Step 2: Spatial graph computation completed")
        return graph_dict

    def _optimize_cluster(
        self,
        adata: sc.AnnData,
        resolution_range: List[float] = None
    ) -> float:
        """
        Optimize clustering resolution
        
        Parameters:
        -----------
        adata : sc.AnnData
            Spatial transcriptomics data
        resolution_range : List[float]
            Range of resolutions to test (default: 0.1-2.5 in 0.01 steps)
            
        Returns:
        --------
        float
            Optimal resolution value
        """
        if resolution_range is None:
            resolution_range = list(np.arange(0.1, 2.5, 0.01))
            
        scores = []
        for r in resolution_range:
            sc.tl.leiden(adata, resolution=r, flavor="igraph", n_iterations=2, directed=False)
            s = calinski_harabasz_score(adata.X, adata.obs["leiden"])
            scores.append(s)
            
        best_idx = np.argmax(scores)
        best_res = resolution_range[best_idx]
        print(f"Optimal resolution: {best_res:.2f}")
        return best_res

    def _priori_cluster(
        self,
        adata: sc.AnnData,
        n_domains: int
    ) -> float:
        """
        Find resolution matching prior domain count
        
        Parameters:
        -----------
        adata : sc.AnnData
            Spatial transcriptomics data
        n_domains : int
            Target number of domains
            
        Returns:
        --------
        float
            Resolution yielding target domain count
        """
        for res in sorted(np.arange(0.1, 2.5, 0.01), reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res, flavor="igraph", n_iterations=2, directed=False)
            if len(adata.obs['leiden'].unique()) == n_domains:
                print(f"Found resolution: {res:.2f} for {n_domains} domains")
                return res
        return 1.0

    def _get_multiple_adata(
        self,
        adata_list: List[sc.AnnData],
        data_name_list: List[str],
        graph_list: List[dict]
    ) -> tuple:
        """
        Integrate multiple spatial datasets
        
        Parameters:
        -----------
        adata_list : List[sc.AnnData]
            List of spatial datasets
        data_name_list : List[str]
            List of dataset names
        graph_list : List[dict]
            List of spatial graphs
            
        Returns:
        --------
        tuple: (sc.AnnData, dict)
            Integrated AnnData object and combined graph
        """
        multiple_adata = None
        multiple_graph = None
        
        for i, (current_adata, current_name) in enumerate(zip(adata_list, data_name_list)):
            current_adata.obs['batch_name'] = current_name
            current_adata.obs['batch_name'] = current_adata.obs['batch_name'].astype('category')
            
            if i == 0:
                multiple_adata = current_adata
                multiple_graph = graph_list[i]
            else:
                var_names = multiple_adata.var_names.intersection(current_adata.var_names)
                multiple_adata = multiple_adata[:, var_names]
                current_adata = current_adata[:, var_names]
                multiple_adata = anndata.concat([multiple_adata, current_adata])
                multiple_graph = combine_graph_dict(multiple_graph, graph_list[i])
        
        # Convert batch names to numeric codes
        multiple_adata.obs["batch"] = pd.Categorical(
            multiple_adata.obs['batch_name']
        ).codes.astype(np.int64)
        
        return multiple_adata, multiple_graph

    def _data_process(
        self,
        adata: sc.AnnData,
        pca_n_comps: int = 200
    ) -> np.ndarray:
        """
        Preprocess spatial transcriptomics data
        
        Parameters:
        -----------
        adata : sc.AnnData
            Spatial transcriptomics data
        pca_n_comps : int
            Number of PCA components (default: 200)
            
        Returns:
        --------
        np.ndarray
            Processed data matrix
        """
        adata.raw = adata
        adata.X = adata.obsm["augment_gene_data"].astype(np.float64)
        
        # Normalization pipeline
        data = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
        data = sc.pp.log1p(data)
        data = sc.pp.scale(data)
        data = sc.pp.pca(data, n_comps=pca_n_comps)
        
        return data

    def _fit(
        self,
        data: np.ndarray,
        graph_dict: dict,
        domains: Optional[np.ndarray] = None,
        n_domains: Optional[int] = None,
        conv_type: str = "GATConv",
        linear_encoder_hidden: List[int] = [32, 20],
        linear_decoder_hidden: List[int] = [32],
        conv_hidden: List[int] = [32, 8],
        p_drop: float = 0.01,
        dec_cluster_n: int = 20,
        kl_weight: float = 1,
        mse_weight: float = 1,
        bce_kld_weight: float = 1,
        domain_weight: float = 1
    ) -> np.ndarray:
        """
        Run DeepST model training
        
        Parameters:
        -----------
        data : np.ndarray
            Processed input data [n_spots, n_features]
        graph_dict : dict
            Spatial graph dictionary
        domains : np.ndarray, optional
            Domain labels for integration (default: None)
        n_domains : int, optional
            Number of domains for integration (default: None)
        conv_type : str
            Graph convolution type (default: "GATConv")
        linear_encoder_hidden : List[int]
            Encoder hidden layer sizes (default: [32, 20])
        linear_decoder_hidden : List[int]
            Decoder hidden layer sizes (default: [32])
        conv_hidden : List[int]
            GNN hidden layer sizes (default: [32, 8])
        p_drop : float
            Dropout probability (default: 0.01)
        dec_cluster_n : int
            Number of clusters (default: 20)
        kl_weight : float
            KL divergence weight (default: 1)
        mse_weight : float
            MSE loss weight (default: 1)
        bce_kld_weight : float
            BCE+KLD loss weight (default: 1)
        domain_weight : float
            Domain loss weight (default: 1)
            
        Returns:
        --------
        np.ndarray
            DeepST embeddings [n_spots, n_features]
        """
        print("Running DeepST analysis...")
        start_time = time.time()
        
        # Initialize model
        model = DeepST_model(
                        input_dim=data.shape[1],
                        Conv_type=conv_type,
                        linear_encoder_hidden=linear_encoder_hidden,
                        linear_decoder_hidden=linear_decoder_hidden,
                        conv_hidden=conv_hidden,
                        p_drop=p_drop,
                        dec_cluster_n=dec_cluster_n,
                        )
        
        # Configure for task type
        if self.task == "Identify_Domain":
            trainer = train(
                data, graph_dict, model,
                pre_epochs=self.pre_epochs,
                epochs=self.epochs,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
                bce_kld_weight=bce_kld_weight,
                domain_weight=domain_weight,
                use_gpu=self.use_gpu
            )
        elif self.task == "Integration":
            if n_domains is None:
                raise ValueError("n_domains must be specified for integration task")
                
            adv_model = AdversarialNetwork(model=model, n_domains=n_domains)
            trainer = train(
                data, graph_dict, adv_model,
                domains=domains,
                pre_epochs=self.pre_epochs,
                epochs=self.epochs,
                kl_weight=kl_weight,
                mse_weight=mse_weight,
                bce_kld_weight=bce_kld_weight,
                domain_weight=domain_weight,
                use_gpu=self.use_gpu
            )
        else:
            raise ValueError(f"Unknown task type: {self.task}")
        
        # Run training
        trainer.fit()
        embeddings, _ = trainer.process()
        
        # Print stats
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        total_time = (time.time() - start_time) / 60
        print(f"Step 3: DeepST training completed")
        print(f"Memory usage: {mem_usage:.2f} GB")
        print(f"Total time: {total_time:.2f} minutes")
        print("Analysis completed successfully")
        
        return embeddings

    def _get_cluster_data(
        self,
        adata: sc.AnnData,
        n_domains: int,
        priori: bool = True,
        batch_key: Optional[str] = None,
        use_obsm: str = "DeepST_embed",
        key_added: str = "DeepST_domain",
        output_key: str = "DeepST_refine_domain",
        shape: str = "hexagon",  # or "square"
    ) -> sc.AnnData:
        """
        Perform spatial clustering using DeepST embeddings and optionally refine results.

        This method applies Leiden clustering to spatial transcriptomics data using the specified
        embedding (`use_obsm`) and optionally refines cluster labels based on spatial proximity.
        It supports both single-slice and multi-slice datasets (via `batch_key`).

        Parameters
        ----------
        adata : sc.AnnData
            Annotated data matrix. Requires:
              - `adata.obsm[use_obsm]`: embedding representation for clustering
              - `adata.obsm['spatial']`: spatial coordinates of each spot/cell

        n_domains : int
            Target number of spatial domains (clusters).

        priori : bool, optional (default: True)
            If True, use `n_domains` directly. If False, optimize number of clusters automatically.

        batch_key : str or None, optional (default: None)
            Key in `adata.obs` that identifies distinct tissue slices. If None, assumes single-slice data.

        use_obsm : str, optional (default: "DeepST_embed")
            Name of the key in `adata.obsm` containing embedding to use for clustering.

        key_added : str, optional (default: "DeepST_domain")
            Name of the column in `adata.obs` where Leiden clustering results will be stored.

        output_key : str, optional (default: "DeepST_refine_domain")
            Name of the column in `adata.obs` to store refined cluster labels based on spatial neighbors.

        shape : str, optional (default: "hexagon")
            Neighborhood structure to use during refinement.
            Choose "hexagon" for Visium-like data or "square" for grid-based ST platforms.

        Returns
        -------
        adata : sc.AnnData
            The updated AnnData object with:
              - `adata.obs[key_added]`: initial clustering labels from Leiden
              - `adata.obs[output_key]`: refined domain labels (post-spatial-smoothing)
        """
        sc.pp.neighbors(adata, use_rep=use_obsm)
        
        if priori:
            res = self._priori_cluster(adata, n_domains)
        else:
            res = self._optimize_cluster(adata)
            
        sc.tl.leiden(adata, key_added=key_added, resolution=res, flavor="igraph", n_iterations=2, directed=False)

        if batch_key and batch_key in adata.obs.columns:
            result = []
            for b in adata.obs[batch_key].unique():
                sub = adata[adata.obs[batch_key] == b]
                adj_2d = distance.cdist(sub.obsm['spatial'], sub.obsm['spatial'], 'euclidean')
                refined = refine(sub.obs_names.tolist(), sub.obs[key_added].tolist(), adj_2d, shape)
                result.extend(zip(sub.obs_names.tolist(), refined))
            adata.obs[output_key] = pd.Series(dict(result))
        else:
            adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
            refined = refine(adata.obs_names.tolist(), adata.obs[key_added].tolist(), adj_2d, shape)
            adata.obs[output_key] = refined
        
        return adata
