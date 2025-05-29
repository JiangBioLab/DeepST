#!/usr/bin/env python3
"""
Spatial Transcriptomics Data Augmentation Module

This module provides functions for spatial data analysis and gene expression augmentation
by combining spatial proximity, gene expression similarity, and morphological information.

Functions:
1. cal_spatial_weight: Calculate spatial neighborhood weights
2. cal_gene_weight: Calculate gene expression similarity weights
3. cal_weight_matrix: Combine spatial, gene and morphological weights
4. find_adjacent_spot: Identify neighboring spots and their weights
5. augment_gene_data: Augment gene expression using neighboring spots
6. augment_adata: Complete pipeline for data augmentation
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree


def cal_spatial_weight(
    data,
    spatial_k=50,
    spatial_type="BallTree",
):
    """
    Calculate binary spatial weight matrix based on k-nearest neighbors.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Spatial coordinates array of shape (n_spots, 2)
    spatial_k : int, optional (default=50)
        Number of nearest neighbors to consider
    spatial_type : str, optional (default="BallTree")
        Algorithm for neighbor search:
        - "NearestNeighbors": sklearn's ball tree algorithm
        - "KDTree": KDTree algorithm
        - "BallTree": BallTree algorithm
        
    Returns:
    --------
    numpy.ndarray
        Binary spatial weight matrix of shape (n_spots, n_spots)
        where 1 indicates neighbors, 0 otherwise
    """
    if spatial_type == "NearestNeighbors":
        nbrs = NearestNeighbors(n_neighbors=spatial_k+1, algorithm='ball_tree').fit(data)
        _, indices = nbrs.kneighbors(data)
    elif spatial_type == "KDTree":
        tree = KDTree(data, leaf_size=2) 
        _, indices = tree.query(data, k=spatial_k+1)
    elif spatial_type == "BallTree":
        tree = BallTree(data, leaf_size=2)
        _, indices = tree.query(data, k=spatial_k+1)
    
    # Remove self from neighbors and create binary matrix
    indices = indices[:, 1:]
    spatial_weight = np.zeros((data.shape[0], data.shape[0]))
    for i in range(indices.shape[0]):
        spatial_weight[i, indices[i]] = 1
        
    return spatial_weight


def cal_gene_weight(
    data,
    n_components=50,
    gene_dist_type="cosine",
):
    """
    Calculate gene expression similarity matrix with optional PCA reduction.
    
    Parameters:
    -----------
    data : numpy.ndarray or scipy.sparse.csr_matrix
        Gene expression matrix of shape (n_spots, n_genes)
    n_components : int, optional (default=50)
        Number of PCA components if n_genes > 500
    gene_dist_type : str, optional (default="cosine")
        Distance metric for calculating gene similarity
        
    Returns:
    --------
    numpy.ndarray
        Gene expression similarity matrix of shape (n_spots, n_spots)
        Values range from -1 to 1 (1 - distance)
    """
    if isinstance(data, csr_matrix):
        data = data.toarray()
    
    # Apply PCA for high-dimensional data
    if data.shape[1] > 500:
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)
    
    return 1 - pairwise_distances(data, metric=gene_dist_type)


def cal_weight_matrix(
    adata,
    md_dist_type="cosine",
    gb_dist_type="correlation",
    n_components=50,
    use_morphological=True,
    spatial_k=30,
    spatial_type="BallTree",
    verbose=False,
):
    """
    Calculate combined weight matrix incorporating spatial, gene, and morphological information.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        Spatial transcriptomics dataset containing:
        - obsm['spatial']: Spatial coordinates
        - X: Gene expression matrix
        - obsm['image_feat_pca']: Morphological features (if use_morphological=True)
    md_dist_type : str, optional (default="cosine")
        Distance metric for morphological similarity
    gb_dist_type : str, optional (default="correlation")
        Distance metric for gene expression similarity
    n_components : int, optional (default=50)
        Number of PCA components for gene expression
    use_morphological : bool, optional (default=True)
        Whether to include morphological similarity
    spatial_k : int, optional (default=30)
        Number of spatial neighbors to consider
    spatial_type : str, optional (default="BallTree")
        Method for spatial neighbor calculation ("BallTree", "KDTree", "NearestNeighbors", "LinearRegress")
    verbose : bool, optional (default=False)
        Whether to store intermediate matrices in adata.obsm
        
    Returns:
    --------
    anndata.AnnData
        Updated AnnData object with:
        - obsm['weights_matrix_all']: Combined weight matrix
        - obsm['gene_correlation']: Gene weights (if verbose=True)
        - obsm['physical_distance']: Spatial weights (if verbose=True)
        - obsm['morphological_similarity']: Morphological weights (if verbose=True and use_morphological=True)
    """
    # Calculate spatial weights
    if spatial_type == "LinearRegress":
        img_row = adata.obs["imagerow"]
        img_col = adata.obs["imagecol"]
        array_row = adata.obs["array_row"]
        array_col = adata.obs["array_col"]
        rate = 3
        
        # Fit linear regression models
        reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)
        reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)
        
        # Calculate physical distances
        physical_distance = pairwise_distances(
            adata.obs[["imagecol", "imagerow"]], 
            metric="euclidean"
        )
        unit = math.sqrt(reg_row.coef_**2 + reg_col.coef_**2)
        physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
    else:
        physical_distance = cal_spatial_weight(
            adata.obsm['spatial'], 
            spatial_k=spatial_k, 
            spatial_type=spatial_type
        )
    
    print(f"Spatial weights calculated. Average neighbors: {physical_distance.sum()/adata.shape[0]:.1f}")
    
    # Calculate gene expression weights
    gene_correlation = cal_gene_weight(
        data=adata.X.copy(),
        gene_dist_type=gb_dist_type,
        n_components=n_components
    )
    print("Gene expression weights calculated.")
    
    if verbose:
        adata.obsm["gene_correlation"] = gene_correlation
        adata.obsm["physical_distance"] = physical_distance
    
    # Calculate and combine morphological weights if needed
    if use_morphological:
        morphological_similarity = 1 - pairwise_distances(
            np.array(adata.obsm["image_feat_pca"]), 
            metric=md_dist_type
        )
        morphological_similarity[morphological_similarity < 0] = 0
        print("Morphological weights calculated.")
        
        if verbose:
            adata.obsm["morphological_similarity"] = morphological_similarity
        
        # Combine all three weights
        adata.obsm["weights_matrix_all"] = (
            physical_distance * 
            gene_correlation * 
            morphological_similarity
        )
    else:
        # Combine only spatial and gene weights
        adata.obsm["weights_matrix_all"] = (
            gene_correlation * 
            physical_distance
        )
    
    print("Final weight matrix calculated and stored in adata.obsm['weights_matrix_all']")
    return adata


def find_adjacent_spot(
    adata,
    use_data="raw",
    neighbour_k=4,
    verbose=False,
):
    """
    Identify neighboring spots and calculate their weighted gene expression contributions.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        Dataset containing weights_matrix_all in obsm
    use_data : str, optional (default="raw")
        Data source to use:
        - "raw": uses adata.X
        - other: uses adata.obsm[use_data]
    neighbour_k : int, optional (default=4)
        Number of top neighbors to consider
    verbose : bool, optional (default=False)
        Whether to store neighbor weights in adata.obsm
        
    Returns:
    --------
    anndata.AnnData
        Updated AnnData with:
        - obsm['adjacent_data']: Weighted neighbor gene expression
        - obsm['adjacent_weight']: Neighbor weights (if verbose=True)
    """
    # Get expression data from specified source
    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            gene_matrix = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            gene_matrix = adata.X
        elif isinstance(adata.X, pd.DataFrame):
            gene_matrix = adata.X.values
        else:
            raise ValueError(f"Unsupported data type: {type(adata.X)}")
    else:
        gene_matrix = adata.obsm[use_data]
    
    weights_list = []
    final_coordinates = []
    
    # Process each spot to find neighbors
    with tqdm(total=len(adata), desc="Finding adjacent spots",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for i in range(adata.shape[0]):
            # Get top neighbors based on combined weights
            current_spot = adata.obsm['weights_matrix_all'][i].argsort()[-neighbour_k:][:neighbour_k-1]
            spot_weight = adata.obsm['weights_matrix_all'][i][current_spot]
            spot_matrix = gene_matrix[current_spot]
            
            # Calculate weighted contribution
            if spot_weight.sum() > 0:
                spot_weight_scaled = spot_weight / spot_weight.sum()
                weights_list.append(spot_weight_scaled)
                spot_matrix_scaled = spot_weight_scaled.reshape(-1,1) * spot_matrix
                spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
            else:
                spot_matrix_final = np.zeros(gene_matrix.shape[1])
                weights_list.append(np.zeros(len(current_spot)))
            
            final_coordinates.append(spot_matrix_final)
            pbar.update(1)
    
    # Store results
    adata.obsm['adjacent_data'] = np.array(final_coordinates)
    if verbose:
        adata.obsm['adjacent_weight'] = np.array(weights_list)
    
    return adata


def augment_gene_data(
    adata,
    adjacent_weight=0.2,
):
    """
    Augment gene expression data using neighboring spot information.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        Dataset containing:
        - X: Original gene expression
        - obsm['adjacent_data']: Neighbor contributions
    adjacent_weight : float, optional (default=0.2)
        Weight for neighbor contribution (0-1)
        
    Returns:
    --------
    anndata.AnnData
        Updated AnnData with augmented data in obsm['augment_gene_data']
    """
    if isinstance(adata.X, np.ndarray):
        augmented_matrix = adata.X + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
    elif isinstance(adata.X, csr_matrix):
        augmented_matrix = adata.X.toarray() + adjacent_weight * adata.obsm["adjacent_data"].astype(float)
    
    adata.obsm["augment_gene_data"] = augmented_matrix
    return adata


def augment_adata(
    adata,
    md_dist_type="cosine",
    gb_dist_type="correlation",
    n_components=50,
    use_morphological=True,
    use_data="raw",
    neighbour_k=4,
    adjacent_weight=0.2,
    spatial_k=30,
    spatial_type="KDTree",
):
    """
    Complete pipeline for spatial transcriptomics data augmentation.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        Input spatial transcriptomics data
    md_dist_type : str, optional (default="cosine")
        Morphological distance metric
    gb_dist_type : str, optional (default="correlation")
        Gene expression distance metric
    n_components : int, optional (default=50)
        PCA components for gene expression
    use_morphological : bool, optional (default=True)
        Whether to use morphological features
    use_data : str, optional (default="raw")
        Data source for expression
    neighbour_k : int, optional (default=4)
        Number of neighbors to consider
    adjacent_weight : float, optional (default=0.2)
        Weight for neighbor contributions
    spatial_k : int, optional (default=30)
        Spatial neighbors to consider
    spatial_type : str, optional (default="KDTree")
        Spatial neighbor algorithm
        
    Returns:
    --------
    anndata.AnnData
        Augmented dataset with:
        - obsm['weights_matrix_all']: Combined weights
        - obsm['adjacent_data']: Neighbor contributions
        - obsm['augment_gene_data']: Final augmented data
    """
    # Step 1: Calculate combined weight matrix
    adata = cal_weight_matrix(
        adata,
        md_dist_type=md_dist_type,
        gb_dist_type=gb_dist_type,
        n_components=n_components,
        use_morphological=use_morphological,
        spatial_k=spatial_k,
        spatial_type=spatial_type,
    )
    
    # Step 2: Find neighboring spots
    adata = find_adjacent_spot(
        adata,
        use_data=use_data,
        neighbour_k=neighbour_k,
    )
    
    # Step 3: Augment gene expression
    adata = augment_gene_data(
        adata,
        adjacent_weight=adjacent_weight,
    )
    
    return adata