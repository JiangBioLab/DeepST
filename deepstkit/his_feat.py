#!/usr/bin/env python
"""
This module provides functions for extracting image features from spatial transcriptomics data.

Main Functions:
- image_feature: Class for extracting CNN-based image features from spot images
- image_crop: Function to crop spot images from the whole tissue image

Author: ChangXu
Created Time: Mon 23 Apr 2021 08:26:32
File Name: his_feat.py
"""

import os
import math
import anndata
import numpy as np 
import scanpy as sc
import pandas as pd 
from PIL import Image
from pathlib import Path
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
import random

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torchvision.transforms as transforms


class image_feature:
    def __init__(
        self,
        adata,
        pca_components=50,
        cnnType='ResNet50',
        verbose=False,
        seeds=88,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnnType = cnnType

    def load_cnn_model(self):
        """Load and return the specified CNN model with appropriate weights."""
        # Define model map using weights
        model_map = {
            'ResNet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
            'Resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT),
            'Vgg19': (models.vgg19, models.VGG19_Weights.DEFAULT),
            'Vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT),
            'DenseNet121': (models.densenet121, models.DenseNet121_Weights.DEFAULT),
            'Inception_v3': (models.inception_v3, models.Inception_V3_Weights.DEFAULT)
        }

        if self.cnnType not in model_map:
            raise ValueError(f"{self.cnnType} is not a valid CNN type. Options: {list(model_map.keys())}")

        model_func, weights = model_map[self.cnnType]
        model = model_func(weights=weights)
        model.to(self.device)
        return model

    def extract_image_feat(self):
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomAutocontrast(),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
            transforms.RandomInvert(),
            transforms.RandomAdjustSharpness(random.uniform(0, 1)),
            transforms.RandomSolarize(random.uniform(0, 1)),
            transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), 
                                  shear=(-0.3, 0.3, -0.3, 0.3)),
            transforms.RandomErasing()
        ]
        img_to_tensor = transforms.Compose(transform_list)

        features_list = []
        spot_names = []

        model = self.load_cnn_model()
        model.eval()

        if "slices_path" not in self.adata.obs.keys():
            raise ValueError("Please run image_crop() first to generate spot images")

        with tqdm(total=len(self.adata),
                  desc="Extracting image features",
                  bar_format="{l_bar}{bar} [ time left: {remaining} ]", ncols=80, dynamic_ncols=True, leave=True) as pbar:
            for spot, slice_path in self.adata.obs['slices_path'].items():
                spot_slice = Image.open(slice_path).resize((224, 224))
                spot_slice = np.asarray(spot_slice, dtype=np.float32)

                tensor = img_to_tensor(spot_slice).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = model(Variable(tensor)).cpu().numpy().ravel()

                features_list.append(features)
                spot_names.append(spot)
                pbar.update(1)

        feat_df = pd.DataFrame(features_list, index=spot_names)
        self.adata.obsm["image_feat"] = feat_df.to_numpy()

        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        self.adata.obsm["image_feat_pca"] = pca.fit_transform(self.adata.obsm["image_feat"])

        if self.verbose:
            print("Image features added to adata.obsm['image_feat']")
            print("PCA-reduced features added to adata.obsm['image_feat_pca']")

        return self.adata


def image_crop(
    adata,
    save_path,
    library_id=None,
    crop_size=50,
    target_size=224,
    verbose=False,
):
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)

    os.makedirs(save_path, exist_ok=True)
    tile_names = []

    with tqdm(total=len(adata),
              desc="Cropping spot images",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]", ncols=80, dynamic_ncols=True, leave=True) as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            left = imagecol - crop_size / 2
            upper = imagerow - crop_size / 2
            right = imagecol + crop_size / 2
            lower = imagerow + crop_size / 2

            tile = img_pillow.crop((left, upper, right, lower))
            tile = tile.resize((target_size, target_size))

            tile_name = f"{imagecol}-{imagerow}-{crop_size}.png"
            out_path = Path(save_path) / tile_name
            tile.save(out_path, "PNG")
            tile_names.append(str(out_path))

            if verbose:
                print(f"Generated tile at location ({imagecol}, {imagerow})")
            pbar.update(1)

    adata.obs["slices_path"] = tile_names
    if verbose:
        print("Spot image paths added to adata.obs['slices_path']")

    return adata
