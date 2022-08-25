import os,csv,re,sys
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
import random, torch
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


BASE_PATH = Path('/home/xuchang/Project/STMAP/DLPFC')
output_path = Path('/home/xuchang/Project/DeepST/code_20220617/All_methods/SpaGCN_results')

sample_list = ['151507', '151508', '151509', '151510', 
                '151669', '151670', '151671', '151672', 
                '151673', '151674', '151675', '151676']

ARI_list = []
for sample_name in sample_list:
    dir_input = Path(f'{BASE_PATH}/{sample_name}/')
    dir_output = Path(f'{output_path}/{sample_name}/')
    dir_output.mkdir(parents=True, exist_ok=True)

    if sample_name in ['151669', '151670', '151671', '151672']:
        n_clusters = 5
    else:
        n_clusters = 7
    ##### read data
    adata = sc.read_10x_h5(f'{dir_input}/filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    spatial=pd.read_csv(f"{dir_input}/spatial/tissue_positions_list.csv",sep=",",header=None,na_filter=False,index_col=0)

    adata.obs["x1"]=spatial[1]
    adata.obs["x2"]=spatial[2]
    adata.obs["x3"]=spatial[3]
    adata.obs["x4"]=spatial[4]
    adata.obs["x5"]=spatial[5]

    adata=adata[adata.obs["x1"]==1]
    adata.var_names=[i.upper() for i in list(adata.var_names)]
    adata.var["genename"]=adata.var.index.astype("str")
    # adata.write_h5ad(f"{dir_output}/sample_data.h5ad")

    #Read in hitology image
    img=cv2.imread(f"{dir_input}/spatial/full_image.tif")

    #Set coordinates
    adata.obs["x_array"]=adata.obs["x2"]
    adata.obs["y_array"]=adata.obs["x3"]
    adata.obs["x_pixel"]=adata.obs["x4"]
    adata.obs["y_pixel"]=adata.obs["x5"]
    x_array=adata.obs["x_array"].tolist()
    y_array=adata.obs["y_array"].tolist()
    x_pixel=adata.obs["x_pixel"].tolist()
    y_pixel=adata.obs["y_pixel"].tolist()

    #Test coordinates on the image
    img_new=img.copy()
    for i in range(len(x_pixel)):
        x=x_pixel[i]
        y=y_pixel[i]
        img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0

    cv2.imwrite(f'{dir_output}/sample_map.jpg', img_new)

    #Calculate adjacent matrix
    b=49
    a=1
    adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=a, histology=True)
    np.savetxt(f'{dir_output}/adj.csv', adj, delimiter=',')


    ##### Spatial domain detection using SpaGCN
    spg.prefilter_genes(adata, min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    ### 4.2 Set hyper-parameters
    p=0.5 
    spg.test_l(adj,[1, 10, 100, 500, 1000])
    l=spg.find_l(p=p,adj=adj,start=100, end=500,sep=1, tol=0.01)
    n_clusters=n_clusters
    r_seed=t_seed=n_seed=100
    res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, 
                        t_seed=t_seed, n_seed=n_seed)

    ### 4.3 Run SpaGCN
    clf=spg.SpaGCN()
    clf.set_l(l)
    #Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    #Run
    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob=clf.predict()
    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')
    #Do cluster refinement(optional)
    adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    #Save results
    # adata.write_h5ad(f"{dir_output}/results.h5ad")
    # adata.obs.to_csv(f'{dir_output}/metadata.tsv', sep='\t')
    
    #Set colors used
    # adata=sc.read(f"{dir_output}/results.h5ad")
    plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
    #Plot spatial domains
    domains="pred"
    num_celltype=len(adata.obs[domains].unique())
    adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
    ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig(f"{dir_output}/pred.png", dpi=300)
    plt.close()

    #Plot refined spatial domains
    domains="refined_pred"
    num_celltype=len(adata.obs[domains].unique())
    adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
    ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig(f"{dir_output}/refined_pred.png", dpi=300)
    plt.close()
    
    df_meta = pd.read_csv(f'{dir_input}/metadata.tsv', sep='\t')
    df_meta['SpaGCN'] = adata.obs["refined_pred"].tolist()
    df_meta.to_csv(f'{dir_output}/metadata.tsv', sep='\t', index=False)
    df_meta = df_meta[~pd.isnull(df_meta['layer_guess'])]
    ARI = metrics.adjusted_rand_score(df_meta['layer_guess'], df_meta['SpaGCN'])
    print('===== Project: {} ARI score: {:.3f}'.format(sample_name, ARI))
    ARI_list.append(ARI)

print('===== Project: AVG ARI score: {:.3f}'.format(np.mean(ARI_list)))

