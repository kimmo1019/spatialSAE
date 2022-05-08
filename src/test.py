import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
from scipy.sparse import issparse
import random
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from spatialSAE import *

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

img = np.load('../tutorial/img.npy')
############################################################
slice_id = '151673'
adata = sc.read_visium(path='../data/DLPFC/%s'%slice_id, count_file='%s_filtered_feature_bc_matrix.h5'%slice_id)
adata.var_names_make_unique()
sc.pp.filter_genes(adata,min_cells=3)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
print(adata.shape, adata.X.max())

#load annotation label
annot_df = pd.read_csv('../data/DLPFC/DLPFC_annotations/%s_truth.txt'%slice_id, sep='\t', header=None, index_col=0)
annot_df.columns = ['annotation']
adata.obs['annotation'] = annot_df.loc[adata.obs_names, 'annotation']

x_pixel = adata.obsm['spatial'][:,0]
y_pixel = adata.obsm['spatial'][:,1]

s=1
b=49
p=0.5

adj = calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, p=p, histology=True, use_exp=True)
#adj = calculate_binary_adj_matrix(x=x_pixel,y=y_pixel, k_cutoff=6, model='KNN')

#Set seed
random.seed(0)
np.random.seed(0)
clf = spatialSAE()

clf.train(adata, adj, use_pca=False, num_pcs=100,
        batch_size=32, max_epochs=500,
        hidden_units=[1000, 500, 100], dim=adata.shape[1],
        alpha=0.00, beta=0.000, lr=0.0001, dropout=0.3, annotation=adata.obs['annotation'])
