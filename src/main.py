import numpy as np
import pandas as pd
import scanpy as sc
import random
import warnings
warnings.filterwarnings("ignore")
from spatialSAE import *

def check_symmetric(a, rtol=1e-03, atol=1e-03):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


################################ 10X Visium DLPFC dataset ################################
'''
img = np.load('../tutorial/img.npy')
slice_id = '151673'
adata = sc.read_visium(path='../data/DLPFC/%s'%slice_id, count_file='%s_filtered_feature_bc_matrix.h5'%slice_id)
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
adata = adata[:, adata.var.highly_variable]
print('highly varible genes', adata.shape)
sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
#load annotation label
annot_df = pd.read_csv('../data/DLPFC/DLPFC_annotations/%s_truth.txt'%slice_id, sep='\t', header=None, index_col=0)
#annot_df.fillna('NA', inplace=True)
annot_df.columns = ['annotation']
adata.obs['annotation'] = annot_df.loc[adata.obs_names, 'annotation']
#filter NA cells/spots
select_idx = pd.notnull(adata.obs['annotation'])    
adata = adata[select_idx,:]
print(adata.shape, len(adata.obs['annotation']))
'''

################################ STARmap cortex dataset ################################
count_mat = np.loadtxt('../data/STARmap/cell_barcode_count.csv',delimiter=',')
adata = sc.AnnData(count_mat) #(1549, 1020)
adata.var_names = [item.split(',')[0] for item in open('../data/STARmap/genes.csv').readlines()]
adata.obs_names = [str(i) for i in range(adata.n_obs)]
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
annot_df = pd.read_csv('../data/STARmap/class_labels.csv')
meta = pd.DataFrame(np.zeros((adata.shape[0],2)),columns=['a','b'])
adata.obs['cluster_id'] = None #can not directly set adata.obs['cluster_id']=annot_df['ClusterName']
adata.obs['cell_type'] = None #can not change
adata.obs['cluster_id'][:len(annot_df)] = annot_df["ClusterID"]
adata.obs['cell_type'][:len(annot_df)] = annot_df['ClusterName']
adata.obsm['spatial'] = np.load('../data/STARmap/all_centroids.npy')
adata.obs['x_pixel'] = adata.obsm['spatial'][:,0]
adata.obs['y_pixel'] = adata.obsm['spatial'][:,1]
#filter NA cells/spots
select_idx = pd.notnull(adata.obs['cell_type'])    
adata = adata[select_idx,:] #(1207, 1020)
adata.obs['annotation'] = get_spatial_label(adata.obsm['spatial'][:,1])
print(adata.shape, np.unique(adata.obs['annotation']))
############################


x_pixel = adata.obsm['spatial'][:,0]
y_pixel = adata.obsm['spatial'][:,1]

#adj = calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=49, alpha=1, p=0.5, histology=True, use_exp=True)

#adj = calculate_binary_adj_matrix(coor=adata.obsm['spatial'], rad_cutoff=150, model='Radius')
adj, adj_indices = calculate_binary_adj_matrix(coor=adata.obsm['spatial'], k_cutoff=4, model='KNN',return_indice=True)

#pre_labels = louvain_clustering(X=adata.obsm['X_pca'].copy(), K=7, res_min=0.01, res_max=1)
#adj, adj_indices = get_ground_truth_adj_matrix(coor=adata.obsm['spatial'], labels=pre_labels, n_neighbors=6, return_indice=True)

#adj, adj_indices = get_ground_truth_adj_matrix(coor=adata.obsm['spatial'], labels=adata.obs['annotation'], n_neighbors=6, return_indice=True)

adata.obsm['adj'] = adj
adata.obsm['adj_indices'] = adj_indices.astype('int16')

#Set seed
random.seed(0)
np.random.seed(0)
clf = spatialSAE()

# clf.train(adata, batch_size=64, max_epochs=100,
#         hidden_units=[1000, 500, 64], dim=adata.shape[1],
#         alpha=0.0, beta=0.0, gama=0.0, lr=0.0001, 
#         dropout=0.3, annotation=adata.obs['annotation'],tv_dim=3,
#         tau = 0.0, gd_dim=20, nb_clusters=7, use_gcn=True)

clf.train(adata, batch_size=32, max_epochs=200,
        hidden_units=[500, 128], dim=adata.shape[1],use_pca=False,
        alpha=0.0, beta=0.0, gama=0.1, lr=0.0001, 
        dropout=0.2, annotation=adata.obs['annotation'], tv_dim=[64,128],
        tau = 0.4, gd_dim=[0,64], nb_clusters=7, use_gcn=True)


# clf.train(adata, batch_size=32, max_epochs=100,
#         hidden_units=[30, 10, 3], dim=50, use_pca=True,
#         alpha=0.0, beta=0.0, gama=1., lr=0.0001, 
#         dropout=0.0, annotation=adata.obs['annotation'], tv_dim=[0,3],
#         tau = 0.0, gd_dim=32, nb_clusters=7, use_gcn=False)

# clf.train(adata, batch_size=32, max_epochs=200,
#         hidden_units=[500, 128], dim=adata.shape[1],use_pca=False,
#         alpha=0.0, beta=0.0, gama=0.1, lr=0.0001, 
#         dropout=0.2, annotation=adata.obs['annotation'], tv_dim=[64,128],
#         tau = 0.1, gd_dim=[0,64], nb_clusters=7, use_gcn=True)
