import os
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from anndata import AnnData
from sklearn.decomposition import PCA
import math
from . model import StructuredAE

class spatialSAE(object):
    def __init__(self):
        super(spatialSAE, self).__init__()

    def train(self,adata,adj,
            use_pca=False,
            num_pcs=50, 
            max_epochs=500,
            batch_size=128,
            **params):
        self.num_pcs=num_pcs
        self.max_epochs=max_epochs
        self.batch_size=batch_size

        assert adata.shape[0]==adj.shape[0]==adj.shape[1]
        if use_pca and isinstance(num_pcs, int):
            if 'X_pca' in adata.obsm:
                embed = adata.obsm['X_pca']
            else:
                embed = PCA(n_components=self.num_pcs).fit_transform(adata.X)
        else:
            embed = adata.X.A
        #----------Train model----------
        print('Hyperparameters: ',params)
        self.model = StructuredAE(params)
        self.model.fit(embed, adj,bs=self.batch_size, max_epochs=self.max_epochs)

    def predict(self):
        return self.model.predict(self.embed)
