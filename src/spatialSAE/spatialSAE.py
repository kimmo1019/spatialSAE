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
            num_pcs=100, 
            max_epochs=500,
            batch_size=128,
            **params):
        self.num_pcs=num_pcs
        self.max_epochs=max_epochs
        self.batch_size=batch_size

        assert adata.shape[0]==adj.shape[0]==adj.shape[1]
        if use_pca and isinstance(num_pcs, int):
            pca = PCA(n_components=self.num_pcs)
            if issparse(adata.X):
                pca.fit(adata.X.A)
                embed=pca.transform(adata.X.A)
            else:
                pca.fit(adata.X)
                embed=pca.transform(adata.X)
        else:
            if issparse(adata.X):
                embed = adata.X.A
            else:
                embed = adata.X
        #----------Train model----------
        print('Hyperparameters: ',params)
        self.model = StructuredAE(params)
        self.model.fit(embed, adj,bs=self.batch_size, max_epochs=self.max_epochs, shuffle=True)

    def predict(self):
        return self.model.predict(self.embed)
