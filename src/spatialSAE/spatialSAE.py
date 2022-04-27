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
            num_pcs=50, 
            lr=0.005,
            max_epochs=2000,
            **params):
        self.num_pcs=num_pcs
        self.lr=lr
        self.max_epochs=max_epochs

        assert adata.shape[0]==adj.shape[0]==adj.shape[1]
        pca = PCA(n_components=self.num_pcs)
        if issparse(adata.X):
            pca.fit(adata.X.A)
            embed=pca.transform(adata.X.A)
        else:
            pca.fit(adata.X)
            embed=pca.transform(adata.X)
        #----------Train model----------
        if self.l is None:
            raise ValueError('l should be set before fitting the model!')
        adj_exp=np.exp(-1*(adj**2)/(2*(self.l**2)))
        params = {}
        self.model = StructuredAE(params)
        self.model.fit(embed, adj_exp, max_epochs=self.max_epochs, shuffle=True)

    def predict(self):
        z=self.model.predict(self.embed)
        return z
