import os
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from anndata import AnnData
from sklearn.decomposition import PCA
import math
from . model import StructuredAE
from . util import calculate_binary_adj_matrix

class spatialSAE(object):
    def __init__(self):
        super(spatialSAE, self).__init__()

    def train(self, adata, **params):
        #----------Train model----------
        self.model = StructuredAE(params)
        self.model.fit(X=adata.X.A, adj=adata.obsm['adj'], adj_indices=adata.obsm['adj_indices'])

    def predict(self):
        return self.model.predict(self.embed)
