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
#import cv2
from spatialSAE import *

adata = sc.read('../tutorial/adata.h5')

img = np.load('../tutorial/img.npy')

#Calculate adjacent matrix
s=1
b=49
p=0.5 

x_pixel=adata.obs["x_pixel"].tolist()
y_pixel=adata.obs["y_pixel"].tolist()

adj = calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, p=p, histology=True, use_exp=True)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
print(check_symmetric(adj))

#Set seed
random.seed(0)
np.random.seed(0)
clf = spatialSAE()

clf.train(adata, adj, use_pca=False, num_pcs=100,
        batch_size=32, max_epochs=500,
        hidden_units=[3000, 1000, 100], dim=adata.shape[1],
        alpha=0.00, beta=0.0000, lr=0.0001, use_resnet=False)
