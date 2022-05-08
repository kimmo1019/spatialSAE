import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import numba

@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
    sum=0
    for i in range(t1.shape[0]):
        sum+=(t1[i]-t2[i])**2
    return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n=X.shape[0]
    adj=np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j]=euclid_dist(X[i], X[j])
    return adj

def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49):
    #beta to control the range of neighbourhood when calculate grey vale for one spot
    beta_half=round(beta/2)
    g=[]
    for i in range(len(x_pixel)):
        max_x=image.shape[0]
        max_y=image.shape[1]
        nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
        g.append(np.mean(np.mean(nbs,axis=0),axis=0))
    c0, c1, c2=[], [], []
    for i in g:
        c0.append(i[0])
        c1.append(i[1])
        c2.append(i[2])
    c0=np.array(c0)
    c1=np.array(c1)
    c2=np.array(c2)
    c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
    return c3

def calculate_binary_adj_matrix(x, y, rad_cutoff=None, k_cutoff=None, model='Radius'):
    assert len(x)==len(y)
    import sklearn.neighbors
    coor = np.vstack([x,y]).T
    assert(model in ['Radius', 'KNN'])
    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
    A = np.empty((len(x), len(y)))
    for i in range(indices.shape[0]):
        for ind in indices[i]:
            A[i][ind] = 1
            A[ind][i] = 1
    return A.astype(np.float32)

def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, p=0.5, histology=True, use_exp=True):
    #x,y,x_pixel, y_pixel are lists
    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
        print("Calculateing adj matrix using histology image...")
        #beta to control the range of neighbourhood when calculate grey vale for one spot
        #alpha to control the color scale
        beta_half=round(beta/2)
        g=[]
        for i in range(len(x_pixel)):
            max_x=image.shape[0]
            max_y=image.shape[1]
            nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
            g.append(np.mean(np.mean(nbs,axis=0),axis=0))
        c0, c1, c2=[], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0=np.array(c0)
        c1=np.array(c1)
        c2=np.array(c2)
        print("Var of c0,c1,c2 = ", np.var(c0),np.var(c1),np.var(c2))
        c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
        c4=(c3-np.mean(c3))/np.std(c3)
        z_scale=np.max([np.std(x), np.std(y)])*alpha
        z=c4*z_scale
        z=z.tolist()
        print("Var of x,y,z = ", np.var(x),np.var(y),np.var(z))
        X=np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculateing adj matrix using xy only...")
        X=np.array([x, y]).T.astype(np.float32)
    adj = pairwise_distance(X)
    if use_exp:
        #Find the l value given p
        l = search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
        adj = np.exp(-1*(adj**2)/(2*(l**2)))
    return adj

def calculate_p(adj, l):
    adj_exp=np.exp(-1*(adj**2)/(2*(l**2)))
    return np.mean(np.sum(adj_exp,1))-1


def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    run=0
    p_low=calculate_p(adj, start)
    p_high=calculate_p(adj, end)
    if p_low>p+tol:
        print("l not found, try smaller start point.")
        return None
    elif p_high<p-tol:
        print("l not found, try bigger end point.")
        return None
    elif  np.abs(p_low-p) <=tol:
        print("recommended l = ", str(start))
        return start
    elif  np.abs(p_high-p) <=tol:
        print("recommended l = ", str(end))
        return end
    while (p_low+tol)<p<(p_high-tol):
        run+=1
        print("Run "+str(run)+": l ["+str(start)+", "+str(end)+"], p ["+str(p_low)+", "+str(p_high)+"]")
        if run >max_run:
            print("Exact l not found, closest values are:\n"+"l="+str(start)+": "+"p="+str(p_low)+"\nl="+str(end)+": "+"p="+str(p_high))
            return None
        mid=(start+end)/2
        p_mid=calculate_p(adj, mid)
        if np.abs(p_mid-p)<=tol:
            print("recommended l = ", str(mid))
            return mid
        if p_mid<=p:
            start=mid
            p_low=p_mid
        else:
            end=mid
            p_high=p_mid


def get_cluster_label(X, res):
    adata=sc.AnnData(X)
    sc.pp.neighbors(adata, n_neighbors=10, use_rep = "X")
    sc.tl.louvain(adata,resolution=res)
    y_pred=adata.obs['louvain'].astype(int).to_numpy()
    return y_pred

def louvain_clustering(K, X, res_min=0.01, res_max=2, tol=1e-5, verbose=False):
    res = (res_min+res_max)/2.
    y_pred = get_cluster_label(X, res = res)
    if verbose:
        print('Resolution: ', res, 'number of clusters: ',len(np.unique(y_pred)))
    if len(np.unique(y_pred))==K or res_max-res_min<tol:
        if verbose:
            print('Best resolution: ', res, 'number of clusters: ',len(np.unique(y_pred)))
        return y_pred
    elif len(np.unique(y_pred))>K:
        res_max = res
        return louvain_clustering(K, X, res_min=res_min, res_max=res_max)
    else:
        res_min = res
        return louvain_clustering(K, X, res_min=res_min, res_max=res_max)


def mclust_clustering(K, X, modelNames='EEE',random_seed=2020):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    f = rpy2.robjects.numpy2ri.numpy2rpy
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    results = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(X), K, modelNames)
    print(results)
    cluster_labels = np.array(results[-2]).astype('int')
    return cluster_labels


def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred