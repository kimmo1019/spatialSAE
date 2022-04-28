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

def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, p=0.5, histology=True):
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
	#Find the l value given p
	l=search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
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

def louvain_cluster(X, res=1.0):
    adata=sc.AnnData(X)
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.louvain(adata,resolution=res)
    y_pred=adata.obs['louvain'].astype(int).to_numpy()
    n_clusters=len(np.unique(y_pred))
    return y_pred

def find_best_res(K, X, res_min=0.1, res_max=2):
    res = (res_min+res_max)/2.
    y_pred = louvain_cluster(X, res = res)
    #print('Resolution: ', res, 'number of clusters: ',len(np.unique(y_pred)))
    if len(np.unique(y_pred))>K:
        res_max = res
        return find_best_res(K, X, res_min=res_min, res_max=res_max)
    elif len(np.unique(y_pred))<K:
        res_min = res
        return find_best_res(K, X, res_min=res_min, res_max=res_max)
    else:
        #print('Best resolution: ', res, 'number of clusters: ',len(np.unique(y_pred)))
        return y_pred
