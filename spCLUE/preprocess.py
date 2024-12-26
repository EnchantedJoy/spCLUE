import scanpy as sc
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import numpy as np
import scipy.sparse as sp



def preprocess(adata, hvgNumber=None):
    print("normalized data ---------------->")
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    if not hvgNumber is None:
        print(f"========== selecting HVG ============")
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="count",n_top_genes=hvgNumber, subset=False)
        adata = adata[:, adata.var["highly_variable"] == True]
        sc.pp.scale(adata)
        return adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    return adata


def calcGAEParams(graph, n_samples):
    '''graph is a bipartite graph, return pos_weight and norm_val
    '''
    non_zero_cnt = graph.sum()
    norm_val = (n_samples * n_samples) / (2 * (n_samples * n_samples - non_zero_cnt))
    pos_weight = (n_samples * n_samples - non_zero_cnt) / non_zero_cnt
    return norm_val, pos_weight


def calcGraphWeight(coor, eps=1e-6):
    dist = cdist(coor, coor, "euclidean")
    dist = dist / (np.max(dist) + eps)
    return dist


def correlation_graph(A, B):
    '''calculate correlation between A and B.
    Args:
        A (np.ndarray): sample matrix, shape: [samples, features].
        B (np.ndarray): sample matrix, shape: [samples, features].
    Returns: 
        corr (np.ndarray): correlation matrix of features, shape: [features, features].
    '''
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm / (np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T * np.sqrt(np.sum(bm**2, axis=0, keepdims=True)))


def prepare_graph(adata, key="spatial", n_neighbors=12, n_comps=50, eps=1e-8, svd_solver="randomized", self_weight=0.3):
    n_spots = adata.shape[0]
    assert key in ["spatial", "expr"], "case should be [spatial] or [expr]"
    if key == "spatial":
        print("create adjacent matrix from spatial idx --------------->")
        expr = adata.obsm[key]
        weights = 1. / (cdist(expr, expr, "euclidean") + eps)
    else:
        print("create adjacent matrix from pca expr --------------->")
        expr = PCA(n_components=n_comps, random_state=0, svd_solver=svd_solver).fit_transform(adata.X)
        weights = correlation_graph(expr.T, expr.T)

    print("create knn graph ---->")
    threshold = np.sort(weights)[:, -n_neighbors - 1:-n_neighbors]
    weights[weights < threshold] = 0
    weights = (weights + weights.T) / 2
    weights = weights * (1 - np.eye(n_spots))  # drop the diag

    adjFilter = 0. if key == "spatial" else 0.1
    # convert to bipartite case
    adjBip = np.where(weights > adjFilter, 1, 0)
    print(f"{key} knn graph created ----<")

    return sp.coo_matrix(symm_norm(adjBip, weightDiag=self_weight))

def symm_norm(adj, weightDiag=.3, eps=1e-8):
    '''
    args: adjacent matrix with diag = 0
    return: D^{-1/2} (A + I) D^{-1 / 2}
    '''
    n_spot = adj.shape[0]
    adj_self = (1 - weightDiag) * adj + np.eye(n_spot) * weightDiag  
    degrees = 1. / np.sqrt((np.sum(adj_self, axis=1) + eps))
    adj_self *= degrees
    adj_self *= degrees[:, None]
    return adj_self.astype(np.float32)
