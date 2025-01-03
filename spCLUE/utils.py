import math
import itertools
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import torch
from torch.backends import cudnn
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import random


def adjust_learning_rate(optimizer, epoch, lr):
    p = {
        'epochs': 500,
        'optimizer': 'sgd',
        'optimizer_kwargs': {
            'nesterov': False,
            'weight_decay': 0.0001,
            'momentum': 0.9,
        },
        'scheduler': 'cosine',
        'scheduler_kwargs': {
            'lr_decay_rate': 0.1
        },
    }

    new_lr = None

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate']**3)
        new_lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            new_lr = lr * (p['scheduler_kwargs']['lr_decay_rate']**steps)

    elif p['scheduler'] == 'constant':
        new_lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return lr


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def batch_refine_label(adata, radius=30, key="label", suffix=None, batch_key="batchID"):
    n_neigh = radius
    old_type = adata.obs[key].values
    batch_keys = list(set(adata.obs[batch_key]))
    new_type_all = []
    for bk in batch_keys:
        new_type = []
        adata_tmp = adata[adata.obs[batch_key] == bk]
        old_type = adata_tmp.obs[key].values
        # calculate distance
        position = adata_tmp.obsm["spatial"]
        distance = cdist(position, position, metric="euclidean")

        n_cell = distance.shape[0]

        for i in range(n_cell):
            vec = distance[i, :]
            index = vec.argsort()
            neigh_type = []
            for j in range(1, n_neigh + 1):
                neigh_type.append(old_type[index[j]])
            max_type = max(neigh_type, key=neigh_type.count)
            new_type.append(max_type)

        suffix_add = "" if suffix is None else "_" + suffix
        new_type_all += [str(i) for i in list(new_type)]
    adata.obs[f"{key}_refined" + suffix_add] = np.array(new_type_all)
    return np.array(new_type_all)


def refine_label(adata, radius=30, key='label', suffix=None):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    #calculate distance
    position = adata.obsm['spatial']
    distance = cdist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    suffix_add = "" if suffix is None else "_" + suffix
    new_type = [str(i) for i in list(new_type)]
    adata.obs[f'{key}_refined' + suffix_add] = np.array(new_type)
    return np.array(new_type)

def calculateMetrics(true, pred, embedding):
    ## return a list of your metrics;
    metric_list = []
    metric_list.append(round(adjusted_rand_score(true, pred), 4))
    metric_list.append(round(normalized_mutual_info_score(true, pred), 4))
    metric_list.append(round(silhouette_score(embedding, pred), 4))
    metric_list.append(round(calinski_harabasz_score(embedding, pred), 4))
    return metric_list

def clustering(adata,
               n_clusters=12,
               radius=30,
               key="embed",
               refinement=False,
               suffix = None,
               cluster_methods=None):
    embedding = adata.obsm[key]
    if cluster_methods is None:
        cluster_methods = "mclust"
    if "kmeans" == cluster_methods:
        kmeans = KMeans(n_clusters=n_clusters,
                        init="k-means++",
                        random_state=2023)
        pred = kmeans.fit_predict(embedding)
        adata.obs["kmeans"] = pred
    elif cluster_methods == "leiden":
        res = searchRes(adata, n_clusters)
        adata.uns["LeidenRes"] = res
        sc.tl.leiden(adata, resolution=res, random_state=2023)
        pred = adata.obs["leiden"].to_numpy(dtype=np.int64)
    elif cluster_methods == "mclust":
        pred = mclust_R(embedding, n_clusters)
        adata.obs["mclust"] = pred
    elif cluster_methods == "pred":
        pred = adata.obs["pred"].to_numpy(dtype=np.int64)
    else:
        raise Exception(
            "please specify correct method !!! \n[ kmeans / leiden / mclust / pred ]"
        )

    if refinement:
        refined_labels = refine_label(adata, radius, key=cluster_methods, suffix=suffix)
    return adata


def mclust_R(data, n_clusters, modelNames="EEE", random_seed=2023):

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data), n_clusters, modelNames)
    mclust_res = np.array(res[-2])

    return mclust_res.astype(np.int32) - 1


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def searchRes(adata, fixed_clus_count, increment=0.01):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]

        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.02, 2.5, increment))):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        print(f"{res}: {count_unique_leiden}")
        if count_unique_leiden >= fixed_clus_count:
            break
    return res


############################### ============  MNN utils ======================== [from scDML]
def nn(ds1, ds2, names1, names2, knn=50, metric_p=2, return_distance=False, metric="cosine", flag="in"):
    # Find nearest neighbors of first dataset.
    if (flag == "in"):
        nn_ = NearestNeighbors(n_neighbors=knn, metric=metric)  # remove self
        nn_.fit(ds2)
        nn_distances, ind = nn_.kneighbors(ds1, return_distance=True)
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[1:]:
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b[1:]):
                    match[(names1[a], names2[b_i])] = nn_distances[a, b_ind + 1]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match
    else:
        nn_ = NearestNeighbors(n_neighbors=knn, metric=metric)  # remove self
        nn_.fit(ds2)
        nn_distances, ind = nn_.kneighbors(ds1, return_distance=True)
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[0:]:
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b):
                    match[(names1[a], names2[b_i])] = nn_distances[a, b_ind]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match


### - this function requires the [hnswlib] package; `import hnswlib`
def nn_approx(ds1, ds2, names1, names2, knn=50, return_distance=False, metric="cosine", flag="in"):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    if (metric == "euclidean"):
        tree = hnswlib.Index(space="l2", dim=dim)
    elif (metric == "cosine"):
        tree = hnswlib.Index(space="cosine", dim=dim)
    #square loss: 'l2' : d = sum((Ai - Bi) ^ 2)
    #Inner  product 'ip': d = 1.0 - sum(Ai * Bi)
    #Cosine similarity: 'cosine':d = 1.0 - sum(Ai * Bi) / sqrt(sum(Ai * Ai) * sum(Bi * Bi))
    tree.init_index(max_elements=num_elements, ef_construction=200,
                    M=32)  # refer to https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md for detail
    tree.set_ef(50)
    tree.add_items(ds2)
    ind, distances = tree.knn_query(ds1, k=knn)
    if (flag == "in"):
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[1:]:  ##
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b[1:]):
                    match[(names1[a], names2[b_i])] = distances[a, b_ind + 1]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match
    else:
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[0:]:  ##
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b):
                    match[(names1[a], names2[b_i])] = distances[a, b_ind]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match


### - this function requires the [annoy] package; `from annoy import AnnoyIndex`
def nn_annoy(ds1, ds2, names1, names2, knn=20, save=True, return_distance=False, metric="cosine", flag="in"):
    """ Assumes that Y is zero-indexed. """
    # Build index.
    if (metric == "cosine"):
        tree = AnnoyIndex(ds2.shape[1], metric="angular")  #metric
        tree.set_seed(100)
    else:
        tree = AnnoyIndex(ds2.shape[1], metric=metric)  #metric
        tree.set_seed(100)
    if save:
        tree.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        tree.add_item(i, ds2[i, :])
    tree.build(60)  #n_trees=50
    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(tree.get_nns_by_vector(ds1[i, :], knn, search_k=-1))  #search_k=-1 means extract search neighbors
    ind = np.array(ind)
    # Match.
    if (flag == "in"):
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[1:]:
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            # get distance
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[1:]:
                    match[(names1[a], names2[b_i])] = tree.get_distance(a, b_i)
            return match
    else:
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[0:]:
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            # get distance
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b:
                    match[(names1[a], names2[b_i])] = tree.get_distance(a, b_i)
            return match


### - change ${approx}=True to use approximation algorithms -->
def mnn(ds1,
        ds2,
        names1,
        names2,
        knn=3,
        save=False,
        approx=False,
        approx_method="hnswlib",
        return_distance=False,
        metric="cosine",
        flag="in"):
    # Find nearest neighbors in first direction.

    if approx:
        if approx_method == "hnswlib":
            #hnswlib
            match1 = nn_approx(ds1,
                               ds2,
                               names1,
                               names2,
                               knn=knn,
                               return_distance=return_distance,
                               metric=metric,
                               flag=flag)  # save_on_disk = save_on_disk)
            # Find nearest neighbors in second direction.
            match2 = nn_approx(ds2,
                               ds1,
                               names2,
                               names1,
                               knn=knn,
                               return_distance=return_distance,
                               metric=metric,
                               flag=flag)  # , save_on_disk = save_on_disk)
        else:
            #annoy
            match1 = nn_annoy(ds1,
                              ds2,
                              names1,
                              names2,
                              knn=knn,
                              save=save,
                              return_distance=return_distance,
                              metric=metric,
                              flag=flag)  # save_on_disk = save_on_disk)
            # Find nearest neighbors in second direction.
            match2 = nn_annoy(ds2,
                              ds1,
                              names2,
                              names1,
                              knn=knn,
                              save=save,
                              return_distance=return_distance,
                              metric=metric,
                              flag=flag)  # , save_on_disk = save_on_disk)

    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn, return_distance=return_distance, metric=metric, flag=flag)
        match2 = nn(ds2, ds1, names2, names1, knn=knn, return_distance=return_distance, metric=metric, flag=flag)
    # Compute mutual nearest neighbors.
    if (flag == "in"):
        if not return_distance:
            # ${match}s are set
            mutual = match1 | set([(b, a) for a, b in match1])
            return mutual
        else:
            # ${match}s are dict
            mutual = []
            distances = []
            for a, b in match1.keys():
                mutual.append((a, b))
                mutual.append((b, a))
                distances.append(match1[(a, b)])
                distances.append(match1[(a, b)])
            return mutual, distances
    else:
        if not return_distance:
            # mutuals are set
            mutual = match1 & set([(b, a) for a, b in match2])
            ####################################################
            # change mnn pair to symmetric
            mutual = mutual | set([(b, a) for (a, b) in mutual])
            ####################################################
            return mutual
        else:
            # mutal are set
            mutual = set([(a, b) for a, b in match1.keys()]) & set([(b, a) for a, b in match2.keys()])
            ## more_in_symm = (mutual | set([(b, a) for (a, b) in mutual])) - mutual
            mutual = list(mutual)
            #distance list of numpy array
            distances = []
            for element_i in mutual:
                distances.append(match1[element_i])
            # for b, a in more_in_symm:
            #     distances.append(match1[(a, b)])
            # mutual += more_in_symm
            return mutual, distances


## - calculate KNN and MNN from data_matrix(embedding matrix), not anndata
def get_dict_mnn(data_matrix,
                 batch_index,
                 k=5,
                 save=True,
                 approx=False,
                 approx_method="hnswlib",
                 verbose=False,
                 return_distance=False,
                 metric="cosine",
                 flag="in",
                 log=None):
    '''
    data_matrix: ndarray, [m1 + m2 + m3 + m4, d];
    '''
    cell_names = np.array(range(len(data_matrix)))
    #batch_list = adata.obs[batch_key] if batch_key in adata.obs.columns else np.ones(adata.shape[0], dtype=str)
    batch_unique = np.unique(batch_index)
    cells_batch = []
    for i in batch_unique:
        cells_batch.append(cell_names[batch_index == i])
    mnns = set()
    mnns_distance = []
    if (flag == "in"):
        num_KNN = 0
        ## print some information;
        print(f"Calculate KNN pair intra batch...........")
        print(f"number of knn: {k}")
        print(f"metric of distance is: {metric}")
        for comb in list(itertools.combinations(range(len(cells_batch)), 1)):
            ## comb = (0,)
            i = comb[0]  # ith batch
            j = comb[0]  # ith batch
            print(f"Processing datasets: ({batch_unique[i]}, {batch_unique[j]})")
            target = list(cells_batch[j])
            ref = list(cells_batch[i])
            #ds1 = adata[target].obsm[dr_name]
            ds1 = data_matrix[target]
            ds2 = data_matrix[ref]
            names1 = target
            names2 = ref
            match = mnn(ds1,
                        ds2,
                        names1,
                        names2,
                        knn=k,
                        save=save,
                        approx=approx,
                        approx_method=approx_method,
                        return_distance=return_distance,
                        metric=metric,
                        flag=flag)
            mnns = mnns | match
            # mnns_distance.append(distances) # not need
            print(f"There are ({len(match)}) KNN pairs when processing ({batch_unique[i]}, {batch_unique[j]})")
            num_KNN += len(match)
        print(f"Total number of KNN pairs is {num_KNN}.")
        if not return_distance:
            return list(zip(*list(mnns)))
        else:
            return mnns, mnns_distance
    else:
        num_MNN = 0
        print(f"Calculate MNN pair inter batch...........")
        print(f"number of knn: {k}")
        print(f"metric of distance is: {metric}")
        for comb in list(itertools.combinations(range(len(cells_batch)), 2)):
            # comb = (2,3)
            i = comb[0]  # i batch
            j = comb[1]  # jth batch
            print(f"Processing datasets: ({batch_unique[i]}, {batch_unique[j]})")
            target = list(cells_batch[j])
            ref = list(cells_batch[i])
            ds1 = data_matrix[target]
            ds2 = data_matrix[ref]
            names1 = target
            names2 = ref
            match = mnn(ds1,
                        ds2,
                        names1,
                        names2,
                        knn=k,
                        save=save,
                        approx=approx,
                        approx_method=approx_method,
                        return_distance=return_distance,
                        metric=metric,
                        flag=flag)
            mnns = mnns | match
            # mnns_distance.append(distances)
            print(f"There are ({len(match)}) MNN pairs when processing ({batch_unique[i]}, {batch_unique[j]})")
            num_MNN += len(match)
        print(f"Total number of KNN pairs is {num_MNN}.")
        if not return_distance:
            return list(zip(*list(mnns)))
        else:
            return mnns, mnns_distance


def convertSet2Coo(graph: list, n_spots: int):
    ## sp.coo_matrix((data, (rows, cols)), shape=(m, n))
    return sp.coo_matrix(([1.] * len(graph[0]), (graph[0], graph[1])), shape=(n_spots, n_spots))
