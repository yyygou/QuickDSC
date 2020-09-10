# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
cimport numpy as np
from sklearn.neighbors import KDTree, BallTree
from sklearn.neighbors.kde import KernelDensity
import scipy
import math
import sys
from annoy import AnnoyIndex
import datetime
import copy


cdef extern from "mutual_neighborhood_graph.h":
    void compute_mutual_knn(int n, int k, int d,
                    double * dataset,
                    double * radii,
                    int * neighbors,
                    double beta,
                    double epsilon,
                    int * memberships,
                    int * result_parent,
                    double * mode_result)
    void cluster_remaining(int n, int k, int d,
                   double * dataset,
                   double * radii,
                   int * neighbors,
                   int * initial_memberships,
                   int * result)
    void center_remaining(int n, int k, int d,
                   double * dataset,
                   double * radii,
                   int * neighbors,
                   int * parent,
                   double * weight)



cdef compute_mutual_knn_np(n, k, d,
                    np.ndarray[double,  ndim=2, mode="c"] dataset,
                    np.ndarray[double,  ndim=1, mode="c"] radii,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] neighbors,
                    beta,
                    epsilon,
                    np.ndarray[np.int32_t, ndim=1, mode="c"] memberships,
                    np.ndarray[np.int32_t, ndim=1, mode="c"] result_parent,
                    np.ndarray[double,  ndim=1, mode="c"] mode_result):
    compute_mutual_knn(n, k, d,
                        <double *> np.PyArray_DATA(dataset),
                        <double *> np.PyArray_DATA(radii),
                        <int *> np.PyArray_DATA(neighbors),
                        beta, epsilon,
                        <int *> np.PyArray_DATA(memberships),
                        <int *> np.PyArray_DATA(result_parent),
                        <double *> np.PyArray_DATA(mode_result)
                        ) 

cdef cluster_remaining_np(n, k, d,
                    np.ndarray[double,  ndim=2, mode="c"] dataset,
                    np.ndarray[double,  ndim=1, mode="c"] radii,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] neighbors,
                    np.ndarray[np.int32_t, ndim=1, mode="c"] initial_memberships,
                    np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    cluster_remaining(n, k, d,
                    <double *> np.PyArray_DATA(dataset),
                    <double *> np.PyArray_DATA(radii),
                    <int *> np.PyArray_DATA(neighbors),
                    <int *> np.PyArray_DATA(initial_memberships),
                    <int *> np.PyArray_DATA(result)
                    )

cdef center_remaining_np(n, k, d,
                    np.ndarray[double,  ndim=2, mode="c"] dataset,
                    np.ndarray[double,  ndim=1, mode="c"] radii,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] neighbors,
                    np.ndarray[np.int32_t, ndim=1, mode="c"] parent,
                    np.ndarray[double,  ndim=1, mode="c"] weight):
    center_remaining(n, k, d,
                    <double *> np.PyArray_DATA(dataset),
                    <double *> np.PyArray_DATA(radii),
                    <int *> np.PyArray_DATA(neighbors),
                    <int *> np.PyArray_DATA(parent),
                    <double *> np.PyArray_DATA(weight))


def maxminnorm(array):
    '''
    normalization function: aim to normalize the rho and delta
    '''
    x_max, x_min = np.max(array, 0), np.min(array, 0)
    array = (array - x_min)/(x_max - x_min)
    return array


class RPTrees:
    def __init__(self, X, metric = 'euclidean'):
        n, d = X.shape
        self.rptree = AnnoyIndex(d, metric)
        for i in range(n):
            self.rptree.add_item(i, X[i])

        self.rptree.build(10)


    def query(self, X, k):
        n, d = X.shape
        neighbors = []
        knn_radius = []
        if k > len(X):
            k = len(X)
        for i in range(n):
            i_neighbors, i_dists = self.rptree.get_nns_by_item(i, k, include_distances=True)
            # print (i, ' ', k , ' ', len(i_neighbors), ' ', i_neighbors[1:], i_dists[1:])
            neighbors.append(i_neighbors[0:])
            knn_radius.append(i_dists[k-1])

        return np.array(neighbors), np.array(knn_radius)


class QuickDSC:
    """
    Parameters
    ----------
    
    k: The number of neighbors (i.e. the k in k-NN density)

    beta: Ranges from 0 to 1. We choose points that have kernel density of at
        least (1 - beta) * F where F is the mode of the empirical density of
        the cluster

    epsilon: For pruning. Sets how much deeper in the cluster tree to look
        in order to connect clusters together. Must be at least 0.

    Attributes
    ----------

    cluster_map: a map from the cluster (zero-based indexed) to the list of points
        in that cluster

    """

    def __init__(self, k, n_clusters, beta, ann="kdtree", metric = "euclidean", epsilon=0):
        self.k = k
        self.n_clusters = n_clusters
        self.beta = beta
        self.ann = ann
        self.metric = metric
        self.epsilon = epsilon

    def query_neighbor(self, k, ann, data, metric='euclidean'):
        if ann == "kdtree":
            kdt = KDTree(data, metric=metric)
            query_res = kdt.query(data, k=k)
            knn_radius = query_res[0][:, k-1]
            neighbors = query_res[1]
        elif ann == "balltree":
            balltree = BallTree(data, metric=metric)
            query_res = balltree.query(data, k=k)
            knn_radius = query_res[0][:, k - 1]
            neighbors = query_res[1]
        elif ann == "rptree":
            rptree = RPTrees(data,  metric=metric)
            neighbors, knn_radius = rptree.query(data, k=k)
        return neighbors, knn_radius

    def find_the_nneigh(self, center, choosen_id, ann, metric):
        nneigh = [-1 for i in range(len(center))]
        for i in choosen_id:
            nneigh[i] = i
        center = np.array(center)
        k = len(center)
        
        neighbors, knn_radius = self.query_neighbor(k, ann, center, metric=metric)

        for i in range(len(center)):
            if nneigh[i] != -1:
                continue
            for j in neighbors[i]:
                # nneigh[j] == j means it is a choosen center
                if nneigh[j] == j:
                    nneigh[i] = j
                    break
        return nneigh

    def find_cc(self, X):
        """
        Modified by YYY.
        Determines the clusters in two steps.
        First step is to compute the knn density estimate and
        distances. This is done using kd tree
        Second step is to build the knn neighbor graphs
        Updates the cluster count and membership attributes

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint in 
            euclidean space
        """
        X = np.array(X)
        n, d = X.shape
        knn_density = None
        neighbors = None

        neighbors, knn_radius = self.query_neighbor(self.k, self.ann, X, metric=self.metric)
            
        memberships = np.zeros(n, dtype=np.int32)
        result = np.zeros(n, dtype=np.int32)
        result_parent = np.zeros(n, dtype=np.int32)
        neighbors = np.ndarray.astype(neighbors, dtype=np.int32)
        knn_radius = np.ndarray.astype(knn_radius, dtype=np.float64)
        mode_result = np.zeros(n, dtype=np.float64)
        X_copy = np.ndarray.astype(X, dtype=np.float64)

        compute_mutual_knn_np(n, self.k, d,X_copy,knn_radius,neighbors,self.beta, self.epsilon,
                            memberships,
                            result_parent,
                            mode_result
                            )
        knn_radius = np.ndarray.astype(knn_radius, dtype=np.float64)


        self.knn_radius = knn_radius
        self.memberships = memberships
        self.result_parent = result_parent
        self.mode_result = mode_result


    def compute_weight(self, X, Density=None):
        X = np.array(X)
        n, d = X.shape

        temp_k = self.k
        if len(X) < self.k:
            temp_k = len(X)

        neighbors, knn_radius = self.query_neighbor(temp_k, self.ann, X, metric=self.metric)

        if Density is not None:
            knn_radius = Density

        neighbors = np.ndarray.astype(neighbors, dtype=np.int32)
        knn_radius = np.ndarray.astype(knn_radius, dtype=np.float64)
        parent = np.zeros(n, dtype=np.int32)
        weight = np.zeros(n, dtype=np.float64)
        X_copy = np.ndarray.astype(X, dtype=np.float64)

        center_remaining_np(n, temp_k, d, X_copy, knn_radius, neighbors, parent, weight)

        self.weight = weight
        self.parent = parent


    def find_subgraph(self, X):
        """
        Determines the clusters in two steps.
        First step is to compute the knn density estimate and
        distances. This is done using kd tree
        Second step is to build the knn neighbor graphs
        Updates the cluster count and membership attributes

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint in 
            euclidean space

        Density: knn_radius matrix. every point's knn_radius in the original
            dataset

        """
        X = np.array(X)
        n, d = X.shape
        knn_density = None
        neighbors = None
        
        neighbors, knn_radius = self.query_neighbor(self.k, self.ann, X, metric=self.metric)
   
        memberships = np.zeros(n, dtype=np.int32)
        result_parent = np.zeros(n, dtype=np.int32)
        result = np.zeros(n, dtype=np.int32)
        neighbors = np.ndarray.astype(neighbors, dtype=np.int32)
        knn_radius = np.ndarray.astype(knn_radius, dtype=np.float64)
        mode_result = np.zeros(n, dtype=np.float64)
        X_copy = np.ndarray.astype(X, dtype=np.float64)

        compute_mutual_knn_np(n, self.k, d, X_copy,knn_radius,neighbors,self.beta, self.epsilon,
                            memberships,
                            result_parent,
                            mode_result)

        cluster_remaining_np(n, self.k, d, X_copy, knn_radius, neighbors, memberships, result)

        self.knn_radius = knn_radius
        self.result = result
        self.mode_result = mode_result
        self.memberships = memberships


    def fit(self, data):
        # 第一步，得到density subgraph
        self.find_subgraph(data)

        # model.memberships : initial memberships for DS set
        # the knn_radius asc order equal to the denstiy desc order
        y_result = self.result
        y_mode_result = self.mode_result
        y_knn_radius = self.knn_radius
        y_memberships = self.memberships
        

        # 第二步，找到top-K center
        DS_center = []
        DS_index = []
        center_radius = []
        DS_cluster_id = []
        Importance_value = []

        for i, d in enumerate(y_mode_result):
            # d == 1 mean it is a mode
            if d== 1:
                DS_center.append(data[i])
                DS_index.append(i)
                center_radius.append(y_knn_radius[i])
                DS_cluster_id.append(y_result[i])

        self.subgraph_number = len(DS_center)
        
        center_radius = np.array(center_radius)
        self.compute_weight(DS_center, center_radius)
        weight = self.weight
       
        for i,w in enumerate(weight):
            if w == 1000000000.0:
                weight[i] = 0
                weight[i] = np.max(weight)*1.2
                break
        
        # print(center_radius)
        # print(weight)
        center_radius_copy = copy.deepcopy(center_radius)
        center_radius = maxminnorm(center_radius)
        weight = maxminnorm(weight)

        for r, w in zip(center_radius, weight):
            Importance_value.append(w/r)
            
        Importance_value = np.array(Importance_value)
        Importance_value_sorted = np.argsort(-Importance_value)
        
        choosen_center = []
        choosen_id = []
        choosen_original_id = []
        for i in range(self.n_clusters):
            idx = Importance_value_sorted[i]
            choosen_id.append(idx)
            choosen_center.append(DS_center[idx])
            choosen_original_id.append(DS_index[idx])
        

        # 第三步，对非top-K center的点按照quickshiftPP remain的方法找到最近的DS
        parent_map = {}
        center_parent = self.parent
        
        for i, p_id in enumerate(center_parent):
            if p_id not in parent_map:
                parent_map[p_id] = [i]
            else:
                parent_map[p_id].append(i)
        
        center_cluster = {}

        # 第一种方法，用remain
        for c in choosen_id:
            center_cluster[c] = [c]
            if c not in parent_map:
                continue
            children = parent_map[c]
            for child in children:
                if child not in choosen_id:
                    center_cluster[c].append(child)
                    if child in parent_map:
                        children.extend(parent_map[child])    
        
        
        # 第二种方法，用K-Mode，找最近的center
        # for c in choosen_id:
        #     center_cluster[c] = []
        # nneigh = self.find_the_nneigh(DS_center, choosen_id, self.ann, self.metric)
        # for i, n in enumerate(nneigh):
        #     # i in center_cluster means it is a choosen center
        #     if i in center_cluster:
        #         continue
        #     center_cluster[n].append(i)
       
        count = 0
        label_map = {}
        for c in center_cluster:
            label_map[DS_cluster_id[c]] = count
            for child in center_cluster[c]:
                label_map[DS_cluster_id[child]] = count
            count += 1

        label_pred = []
        for y in y_result:
            label_pred.append(label_map[y])

        cc_set = []
        for i, y in enumerate(y_memberships):
            if y>=0:
                cc_set.append(i)
        self.labels_ = np.array(label_pred)
        self.rho_radius = center_radius_copy
        self.delta = weight
        self.DS_index = DS_index
        self.choosen_id = choosen_id
        self.cc_set = cc_set
        self.choosen_original_id = choosen_original_id

    def build_decision_map(self, data, n_clusters):
        # model.memberships : initial memberships for DS set
        # the knn_radius asc order equal to the denstiy desc order
        y_result = self.result
        y_mode_result = self.mode_result
        y_knn_radius = self.knn_radius
        y_memberships = self.memberships
        

        # 第二步，找到top-K center
        DS_center = []
        DS_index = []
        center_radius = []
        DS_cluster_id = []
        Importance_value = []

        for i, d in enumerate(y_mode_result):
            # d == 1 mean it is a mode
            if d== 1:
                DS_center.append(data[i])
                DS_index.append(i)
                center_radius.append(y_knn_radius[i])
                DS_cluster_id.append(y_result[i])

        DS_center = np.array(DS_center)
        center_radius = np.array(center_radius)
        
        self.subgraph_number = len(DS_center)
        print('subgraph number', len(DS_center))
        
        self.compute_weight(DS_center, center_radius)
        weight = self.weight
       
        for i,w in enumerate(weight):
            if w == 1000000000.0:
                weight[i] = 0
                weight[i] = np.max(weight) + 10
                break
        
        # print(center_radius)
        # print(weight)
        center_radius_copy = copy.deepcopy(center_radius)
        center_radius = maxminnorm(center_radius)
        weight = maxminnorm(weight)

        for r, w in zip(center_radius, weight):
            Importance_value.append(w/r)
            
        Importance_value = np.array(Importance_value)
        Importance_value_sorted = np.argsort(-Importance_value)
        
        choosen_center = []
        choosen_id = []
        choosen_original_id = []
        for i in range(n_clusters):
            idx = Importance_value_sorted[i]
            choosen_id.append(idx)
            choosen_center.append(DS_center[idx])
            choosen_original_id.append(DS_index[idx])
        

        # 第三步，对非top-K center的点按照quickshiftPP remain的方法找到最近的DS
        parent_map = {}
        center_parent = self.parent
        
        for i, p_id in enumerate(center_parent):
            if p_id not in parent_map:
                parent_map[p_id] = [i]
            else:
                parent_map[p_id].append(i)
        
        center_cluster = {}

        # 第一种方法，用remain
        for c in choosen_id:
            center_cluster[c] = [c]
            if c not in parent_map:
                continue
            children = parent_map[c]
            for child in children:
                if child not in choosen_id:
                    center_cluster[c].append(child)
                    if child in parent_map:
                        children.extend(parent_map[child])    
        

        # 第二种方法，用K-Mode，找最近的center
        #for c in choosen_id:
        #    center_cluster[c] = []
        #nneigh = self.find_the_nneigh(DS_center, choosen_id, self.ann, self.metric)
        #for i, n in enumerate(nneigh):
        #    # i in center_cluster means it is a choosen center
        #    if i in center_cluster:
        #        continue
        #    center_cluster[n].append(i)
 
       
        count = 0
        label_map = {}
        for c in center_cluster:
            label_map[DS_cluster_id[c]] = count
            for child in center_cluster[c]:
                label_map[DS_cluster_id[child]] = count
            count += 1

        label_pred = []
        for y in y_result:
            label_pred.append(label_map[y])

        cc_set = []
        for i, y in enumerate(y_memberships):
            if y>=0:
                cc_set.append(i)
        self.labels_ = np.array(label_pred)
        self.rho_radius = center_radius_copy
        self.delta = weight
        self.DS_index = DS_index
        self.choosen_id = choosen_id
        self.cc_set = cc_set
        self.choosen_original_id = choosen_original_id
        return np.array(label_pred)
        