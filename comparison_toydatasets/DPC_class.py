#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import math
import numpy as np
from plot import plot_rho_delta
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


def maxminnorm(array):
    '''
    normalization function: aim to normalize the rho and delta
    '''
    x_max, x_min = np.max(array, 0), np.min(array, 0)
    array = (array - x_min)/(x_max - x_min)
    return array


def dist(vec1, vec2):
        # return math.sqrt(math.pow(vec1[0] - vec2[0], 2) + math.pow(vec1[1] - vec2[1],2))
        return math.pow(vec1[0] - vec2[0], 2) + math.pow(vec1[1] - vec2[1],2)

def select_dc(max_id, max_dis, min_dis, distances, auto=True):
    '''
    Select the local density threshold, default is the method used in paper, auto is `autoselect_dc`
    Args:
            max_id    : max continues id
            max_dis   : max distance for all points
            min_dis   : min distance for all points
            distances : distance dict
            auto      : use auto dc select or not
    Returns:
        dc that local density threshold
    '''
    if auto:
        return autoselect_dc(max_id, max_dis, min_dis, distances)
    percent = 2.0
    position = int(max_id * (max_id + 1) / 2 * percent / 100)
    dc = sorted(distances.values())[position * 2 + max_id]
    return dc


def autoselect_dc(max_id, max_dis, min_dis, distances):
    '''
    Auto select the local density threshold that let average neighbor is 1-2 percent of all nodes.

    Args:
            max_id    : max continues id
            max_dis   : max distance for all points
            min_dis   : min distance for all points
            distances : distance dict

    Returns:
        dc that local density threshold
    '''
    dc = (max_dis + min_dis) / 2

    while True:
        nneighs = sum([1 for v in distances.values() if v < dc]) / max_id ** 2
        if nneighs >= 0.01 and nneighs <= 0.02:
            break
        # binary search
        if nneighs < 0.01:
            min_dis = dc
        else:
            max_dis = dc
        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break
    return dc


def min_distance(X, dc, max_id, max_dis, distances, rho):
    '''
    Compute all points' min distance to the higher local density point(which is the nearest neighbor)
    Args:
            max_id    : max continues id
            max_dis   : max distance for all points
            distances : distance dict
            rho       : local density vector that index is the point index that start from 1
    Returns:
        min_distance vector, nearest neighbor vector
    '''
    delta, nneigh = [float(max_dis)] * len(rho), [-1] * len(rho)
    rho = np.array(rho)
    sort_rho_idx = np.argsort(-rho)
    for i in range(0, max_id + 1):
        for j in range(0, i):
            old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
            if distances[(old_i, old_j)] < delta[old_i]:
                delta[old_i] = distances[(old_i, old_j)]
                nneigh[old_i] = old_j
    
    # X = np.array(X)
    # kdt = KDTree(X, metric='euclidean')
    # # ind = kdt.query_radius(X, r=dc)
    # n = int(2*max_id/100)
    # dist, ind = kdt.query(X, k=n)
    # for i in range(max_id + 1):
    #     next = -1
    #     best_distance = max_dis
    #     for k in ind[i]:
    #         if i == k:
    #             continue
    #         if rho[i] > rho[k]:
    #             continue
    #         dt = distances[(i, k)]
    #         if best_distance > dt:
    #             best_distance = dt
    #             next = k
    #             break
    #     if next < 0:
    #         for j in range(max_id + 1):
    #             if i == j:
    #                 continue
    #             if rho[i] > rho[j]:
    #                 continue
    #             dt = distances[(i, j)]
    #             if best_distance > dt:
    #                 best_distance = dt
    #                 next = j

    #     delta[i] = best_distance
    #     nneigh[i] = next
    return np.array(delta, np.float32), np.array(nneigh, np.float32)


class DPC:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def load_paperdata(self, data):
        '''
        Load distance from data
        Args:
            distance_f : distance file, the format is column1-index 1, column2-index 2, column3-distance
        Returns:
            distances dict, max distance, min distance, max continues id
        '''
        distances = {}
        min_dis, max_dis = sys.float_info.max, 0.0
        max_id = len(data) - 1

        for i in range(max_id):
            for j in range(i+1, max_id+1):
                dis = float(format(dist(data[i], data[j]), '.3f'))
                # dis = dist(data[i], data[j])
                min_dis, max_dis = min(min_dis, dis), max(max_dis, dis)
                distances[(i, j)] = dis
                distances[(j, i)] = dis

        for i in range(max_id + 1):
            distances[(i, i)] = 0.0

        self.distances = distances
        self.max_dis = max_dis
        self.min_dis = min_dis
        self.max_id = max_id
        

    def local_density(self, dc=None, auto_select_dc=False, guass=True, cutoff=False):
        '''
        Just compute local density
        Args:
            dc            : local density threshold, call select_dc if dc is None
            autoselect_dc : auto select dc or not
            gauss     : use guass func or not(can't use together with cutoff)
            cutoff    : use cutoff func or not(can't use together with guass)
        Returns:
            local density vector that index is the point index that start from 0
        '''
        assert not (dc is not None and auto_select_dc)
        if dc is None:
            dc = select_dc(self.max_id, self.max_dis, self.min_dis, self.distances, auto=auto_select_dc)
        
        assert guass ^ cutoff
        guass_func = lambda dij, dc: math.exp(- (dij / dc) ** 2)
        cutoff_func = lambda dij, dc: 1 if dij < dc else 0
        func = guass and guass_func or cutoff_func
        rho = [0] * (self.max_id + 1)
        for i in range(0, self.max_id):
            for j in range(i + 1, self.max_id + 1):
                rho[i] += func(self.distances[(i, j)], dc)
                rho[j] += func(self.distances[(i, j)], dc)
        
        self.dc = dc
        self.rho = rho
        return np.array(rho, np.float32)
        

    def remain_clustering(self, choosen_id, nneigh):
        label_pred = [-1 for i in range(len(nneigh))]
        parent_map = {}
        
        for i, p_id in enumerate(nneigh):
            if p_id not in parent_map:
                parent_map[p_id] = [i]
            else:
                parent_map[p_id].append(i)
        
        center_cluster = {}
        for c in choosen_id:
            center_cluster[c] = []
            if c in parent_map:
                children = parent_map[c]
            else:
                continue
            for child in children:
                if child in choosen_id:
                    continue
                center_cluster[c].append(child)
                if child in parent_map:
                    children.extend(parent_map[child])
        
        count = 0
        for c in center_cluster:
            label_pred[c] = count
            for child in center_cluster[c]:
                label_pred[child] = count
            count += 1
       
        self.labels_ = np.array(label_pred)


    def fit(self, data):
        self.load_paperdata(data)
        self.local_density(auto_select_dc=False)
        delta, nneigh = min_distance(data, self.dc, self.max_id,self. max_dis, self.distances, self.rho)
        # plot_rho_delta(rho, delta)  # plot to choose the threthold

        rho = maxminnorm(self.rho)
        delta = maxminnorm(delta)

        choosen_id = []
        # graph—value = 密度值 * 距离，graph-sorted是按降序排列
        graph_value = np.zeros(len(rho))
        for idx, (ldensity, mdistance) in enumerate(zip(rho, delta)):
            graph_value[idx] = ldensity * mdistance

        graph_sorted = np.argsort(-graph_value)
        # 密度和距离两者乘积取前k个点作为中心点
        for i in range(self.n_clusters):
            idx = graph_sorted[i]
            choosen_id.append(idx)
        
        self.choosen_id = choosen_id

        self.remain_clustering(choosen_id, nneigh)
        
