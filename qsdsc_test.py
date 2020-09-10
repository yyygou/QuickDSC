# !/usr/bin/python
# -*- coding:utf-8 -*-

# author "chen"
# Time 2020/5/20

# run the quickshiftPP, attain the density subgraph

import datetime
import Dataprocessing
import numpy as np
from sklearn import metrics
from QuickDSC import QuickDSC
from DPC_class import DPC
 

if __name__ == '__main__':
    # read data file
    config_dict = Dataprocessing.read_config()
    
    # parameter
    DATA_File = config_dict['DATA_File']
    dismision = config_dict['dismision']
    k = config_dict['k']
    n_clusters = config_dict['n_clusters']
    beta = config_dict['beta']
    ann = config_dict['ann']

    # 数据集和标签
    data, label_true = Dataprocessing.get_data(DATA_File, dismision)
    print(data.shape)

    # starttime = datetime.datetime.now()
    model = QuickDSC(k, n_clusters, beta, ann=ann)
    model.find_subgraph(data)
    # endtime = datetime.datetime.now()
    # print('total time=', endtime-starttime)
    # label_pred = model.result
    # print('quickshift remain ARI:', metrics.adjusted_rand_score(label_true, label_pred))
    # print('quickshift remain AMI:', metrics.adjusted_mutual_info_score(label_true, label_pred))
    # print('cluster number:', len(set(label_pred)))
    # Dataprocessing.show_cluster(data, label_pred)
    # radius = model.rho_radius
    # weight = model.delta
    # choosen_id = model.choosen_id
    # Dataprocessing.show_diagram(radius, weight, choosen_id)

    # DS_index = model.DS_index
    # Dataprocessing.show_center(choosen_id, DS_index, data)

    # model.memberships : initial memberships for DS set
    # the knn_radius asc order equal to the denstiy desc order
    y_result = model.result
    y_mode_result = model.mode_result
    y_knn_radius = model.knn_radius
    y_memberships = model.memberships
    print(y_memberships)
    print(y_result)
    # 第二步，找到top-K center
    DS_center = []
    DS_index = []
    center_radius = []
    DS_cluster_id = []
    Importance_value = []
    for i, d in enumerate(y_memberships):
        # center_radius.append(y_knn_radius[i])
        # # d == 1 mean it is a mode
        if d!= -1:
            DS_center.append(data[i])
            center_radius.append(y_knn_radius[i])
        #     DS_index.append(i)
        #     center_radius.append(y_knn_radius[i])
        #     DS_cluster_id.append(y_result[i]) 
        #   
    # Dataprocessing.show_data(DS_center)
    print(len(DS_center))
    # center_radius = np.array(center_radius)
    # rho = 1/center_radius
    # print(rho)

    model_dpc = DPC(3)
    model_dpc.load_paperdata(DS_center)
    model_dpc.fit(DS_center, center_radius)
    c_center = model_dpc.choosen_id
    print(c_center)
    print(model_dpc.dc)
    label_pred = model_dpc.labels_
    # Dataprocessing.show_cluster(DS_center, label_pred)
    center = []
    for c in c_center:
        center.append(DS_center[c])
    print(center)
    Dataprocessing.show_cluster(DS_center, label_pred, center)
    Dataprocessing.show_cluster_data(DS_center, label_pred, center, data)