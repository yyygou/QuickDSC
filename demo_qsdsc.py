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
    model.fit(data)
    # endtime = datetime.datetime.now()
    # print('total time=', endtime-starttime)
    label_pred = model.labels_
    print('quickshift remain ARI:', metrics.adjusted_rand_score(label_true, label_pred))
    print('quickshift remain AMI:', metrics.adjusted_mutual_info_score(label_true, label_pred))

    # Dataprocessing.show_cluster(data, label_pred)
    # radius = model.rho_radius
    # weight = model.delta
    # choosen_id = model.choosen_id
    # Dataprocessing.show_diagram(radius, weight, choosen_id)

    # DS_index = model.DS_index
    # Dataprocessing.show_center(choosen_id, DS_index, data)