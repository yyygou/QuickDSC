# !/usr/bin/python
# -*- coding:utf-8 -*-

# author "chen"
# Time 2020/5/20

# run the find_subgraph, attain the density subgraph

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
    
    k = config_dict['k']
    n_clusters = config_dict['n_clusters']
    beta = config_dict['beta']
    ann = config_dict['ann']
    metric = config_dict['metric']

    # 数据集和标签
    data, label_true = Dataprocessing.get_data(DATA_File)
    print(data.shape)

    starttime = datetime.datetime.now()

    model = QuickDSC(k, n_clusters, beta, ann=ann, metric=metric)
    model.fit(data)
    label_pred = model.labels_
    
    endtime = datetime.datetime.now()
    print('total time=', endtime-starttime)

    ARI=metrics.adjusted_rand_score(label_true, label_pred)
    AMI=metrics.adjusted_mutual_info_score(label_true, label_pred)
    NMI=metrics.normalized_mutual_info_score(label_true, label_pred)

    print("Adj. Rand Index Score=" , ARI)
    print("Adj. Mutual Info Score=", AMI)
    print("Norm Mutual Info Score=", NMI)
    