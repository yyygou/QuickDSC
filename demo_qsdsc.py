# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.  

# !/usr/bin/python
# -*- coding:utf-8 -*-
# author "chen"


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

    # read the dataset and label
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
    
    center_id = model.choosen_original_id
    
    Dataprocessing.id_diagram(center_id, model.DS_index, model.cc_set, 1./model.knn_radius, model.delta)
    Dataprocessing.plot_quickdsc(data, center_id, model.DS_index, model.cc_set)
    Dataprocessing.show_cluster(data, label_pred, center_id)
