# !/usr/bin/python
# -*- coding:utf-8 -*-

# author "chen"

import os
import math
import csv
import numpy as np
import configparser as ConfigParser
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def read_config():
    config = ConfigParser.RawConfigParser()
    config.read('config.cfg', encoding = 'utf-8')
    config_dict = {}
    config_dict['DATA_File'] = config.get('section', 'DATA_File')
    config_dict['k'] = config.getint('section', 'k')
    config_dict['beta'] = config.getfloat('section', 'beta')
    config_dict['ann'] = config.get('section', 'ann')
    config_dict['metric'] = config.get('section', 'metric')
    config_dict['n_clusters'] = config.getint('section', 'n_clusters')
    return config_dict
    
# 读取数据集，传入文件名和维度，将数据集读取出来并返回
# 数据集每一行的格式是：X, Y, Z, ..., label
def get_data(filename):
    data = []
    label = []

    with open(filename, 'r') as file_obj:
        csv_reader = csv.reader(file_obj)
        for row in csv_reader:
            point = []
            for d in row[:-1]:
                point.append(float(d))
            data.append(point)
            # if row[-1] == 0, int(row[-1]) will fail
            label.append(int(float(row[-1])))

    X = np.array(data)
    min_max_scaler = preprocessing.MinMaxScaler() 
    X_minMax = min_max_scaler.fit_transform(X)
    return X_minMax, np.array(label, np.int8)

    # return np.array(data, np.float32), np.array(label, np.int8)


# 计算欧几里得距离,a,b分别为两个元组
def dist(a, b):
    d = len(a)
    buffer = 0.0
    for i in range(d):
        tmp = a[i] - b[i]
        buffer += tmp * tmp
    return buffer


def maxminnorm(array):
    '''
    normalization function: aim to normalize the rho and delta
    '''
    x_max, x_min = np.max(array, 0), np.min(array, 0)
    array = (array - x_min)/(x_max - x_min)
    return array


def id_diagram(c_original_center, cc_mode_set, cc_set, rho_original, delta):
    '''
    c_original_center : pick center，选中的K个中心点
    cc_mode_set : the first level cc mode set, （data -- find_cc -- cc set）
                    在第二层中，所有的CC的密度最高点（下一层的输入点）
    cc_set: cc point set,在第二层中，cc中所有的点的集合

    '''
    rho_original = maxminnorm(rho_original)
    delta_original = [0. for i in range(len(rho_original))]
    
    for i, d in zip(cc_mode_set, delta):
        delta_original[i] = d
    
    plt.figure(figsize=[6.40,5.60])
    point_set  = [i for i in range(len(rho_original))]
    point_set = list(set(point_set).difference(cc_set))
    cc_set = list(set(cc_set).difference(set(cc_mode_set)))
    cc_mode_set = list(set(cc_mode_set).difference(set(c_original_center)))


    mode_X = []
    mode_Y = []
    for m in cc_mode_set:
        mode_X.append(rho_original[m])
        mode_Y.append(delta_original[m])
    plt.scatter(mode_X, mode_Y, marker='.', s=200, c='blue', label='centers of other DSs')


    center_X = []
    center_Y = []
    for c in c_original_center:
        center_X.append(rho_original[c])
        center_Y.append(delta_original[c])
    plt.scatter(center_X, center_Y, marker='*', s=200, label='centers of top-K DSs', c='red')
    
    plt.title('Importance Diagram',fontsize=30)
    plt.xlabel('Density:1/$r_k(x_i)$',fontsize=15)
    plt.ylabel('Geometric Weight:$w_i$',fontsize=15)
    ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc='lower center', bbox_to_anchor=(0.5,1), fontsize=12, frameon=False)
    plt.show()


def plot_quickdsc(data, c_original_center, cc_mode_set, cc_set):
    plt.figure(figsize=[6.40,5.60])

    point_set  = [i for i in range(len(data))]
    point_set = list(set(point_set).difference(cc_set))
    cc_set = list(set(cc_set).difference(set(cc_mode_set)))
    cc_mode_set = list(set(cc_mode_set).difference(set(c_original_center)))

    point_X = []
    point_Y = []
    for p in point_set:
        point_X.append(data[p][0])
        point_Y.append(data[p][1])
    plt.scatter(point_X, point_Y, marker='x', c='grey', s=50, label='non-DS samples')


    cc_X = []
    cc_Y = []
    for c in cc_set:
        cc_X.append(data[c][0])
        cc_Y.append(data[c][1])
    plt.scatter(cc_X,cc_Y, marker='.', c='black', s=100, label='non-center samples in DSs')


    mode_X = []
    mode_Y = []
    for m in cc_mode_set:
        mode_X.append(data[m][0])
        mode_Y.append(data[m][1])
    plt.scatter(mode_X, mode_Y, marker='.', s=160, c='blue', label='centers of other DSs')


    center_X = []
    center_Y = []
    for c in c_original_center:
        center_X.append(data[c][0])
        center_Y.append(data[c][1])
    plt.scatter(center_X, center_Y, marker='*', s=200, label='centers of top-K DSs', c='red')
    
    
    ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # reverse the order
    ax=plt.gca()
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc='lower center', bbox_to_anchor=(0.5,1), fontsize=12, frameon=False)
    plt.show()


def show_cluster(data, label_pred, center_id):
    plt.figure(figsize=[6.40,5.60])

    X = []
    Y = []
    for point in data:
        X.append(point[0])
        Y.append(point[1])
    plt.scatter(x=X, y=Y, c=label_pred, s=8)

    center_X = []
    center_Y = []
    for i in center_id:
        center_X.append(data[i][0])
        center_Y.append(data[i][1])
    plt.scatter(x=center_X, y=center_Y, marker='*',c='red',s=150)
    plt.show()

