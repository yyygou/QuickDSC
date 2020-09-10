# !/usr/bin/python
# -*- coding:utf-8 -*-

# author "chen"

import os
import math
import csv
import numpy as np
import configparser as ConfigParser
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def read_config():
    config = ConfigParser.RawConfigParser()
    config.read('config.cfg', encoding = 'utf-8')
    config_dict = {}
    config_dict['DATA_File'] = config.get('section', 'DATA_File')
    config_dict['dismision'] = config.getint('section', 'dismision')
    config_dict['k'] = config.getint('section', 'k')
    config_dict['beta'] = config.getfloat('section', 'beta')
    config_dict['ann'] = config.get('section', 'ann')
    config_dict['n_clusters'] = config.getint('section', 'n_clusters')
    return config_dict


# 读取数据集，传入文件名和维度，将数据集读取出来并返回
# 数据集每一行的格式是：X, Y, Z, ..., label
def get_data(filename, dismision):
    data = []
    label = []

    with open(filename, 'r') as file_obj:
        csv_reader = csv.reader(file_obj)
        for row in csv_reader:
            point = []
            for dim in range(dismision):
                point.append(float(row[dim]))
            data.append(point)
            # if row[-1] == 0, int(row[-1]) will fail
            label.append(int(float(row[-1])))

    return np.array(data, np.float32), np.array(label, np.int8)


# 计算欧几里得距离,a,b分别为两个元组
def dist(a, b):
    d = len(a)
    buffer = 0.0
    for i in range(d):
        tmp = a[i] - b[i]
        buffer += tmp * tmp
    return buffer


def show_cluster(data, label_pred, center):
    plt.figure()
    plt.clf()

    X = []
    Y = []
    for point in data:
        X.append(point[0])
        Y.append(point[1])
    plt.scatter(x=X, y=Y, c=label_pred, s=10)
    
    top_center_X = []
    top_center_Y = []
    for point in center:
        top_center_X.append(point[0])
        top_center_Y.append(point[1])
    plt.scatter(x=top_center_X, y=top_center_Y, c='red', marker='*', s=20)

    plt.show()


def maxminnorm(array):
    '''
    normalization function: aim to normalize the rho and delta
    '''
    x_max, x_min = np.max(array, 0), np.min(array, 0)
    array = (array - x_min)/(x_max - x_min)
    return array


def show_diagram(radius, weight, choosen_id):
    plt.figure()
    center= [i for i in range(len(radius))]
    top_K = choosen_id
    no_top_K = set(center).difference(set(top_K))

    rho = 1./radius
    rho = maxminnorm(rho)
    # weight = maxminnorm(weight)

    top_K_X = []
    top_K_Y = []
    for i in top_K:
        top_K_X.append(rho[i])
        top_K_Y.append(weight[i])
    plt.scatter(x=top_K_X, y=top_K_Y, c='red', marker='*', s=30)

    no_top_K_X = []
    no_top_K_Y = []
    for i in no_top_K:
        no_top_K_X.append(rho[i])
        no_top_K_Y.append(weight[i])
    plt.scatter(x=no_top_K_X, y=no_top_K_Y, c='blue', marker='X', s=20)
    plt.show()


def show_center(choosen_id, DS_index, data):
    top_center = []
    for i in choosen_id:
        top_center.append(DS_index[i])
    data_index = [i for i in range(len(data))]
    no_center = set(data_index).difference(set(top_center))

    X = []
    Y = []
    for n in no_center:
        X.append(data[n][0])
        Y.append(data[n][1])
    plt.scatter(x=X, y=Y, c='black', s=3)
   
    top_center_X = []
    top_center_Y = []
    for i in top_center:
        top_center_X.append(data[i][0])
        top_center_Y.append(data[i][1])
    plt.scatter(x=top_center_X, y=top_center_Y, c='red', marker='*', s=20)

    plt.show()

def show_data(data):
    X = []
    Y = []
    for point in data:
        X.append(point[0])
        Y.append(point[1])
    plt.scatter(x=X, y=Y, c='black', s=3)
    plt.show()

def show_cluster_data(DS_data, label_pred, center, data):
    plt.figure()
    plt.clf()

    background_X = []
    background_Y = []
    for point in data:
        background_X.append(point[0])
        background_Y.append(point[1])
    plt.scatter(x=background_X, y=background_Y, c='red', s=5)
    
    X = []
    Y = []
    for point in DS_data:
        X.append(point[0])
        Y.append(point[1])
    plt.scatter(x=X, y=Y, c=label_pred, s=10)
    
    top_center_X = []
    top_center_Y = []
    for point in center:
        top_center_X.append(point[0])
        top_center_Y.append(point[1])
    plt.scatter(x=top_center_X, y=top_center_Y, c='red', marker='*', s=20)

   
    plt.show()