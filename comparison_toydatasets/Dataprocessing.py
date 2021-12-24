# !/usr/bin/python
# -*- coding:utf-8 -*-

# author "chen"

import os
import math
import csv
import imageio
import numpy as np
import configparser as ConfigParser
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation


def read_config():
    config = ConfigParser.RawConfigParser()
    config.read('config.cfg', encoding = 'utf-8')
    config_dict = {}
    config_dict['DATA_File'] = config.get('section', 'DATA_File')
    config_dict['dismision'] = config.getint('section', 'dismision')
    config_dict['A'] = config.getint('section', 'A')
    config_dict['beta'] = config.getfloat('section', 'beta')
    config_dict['ann'] = config.get('section', 'ann')
    config_dict['K'] = config.getint('section', 'K')
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

    return np.array(data, np.float32), np.array(label, np.float32)


def draw_cluster(data, label_pred):
    plt.figure()
    coo_X = []
    coo_Y = []
    for point in data:
        coo_X.append(point[0])
        coo_Y.append(point[1])
    plt.scatter(x=coo_X, y=coo_Y, c=label_pred, s=5)
    plt.show()

# 将聚类后的数据点可视化出来，包括中心点的集合 (如果有给中心点的集合)
# 数据点的存储形式是用dict来存储，key值表示类标号，value值数据点的集合
def draw(clusters, centers=None, modes_final=None):
    # 颜色列表
    colors_ = list(colors._colors_full_map.values())

    plt.figure()
    # 聚类结果的数据点可视化
    for i in clusters:
        # print(len(clusters[i]))
        coo_X = []
        coo_Y = []
        for point in clusters[i]:
            coo_X.append(point[0])
            coo_Y.append(point[1])
        plt.scatter(x=coo_X, y=coo_Y, color=colors_[i], s=5)
        # plt.scatter(x=coo_X, y=coo_Y, c=5, s=2)
    if centers is not None:
        # 中心点的可视化
        c_X = []
        c_Y = []
        for c in centers:
            c_X.append(c[0])
            c_Y.append(c[1])
        plt.scatter(x=c_X, y=c_Y, color='black', marker='x', s=20)

    if modes_final is not None:
        for i, modes in enumerate(modes_final):
            p_X = []
            p_Y = []
            for point in modes:
                p_X.append(point[0])
                p_Y.append(point[1])
            # plt.scatter(x=p_X, y=p_Y, color=colors_[i], marker='x', s=20)
            plt.scatter(x=p_X, y=p_Y, color='red', marker='x', s=10)
    plt.show()


def show_center(center, data):
    plt.figure()

    X = []
    Y = []
    for point in data:
        X.append(point[0])
        Y.append(point[1])

    plt.scatter(x=X, y=Y, c='red', s=5)

    coo_x = []
    coo_y = []
    for c in center:
        coo_x.append(c[0])
        coo_y.append(c[1])

    plt.scatter(x=coo_x, y=coo_y, c='black', s=50, marker='X')

    plt.show()

def show_center_id(c_center, data):
    plt.figure()

    X = []
    Y = []
    for point in data:
        X.append(point[0])
        Y.append(point[1])

    plt.scatter(x=X, y=Y, c='red', s=5)

    coo_x = []
    coo_y = []
    for c in c_center:
        coo_x.append(data[c][0])
        coo_y.append(data[c][1])

    plt.scatter(x=coo_x, y=coo_y, c='black', s=50, marker='X')

    plt.show()
# # 计算欧几里得距离,a,b分别为两个元组
# def dist(a, b):
#     return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

# 计算欧几里得距离,a,b分别为两个元组
def dist(a, b):
    d = len(a)
    buffer = 0.0
    for i in range(d):
        tmp = a[i] - b[i]
        buffer += tmp * tmp
    return buffer


# def find_nearest_cluster(point, cluster, new_data):
#     index = 0
#     min_dist = dist(point, new_data[cluster[0]])
#     for i, idx in enumerate(cluster):
#         distance = dist(point, new_data[idx])
#         if distance < min_dist:
#             min_dist = distance
#             index = i
#         else:
#             continue
#     return index

def find_nearest_cluster(point, cluster, new_data, distances):
    index = 0
    min_dist = dist(point, new_data[cluster[0]])
    for i, idx in enumerate(cluster):
        distance = dist(point, new_data[idx])
        if distance < min_dist:
            min_dist = distance
            index = i
        else:
            continue
    return index


def maxminnorm(array):
    '''
    normalization function: aim to normalize the rho and delta
    '''
    x_max, x_min = np.max(array, 0), np.min(array, 0)
    array = (array - x_min)/(x_max - x_min)
    return array

def show_diagram(radius, weight, choosen_id):
    plt.figure()
    
    print(choosen_id)
    center= [i for i in range(len(radius))]
    top_K = choosen_id
    no_top_K = set(center).difference(set(top_K))

    rho = 1./radius
    rho = maxminnorm(rho)
    weight = maxminnorm(weight)

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


def show_choosen_center(choosen_id, DS_index, data):
    top_center = []
    for i in choosen_id:
        top_center.append(DS_index[i])
    no_top_center = set(DS_index).difference(set(top_center))

    X = []
    Y = []
    for point in data:
        X.append(point[0])
        Y.append(point[1])
    
    plt.scatter(x=X, y=Y, c='black', s=10)
   
    no_top_center_X = []
    no_top_center_Y = []
    for i in no_top_center:
        no_top_center_X.append(data[i][0])
        no_top_center_Y.append(data[i][1])
    plt.scatter(x=no_top_center_X, y=no_top_center_Y, c='blue', marker='X', s=50)

    top_center_X = []
    top_center_Y = []
    for i in top_center:
        top_center_X.append(data[i][0])
        top_center_Y.append(data[i][1])
    plt.scatter(x=top_center_X, y=top_center_Y, c='red', marker='*', s=50)

    plt.show()