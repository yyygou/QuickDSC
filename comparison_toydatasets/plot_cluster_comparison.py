"""
=========================================================
Comparing different clustering algorithms on toy datasets
=========================================================

This example shows characteristics of different
clustering algorithms on datasets that are "interesting"
but still in 2D. With the exception of the last dataset,
the parameters of each of these dataset-algorithm pairs
has been tuned to produce good clustering results. Some
algorithms are more sensitive to parameter values than
others.

The last dataset is an example of a 'null' situation for
clustering: the data is homogeneous, and there is no good
clustering. For this example, the null dataset uses the
same parameters as the dataset in the row above it, which
represents a mismatch in the parameter values and the
data structure.

While these examples give some intuition about the
algorithms, this intuition might not apply to very high
dimensional data.
"""
# print(__doc__)

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import Dataprocessing
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from quick_shift_ import QuickShift
from QuickshiftPP import *
from DPC_class import DPC
from QuickDSC import QuickDSC


np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============

dismision = 2
n_samples = 1500

noisy_circles_path = './toydatasets/two_circles.csv'
# noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                    #   noise=.05)
X_noisy_circles, label_circles = Dataprocessing.get_data(noisy_circles_path, dismision)
noisy_circles = (X_noisy_circles, label_circles)

noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)

random_state = 170
# blobs with varied variances
# varied_blobs = datasets.make_blobs(n_samples=n_samples,
                            #  cluster_std=[1.0, 2.5, 0.5],
                            #  random_state=random_state)
varied_blobs_path = './toydatasets/varied_blobs.csv'
X_varied_blobs, label_varied_blobs = Dataprocessing.get_data(varied_blobs_path, dismision)
varied_blobs = (X_varied_blobs, label_varied_blobs)

# Anisotropicly distributed data
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

no_structure = np.random.rand(n_samples, 2), None

# ============
# Set up cluster parameters
# ============

plt.figure()

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}

datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2,
                     'min_samples': 20, 'xi': 0.25}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied_blobs, {'eps': .18, 'n_neighbors': 2,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
    (aniso, {'eps': .15, 'n_neighbors': 2,
             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    (blobs, {}),
    (no_structure, {'n_clusters':1})]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============

    k_means = cluster.KMeans(n_clusters=params['n_clusters'])
    dbscan = cluster.DBSCAN(eps=params['eps'])
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    quickshift = QuickShift(bandwidth=bandwidth/3)
    quickshift_pp = QuickshiftPP(k=20, beta=0.7)
    dpc = DPC(n_clusters=params['n_clusters'])

    if i_dataset == 0:
        quick_dpc = QuickDSC(k=10, n_clusters=params['n_clusters'], beta=0.8)
    elif i_dataset == 1:
        quick_dpc = QuickDSC(k=8, n_clusters=params['n_clusters'], beta=0.9)
    elif i_dataset == 2:
        quick_dpc = QuickDSC(k=6, n_clusters=params['n_clusters'], beta=0.7)
    else:
        quick_dpc = QuickDSC(k=10, n_clusters=params['n_clusters'], beta=0.7)


    clustering_algorithms = (
        ('KMeans', k_means),
        ('DBSCAN', dbscan),
        ('MeanShift', ms),
        ('Agglo-Clustering', average_linkage),
        ('BIRCH', birch),
        ('QuickShift', quickshift),
        ('QuickShiftPP', quickshift_pp),
        ('DPC', dpc),
        ('QuickDSC', quick_dpc),
        ('Importance Diagram(Ours)', quick_dpc)
    )

    for name, algorithm in clustering_algorithms:

        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.memberships

        plt.subplot(len(clustering_algorithms), len(datasets), plot_num)

        if i_dataset == 0:
            plt.ylabel(name, size=9)
            if name == 'Importance Diagram(Ours)':
                plt.ylabel('Importance Diagram\n(Ours)', size=9)

        if name == 'Importance Diagram(Ours)':
            rho = algorithm.rho_radius
            rho = 1./rho
            delta = algorithm.delta
            
            rho = Dataprocessing.maxminnorm(rho)
            # delta = Dataprocessing.maxminnorm(delta)

            plt.scatter(rho, delta, s=10, color='black')
            plot_num = plot_num - (len(clustering_algorithms) - 1) * len(datasets) + 1
            continue

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plot_num += len(datasets)

plt.show()
