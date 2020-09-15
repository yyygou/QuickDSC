QuickDSC
======

Clustering by Quick Density Subgraph Estimation.


Usage
======

**Initializiation**:

.. code-block:: python

  QuickDSC(k, n_clusters, beta) 

k: number of neighbors in k-NN

n_clusters: number of clustering result

beta: fluctuation parameter which ranges between 0 and 1.

**Finding Clusters**:

.. code-block:: python

  fit(X)

X is the data matrix, where each row is a datapoint in euclidean space.

fit performs the clustering. The final result can be found in QuickshiftPP.memberships.

**Example**:

.. code-block:: python

  from QuickDSC import QuickDSC
  import Dataprocessing
  import numpy as np

  data, label_true = Dataprocessing.get_data(DATA_File)

  # Compute the clustering.
  model = QuickDSC(k, n_clusters, beta, ann=ann, metric=metric)
  model.fit(X)
  label_pred = model.labels_

  from sklearn import metrics

  ARI = metrics.adjusted_rand_score(label_true, label_pred)
  AMI = metrics.adjusted_mutual_info_score(label_true, label_pred)
  NMI = metrics.normalized_mutual_info_score(label_true, label_pred)

  print("Adj. Rand Index Score=" , ARI)
  print("Adj. Mutual Info Score=", AMI)
  print("Norm Mutual Info Score=", NMI)

Clustering results
=======
This is the clustering result of S3(Gaussian-15)

<img src="https://github.com/Michael-Chen-24/QuickDSC/blob/master/results/S3(id_quickdsc).png" style="zoom:50%;" />
<img src="https://github.com/Michael-Chen-24/QuickDSC/blob/master/results/S3(plot_quickdsc).png" style="zoom:50%;" />
<img src="https://github.com/Michael-Chen-24/QuickDSC/blob/master/results/S3(clusters_quickdsc).png" style="zoom:50%;" />

Install
=======

This package uses distutils, which is the default way of installing
python modules.

To install for all users on Unix/Linux::

  sudo python setup.py build; python setup.py install

To install for all users on Windows::

  python setup.py build; python setup.py install



Dependencies
=======

python 3.6, scikit-learn, Cython



# 