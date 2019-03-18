#!/usr/bin/python
# coding:utf-8
import numpy as np
from sklearn import datasets
from sklearn.cluster import SpectralClustering
from sklearn import metrics
def genData():
    X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4],
                               random_state=11)
    return X, y


def spectralMethod():
    X, y=genData()
    y_pred = SpectralClustering().fit_predict(X)
    print "Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred)