#!/usr/bin/python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
def genData():
    X1, y1 = datasets.make_circles(n_samples=5000, factor=.6,noise=.05)
    X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]],random_state=9)

    X = np.concatenate((X1, X2))
    #plt.scatter(X[:, 0], X[:, 1], marker='o')
    #plt.show()
    return X
def dbsc():
    X=genData()
    y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()