#!/usr/bin/python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import metrics
def genData():
    # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
    X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=9)
    # plt.scatter(X[:, 0], X[:, 1], marker='o')
    # plt.show()
    return X,y
def kMeans():
    X, y=genData()
    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    #plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    #plt.show()
    return X,y_pred

# 用Calinski-Harabasz Index评估的聚类分数
def calinskiHarabasz():
    X, y_pred=kMeans()
    errors=metrics.calinski_harabaz_score(X, y_pred)
    print errors

def miniBatchKMeans():
    X, y=genData()
    y_pred = MiniBatchKMeans(n_clusters=4, batch_size=200, random_state=9).fit_predict(X)
    score = metrics.calinski_harabaz_score(X, y_pred)
    print score
"""

for index, k in enumerate((2,3,4,5)):
    plt.subplot(2,2,index+1)
    y_pred = MiniBatchKMeans(n_clusters=k, batch_size = 200, random_state=9).fit_predict(X)
    score= metrics.calinski_harabaz_score(X, y_pred)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.text(.99, .01, ('k=%d, score: %.2f' % (k,score)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
plt.show()

"""