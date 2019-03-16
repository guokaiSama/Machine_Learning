#!/usr/bin/python
# coding:utf-8
"""
聚类算法学习相关
sklearn的聚类算法接口使用

by:guoKaiSama
"""

import kMeans.kMeans as kMeans
import kMeans.kMeansSklearn as sklearnKMeans
import DBSCAN.DBSCANsklearn as dbsc
if __name__=="__main__":
    # 传统Kmeans
    #kMeans.kMeansMethod()
    # 二分 biKMeans
    #kMeans.biKMeansMethod()

    # sklearn提供的kmeans方法
    #sklearnKMeans.kMeans()
    # 对聚类结果进行评估
    #sklearnKMeans.calinskiHarabasz()
    # sklearn提供的miniBatchKMeans方法
    #sklearnKMeans.miniBatchKMeans()

    # sklearn提供的DBSCAN方法
    dbsc.dbsc()
