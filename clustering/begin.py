#!/usr/bin/python
# coding:utf-8
"""
聚类算法学习相关
sklearn的聚类算法接口使用

by:guoKaiSama
"""

import kMeans.kMeans as kMeans

if __name__=="__main__":
    # 传统Kmeans
    #kMeans.kMeansMethod()
    # 二分 biKMeans
    kMeans.biKMeansMethod()

