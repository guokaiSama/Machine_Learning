#!/usr/bin/python
# coding:utf-8
"""
决策树的算法实现
sklearn的决策树接口使用

by:guoKaiSama
"""
import classifierTree
import regressorTree
if __name__=="__main__":
    # 分类树，sklearn
    #classifierTree.classifierTree()

    # 回归树，sklearn
    #regressorTree.regressorTree()

    #手动实现决策树,鱼数据
    #classifierTree.fishTest()
    # 手动实现决策树,眼镜数据
    #classifierTree.glassTest()
    # 手动实现决策树,蘑菇数据集
    classifierTree.mushroomTest()