#!/usr/bin/python
# coding:utf-8
"""
决策树的算法实现
sklearn的决策树接口使用

by:guoKaiSama
"""
import classifierTree.ID3 as ID3
import classifierTree.C4_5 as C45
import classifierTree.sklearnClassfierTree as SKC
import regressorTree.sklearnRegressorTree as SKR
import regressorTree.giniTree as gini
import decisionGraph
import classifierTree.CART as giniCart
if __name__=="__main__":
    # 利用graphviz来可视化Sklearn训练得到的决策树
    #decisionGraph.sklearnGraph()

    # begin *****************决策树****************** begin #
    # 利用信息增益生成决策树(ID3)
    #ID3.xiguaTest()

    # 眼镜数据与蘑菇数据的实验
    #ID3.glassTest()
    #ID3.mushroomTest()


    # 利用信息增益率生成决策树(C4_5)
    #C45.xiguaTest()

    # CART分类树
    #giniCart.createGiniTree()


    # 分类树，sklearn
    #SKC.classifierTree()
    # end *****************决策树****************** end #


    # begin *****************回归树***************** begin #
    #SKR.regressorTree()

    #利用gini系数生成决策树
    gini.giniTest()