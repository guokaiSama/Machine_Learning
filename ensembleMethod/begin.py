#!/usr/bin/python
# coding:utf-8
"""
集成学习相关
sklearn的集成学习接口使用

by:guoKaiSama
"""

import stacking.stacking as stacking
import AdaBoost.adaBoostSklearn as sklearnAdaboost
if __name__=="__main__":
    #利用gini系数生成决策树
    #stacking.stackingMethod()

    # 使用sklearn的adaBoost算法
    sklearnAdaboost.adaBoostSklearn()
