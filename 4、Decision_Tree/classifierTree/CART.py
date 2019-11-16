#!/usr/bin/python
# coding:utf-8
import itertools
import utils.utils as utils
import numpy as np
import pandas as pd
from math import log
import operator
import re
from collections import defaultdict
import itertools

def calGini(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    gini = 1.0
    for label in labelCounts.keys():
        prop = float(labelCounts[label]) / numEntries
        gini -= prop * prop
    return gini


# 传入的是一个特征值的列表，返回特征值二分的结果
def featuresplit(features):
    count = len(features)  # 特征值的个数
    if count < 2:  # 特征值只有一个值比如'cold_blood'
        li = []
        print "please check sample's features,only one feature value"
        li.append(features)
        return tuple(li)  # 列表转化为元组

    # 由于需要返回二分结果，所以每个分支至少需要一个特征值，所以要从所有的特征组合中选取1个以上的组合
    # itertools的combinations 函数可以返回一个列表选多少个元素的组合结果，例如combinations(list,2)返回的列表元素选2个的组合
    # 我们需要选择1-（count-1）的组合
    featureIndex = range(count)
    featureIndex.pop(0)
    combinationsList = []
    resList = []
    combiLen = 0
    # 遍历所有的组合
    for i in featureIndex:
        temp_combination = list(itertools.combinations(features, len(features[0:i])))
        combinationsList.extend(temp_combination)
        combiLen = len(combinationsList)
    # 每次组合的顺序都是一致的，并且也是对称的，所以我们取首尾组合集合
    # zip函数提供了两个列表对应位置组合的功能
    resList = zip(combinationsList[0:combiLen / 2], combinationsList[combiLen - 1:combiLen / 2 - 1:-1])  # 往回数间隔为1

    return resList  # 二分特征的不同情况

def splitDataSet(dataSet, axis, values):
    retDataSet = []
    # 长度小于2即只有一个特征值
    if len(values) < 2:
        for featVec in dataSet:
            # 如果特征值只有一个，不抽取当选特征
            if featVec[axis] == values[0]:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    else:
        for featVec in dataSet:
            for value in values:
                # 如果特征值多于一个，选取当前特征
                if featVec[axis] == value:
                    retDataSet.append(featVec)
    return retDataSet


# 返回最好的特征以及二分特征值
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    bestGiniGain = 1.0
    bestFeature = -1
    bestBinarySplit=()
    # 遍历特征
    for i in range(numFeatures):
        # 得到特征列
        featList = [example[i] for example in dataSet]
        # 去除重复值的特征列
        uniqueVals = list(set(featList))
        # 返回特征的所有二分情况
        for split in featuresplit(uniqueVals):
            GiniGain = 0.0
            # split是一个元组 特征值只有一个比如:cold_blood 只有一个特征值就没办法继续划分下去了 所以跳出循环继续下一循环
            if len(split)==1:
                continue
            (left,right)=split

            # 对于每一个可能的二分结果计算gini增益
            # 左增益
            left_subDataSet = splitDataSet(dataSet, i, left)
            left_prob = len(left_subDataSet)/float(len(dataSet))
            GiniGain += left_prob * calGini(left_subDataSet)
            # 右增益
            right_subDataSet = splitDataSet(dataSet, i, right)
            right_prob = len(right_subDataSet)/float(len(dataSet))
            GiniGain += right_prob * calGini(right_subDataSet)
            # 比较是否是最好的结果
            if (GiniGain <= bestGiniGain):
                # 记录最好的结果和最好的特征
                bestGiniGain = GiniGain
                bestFeature = i
                bestBinarySplit=(left,right)
    return bestFeature,bestBinarySplit


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回标签
    return sortedClassCount[0][0]


def giniTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # 所有的类别都一样，就不用再划分了
        return classList[0]

    # 如果没有继续可以划分的特征，就多数表决决定分支的类别
    if len(dataSet) == 1:
        return majorityCnt(classList)
    bestFeat,bestBinarySplit = chooseBestFeatureToSplit(dataSet)

    bestFeatLabel = labels[bestFeat]
    if bestFeat==-1:
        return majorityCnt(classList)
    myTree = {bestFeatLabel:{}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = list(set(featValues))
    for value in bestBinarySplit:
        # 拷贝防止其他地方修改  特征标签
        subLabels = labels[:]
        if len(value)<2:
            del(subLabels[bestFeat])
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree
def createGiniTree():
    # 1.创建数据和结果标签
    # 有两个属性，每个属性只有两种取值
    myDat, featureName = utils.createXiGuaDataSet()

    # 训练决策树
    myTree = giniTree(myDat, copy.deepcopy(featureName))

    print(myTree)