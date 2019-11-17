#!/usr/bin/python
# coding:utf-8
import utils.utils as utils
import copy
import ID3
import math
# 选择最优属性列
def calcShannonEntRate(featList):
    # 求list的长度，表示训练数据的个数
    # 最后一列是lable
    numEntries = len(featList)

    # 计算该属性下，这个值出现的次数
    valueCounts = {}

    for featVal in featList:
        if featVal not in valueCounts.keys():
            valueCounts[featVal] = 0
        valueCounts[featVal] += 1

    # 计算IV
    shannonEntIV = 0.0
    for key in valueCounts:
        prob = float(valueCounts[key]) / numEntries
        shannonEntIV -= prob * math.log(prob, 2)
    return shannonEntIV

def chooseBestFeatureToSplit(dataSet):
    """

    Args:
        dataSet 数据集，最后一列是lable
    Returns:
        bestFeature 最优的特征列
    """
    # 计算列数，因为最后一列是lable，所以-1
    numFeatures = len(dataSet[0]) - 1

    # 数据集的原始信息熵
    baseEntropy = ID3.calcShannonEnt(dataSet)

    # 最优的信息增益值, 和最优的Featurn编号
    bestInfoGainRate, bestFeature = 0.0, -1

    for i in range(numFeatures):
        # 获取对应的属性下的所有取值，不包括lable
        featList = [example[i] for example in dataSet]

        # 去重
        uniqueVals = set(featList)

        # 创建一个临时的信息熵
        newEntropy = 0.0
        # 创建一个临时的信息增益率的分母
        newIV=calcShannonEntRate(featList)

        # 遍历某一列的value集合，计算该列的信息熵
        for value in uniqueVals:
            # 按照value和属性列取值
            subDataSet = ID3.splitDataSet(dataSet, i, value)
            # 计算概率
            prob = len(subDataSet)/float(len(dataSet))
            # 计算信息熵
            newEntropy += prob * ID3.calcShannonEnt(subDataSet)

        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        infoGainRate=infoGain/newIV
        print ('infoGainRate=', infoGainRate, 'bestFeature=', i, baseEntropy, newEntropy)
        if (infoGainRate > bestInfoGainRate):
            bestInfoGainRate = infoGainRate
            bestFeature = i
    return bestFeature


# 生成决策树
def createTree(dataSet, featureName):
    """
    Args:
        dataSet -- 要创建决策树的训练数据集
        featureName -- 训练数据集中属性对应的含义
    """
    # 获取每个样本的lable取值（比如no，yes），lables中保存的是
    classList = [example[-1] for example in dataSet]

    # 第一个停止条件：所有lable的取值完全相同，则直接返回该类标签。
    if len(set(classList)) == 1:
        return classList[0]

    # 第二个停止条件：使用完了所有属性，仍然不能将数据集划分成仅包含唯一类别的分组。
    # 如果数据集只有1列(说明只剩下lable列了)，那么最初出现label次数最多的一类，作为结果
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)

    # 获取label的名称
    bestFeatName = featureName[bestFeat]

    # 初始化myTree
    myTree = {bestFeatName: {}}

    # 注：featureName列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 如果浅拷贝的话，会导致函数外的同名变量被删除了元素
    # 因为属性列被删除，所以对应的属性列的含义也应该被删除
    del(featureName[bestFeat])

    # 取出最优列对应的值
    featValues = [example[bestFeat] for example in dataSet]

    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabelName = featureName[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatName][value] = createTree(ID3.splitDataSet(dataSet, bestFeat, value), subLabelName)

    return myTree


# 西瓜书的数据集P76
def xiguaTest():
    # 1.创建数据和结果标签
    # 有两个属性，每个属性只有两种取值
    myDat, featureName = utils.createXiGuaDataSet()

    # 训练决策树
    myTree = createTree(myDat, copy.deepcopy(featureName))

    print(myTree)

    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(ID3.classify(myTree, featureName,[1, 1,1,1,1,1]))

    # 画图可视化展现
    utils.createPlot(myTree)