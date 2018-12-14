#!/usr/bin/python
# coding:utf-8
import utils.utils as utils
import copy
import numpy as np
import math

# 计算香农熵
def calcShannonEnt(dataSet):
    # 求list的长度，表示训练数据的个数
    # 最后一列是lable
    numEntries = len(dataSet)

    # 计算分类标签label出现的次数
    labelCounts = {}

    for featVec in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。
        # 每个键值都记录了当前类别出现的次数。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 对于label标签的占比，求出label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率。
        prob = float(labelCounts[key])/numEntries

        # 计算香农熵，以 2 为底求对数
        shannonEnt -= prob * math.log(prob, 2)

    return shannonEnt


# 切分数据集
def splitDataSet(dataSet, index, value):
    """
    根据index列（属性）进行分类，如果index列的值等于value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
        dataSet 数据集                 待划分的数据集
        index 表示每一行的index列        划分数据集的特征
        value 表示index列对应的value值   需要返回的特征的值。

    """
    retDataSet = []
    for featVec in dataSet:
        # 判断index列的值是否为value
        if featVec[index] == value:
            # 去除掉该值，构成新的数据集
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


# 选择出现次数最多的lable
def majorityCnt(classList):
    """
    Desc:
        选择出现次数最多的一个结果
    Args:
        classList label列的集合
    Returns:
        bestFeature 最优的特征列
    """

    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 选择最优属性列
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
    baseEntropy = calcShannonEnt(dataSet)

    # 最优的信息增益值, 和最优的Featurn编号
    bestInfoGain, bestFeature = 0.0, -1

    for i in range(numFeatures):
        # 获取对应的属性下的所有取值，不包括lable
        featList = [example[i] for example in dataSet]

        # 去重
        uniqueVals = set(featList)

        # 创建一个临时的信息熵
        newEntropy = 0.0

        # 遍历某一列的value集合，计算该列的信息熵
        for value in uniqueVals:
            # 按照value和属性列取值
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算概率
            prob = len(subDataSet)/float(len(dataSet))
            # 计算信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)

        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        print ('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
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
        myTree[bestFeatName][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabelName)

    return myTree


# 对测试数据分类
def classify(inputTree, featureName, testVec):
    """
        inputTree  -- 已经训练好的决策树模型
        featureName -- Feature标签对应的名称
        testVec    -- 测试输入的数据
    """
    # 找到根节点的属性在我们的样本集中处于哪个位置

    # 获取tree的根节点对于的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 找到根节点的属性在我们的样本集中处于哪个位置
    featIndex = featureName.index(firstStr)

    # 测试数据，找到这个值（key）对应的划分（valueOfFeat）
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]

    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)

    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featureName, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


# 西瓜书的数据集P76
def xiguaTest():
    # 1.创建数据和结果标签
    # 有两个属性，每个属性只有两种取值
    myDat, featureName = utils.createXiGuaDataSet()

    # 训练决策树
    myTree = createTree(myDat, copy.deepcopy(featureName))

    print(myTree)

    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, featureName, [1, 1,1,1,1,1]))

    # 画图可视化展现
    utils.createPlot(myTree)


def glassTest():
    # 1.创建数据和结果标签
    myDat, labels = utils.loadGlassDataSet()

    myTree = createTree(myDat, copy.deepcopy(labels))

    print(myTree)

    print(classify(myTree, labels, ["young", "myope", "yes", "reduced"]))

    # 画图可视化展现
    utils.createPlot(myTree)

def mushroomTest():
    # 1.创建数据和结果标签
    myDat, labels = utils.loadMushroomDataSet()

    myTree = createTree(myDat, copy.deepcopy(labels))

    print(myTree)
    print(classify(myTree, labels, ["x", "s", "n", "t", "p", "f", "c", "n", "k", "e", "e", "s",
                                    "s", "w", "w", "p", "w", "o", "p", "k", "s", "u"]))

    # 画图可视化展现
    utils.createPlot(myTree)
