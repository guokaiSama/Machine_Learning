#!/usr/bin/python
# coding:utf-8
"""
分类树
使用sklearn对鸢尾花数据进行分类

by:guoKaiSama
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import copy
from math import log

# sawtooth 波浪方框
decisionNode = dict(boxstyle="sawtooth", fc="0.8")

# round4 矩形方框 , fc表示字体颜色的深浅 0.1~0.9 依次变浅
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# 生成数据集
def createFisheDataSet():
    """
    创建鱼类和非鱼类数据集
    有两个特征：
        1.不浮出水面是否可以生存
        2.是否有脚蹼
    lable:
        yes是鱼类
        no是非鱼类
    """
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 加载眼镜数据集
def loadGlassDataSet():
    fr=open("./data/lenses.txt","rb")
    lecses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lecses,lensesLabels

# 计算香农熵
def calcShannonEnt(dataSet):
    """
    计算dataSet的香农熵
    """

    # 求list的长度，表示训练数据的个数
    numEntries = len(dataSet)

    # 计算分类标签label出现的次数
    labelCounts = {}

    for featVec in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 对于label标签的占比，求出label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率。
        prob = float(labelCounts[key])/numEntries

        # 计算香农熵，以 2 为底求对数
        shannonEnt -= prob * log(prob, 2)

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

# 选择最优属性
def chooseBestFeatureToSplit(dataSet):
    """

    Args:
        dataSet 数据集
    Returns:
        bestFeature 最优的特征列
    """
    # 计算列数
    numFeatures = len(dataSet[0]) - 1

    # 数据集的原始信息熵
    baseEntropy = calcShannonEnt(dataSet)

    # 最优的信息增益值, 和最优的Featurn编号
    bestInfoGain, bestFeature = 0.0, -1

    for i in range(numFeatures):
        # 获取对应的属性下的所有取值
        featList = [example[i] for example in dataSet]

        # 去重
        uniqueVals = set(featList)

        # 创建一个临时的信息熵
        newEntropy = 0.0

        # 遍历某一列的value集合，计算该列的信息熵
        for value in uniqueVals:
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

# 生成决策树
def createTree(dataSet, labels):
    """
    Args:
        dataSet -- 要创建决策树的训练数据集
        labels -- 训练数据集中特征对应的含义的labels
    """
    # 获取所有的lable
    classList = [example[-1] for example in dataSet]

    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)

    # 获取label的名称
    bestFeatLabel = labels[bestFeat]

    # 初始化myTree
    myTree = {bestFeatLabel: {}}

    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[bestFeat])

    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]

    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree

# 对测试数据分类
def classify(inputTree, featLabels, testVec):
    """
        inputTree  -- 已经训练好的决策树模型
        featLabels -- Feature标签对应的名称
        testVec    -- 测试输入的数据
    """
    # 获取tree的根节点对于的key值
    firstStr = list(inputTree.keys())[0]

    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]

    # 判断根节点的名称在label中的位置
    featIndex = featLabels.index(firstStr)

    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]

    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)

    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

# 获取叶节点个数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是否为dict, 不是+1
        if type(secondDict[key]) is dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# 获取树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]

    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是不是dict, 求分枝的深度
        # ----------写法1 start ---------------
        if type(secondDict[key]) is dict:
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        maxDepth = max(maxDepth, thisDepth)
    return maxDepth

# 绘制节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

# 绘制连线上的文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    # 获取叶子节点的数量
    numLeafs = getNumLeafs(myTree)

    # 获取树的深度
    # depth = getTreeDepth(myTree)

    # 找出第1个中心点的位置，然后与 parentPt定点进行划线
    cntrPt = (plotTree.xOff + (1 + numLeafs) / 2 / plotTree.totalW, plotTree.yOff)

    # 并打印输入对应的文字
    plotMidText(cntrPt, parentPt, nodeTxt)

    firstStr = list(myTree.keys())[0]

    # 可视化Node分支点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)

    # 根节点的值
    secondDict = myTree[firstStr]

    # y值 = 最高点-层数的高度[第二个节点位置]
    plotTree.yOff = plotTree.yOff - 1 / plotTree.totalD
    for key in secondDict.keys():
        # 判断该节点是否是Node节点
        if type(secondDict[key]) is dict:
            # 如果是就递归调用[recursion]
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 如果不是，就在原来节点一半的地方找到节点的坐标
            plotTree.xOff = plotTree.xOff + 1 / plotTree.totalW
            # 可视化该节点位置
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # 并打印输入对应的文字
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1 / plotTree.totalD

def createPlot(inTree):
    # 创建一个figure的模版
    fig = plt.figure(1, facecolor='green')
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    # 表示创建一个1行，1列的图，createPlot.ax1 为第 1 个子图，
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # 半个节点的长度
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

# 测试画图
"""
def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # ticks for demo puropses
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
"""

# 测试数据集
def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]

# 把模型保存到pickle
def storeTree(inputTree, filename):
    """
    保存训练好的模型
    """
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)

# 从pickle中读取模型
def grabTree(filename):
    """
    从pickle中读取模型
    """
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

# 判断是否是鱼类的主流程，并将结果使用 matplotlib 画出来
def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = createFisheDataSet()
    # print(myDat, labels)

    # 计算label分类标签的香农熵
    # calcShannonEnt(myDat)

    # # 求第0列 为 1/0的列的数据集【排除第0列】
    # print('1---', splitDataSet(myDat, 0, 1))
    # print('0---', splitDataSet(myDat, 0, 0))

    # # 计算最好的信息增益的列
    # print(chooseBestFeatureToSplit(myDat))

    myTree = createTree(myDat, copy.deepcopy(labels))

    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))

    # 画图可视化展现
    #dtPlot.createPlot(myTree)
    createPlot(myTree)

def glassTest():
    # 1.创建数据和结果标签
    myDat, labels = loadGlassDataSet()
    # print(myDat, labels)

    # 计算label分类标签的香农熵
    # calcShannonEnt(myDat)

    # # 求第0列 为 1/0的列的数据集【排除第0列】
    # print('1---', splitDataSet(myDat, 0, 1))
    # print('0---', splitDataSet(myDat, 0, 0))

    # # 计算最好的信息增益的列
    # print(chooseBestFeatureToSplit(myDat))

    myTree = createTree(myDat, copy.deepcopy(labels))

    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, ["young","myope","yes","reduced"]))

    # 画图可视化展现
    #dtPlot.createPlot(myTree)
    createPlot(myTree)

def classifierTree():
    # 绘制坐标点的步长
    plot_step = 0.02
    n_classes = 3
    plot_colors = "bmy"

    # 加载数据
    iris = load_iris()

    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
        # [:, pair]表示所有的行，pair中指定的列
        X = iris.data[:, pair]
        y = iris.target

        # 训练
        clf = DecisionTreeClassifier().fit(X, y)

        # 绘制决策边界
        plt.subplot(2, 3, pairidx + 1)

        # 求出最大值与最小值
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # 生成网格点坐标矩阵
        """
        语法：X,Y = numpy.meshgrid(x, y)
        输入的x，y，就是网格点的横纵坐标列向量（非矩阵）
        输出的X，Y，就是坐标矩阵。
        """
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))

        # xx.ravel()将多维数组展开成一维
        # np.c_是按行连接两个矩阵，就是把两矩阵对应的数据连接到一起，比如：[1,2,3],[3,2,1]->[[1,3],[2,2],[3,1]]
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 绘制等高线图
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

        plt.xlabel(iris.feature_names[pair[0]])
        plt.ylabel(iris.feature_names[pair[1]])
        plt.axis("tight")

        # 绘制训练点
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                        cmap=plt.cm.Paired)

        plt.axis("tight")

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend()
    plt.show()




