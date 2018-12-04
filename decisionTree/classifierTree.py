#!/usr/bin/python
# coding:utf-8
"""
分类树
使用sklearn对鸢尾花数据进行分类

by:guoKaiSama
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

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




