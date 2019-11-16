#!/usr/bin/python
# coding:utf-8
import numpy as np
import utils.utils as utils


# 1.用最佳方式切分数据集
# 2.生成相应的叶节点
def chooseBestSplit(dataSet, leafType=utils.regLeaf, errType=utils.regErr, ops=(1, 4)):
    """
    Args:
        dataSet   加载的原始数据集
        leafType  建立叶子点的函数
        errType   误差计算函数(求方差和)
        ops       [容许误差下降值，切分的最少样本数]。
    Returns:
        bestIndex feature的index坐标
        bestValue 切分的最优值
    Raises:
    """

    # ops=(1,4)保存了决策树划分停止的threshold值，属于预剪枝策略，用于控制决策树停止时机。
    # 当误差的下降值小于tolS，或划分后的集合size小于tolN时，选择停止继续划分。

    # 最小误差下降值，划分后的误差减小小于这个差值，就不继续划分
    tolS = ops[0]
    # 划分最小 size 小于，就不继续划分了
    tolN = ops[1]

    # .T 对数据集进行转置, .tolist()[0] 转化为数组并取第0列
    # 如果集合size为1，也就是说全部的数据都是同一个类别，不用继续划分。
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        #  exit cond 1
        return None, leafType(dataSet)

    # 计算行列值
    m, n = np.shape(dataSet)

    # 回归树，计算总方差和
    S = errType(dataSet)

    # inf 正无穷大
    bestS, bestIndex, bestValue = np.inf, 0, 0

    # 循环处理每一列对应的属性值
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            # 对该列进行分组，按照val值进行二元切分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 判断二元切分的方式的元素数量是否符合预期
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            # 计算划分后的误差
            newS = errType(mat0) + errType(mat1)

            # 如果划分后误差小于 bestS，则说明找到了新的bestS
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    # 判断二元切分的方式的元素误差是否符合预期
    if (S - bestS) < tolS:
        return None, leafType(dataSet)

    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 对整体的成员进行判断，是否符合预期
    # 如果集合的 size 小于 tolN
    # 当最佳划分后，集合过小，也不划分，产生叶节点
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 如果构建的是回归树，该模型是一个常数，如果是模型树，其模型是一个线性方程。
def createTree(dataSet, leafType=utils.regLeaf, errType=utils.regErr, ops=(1, 4)):
    """
    Args:
        dataSet      原始数据集
        leafType     建立叶子点的函数
        errType      误差计算函数
        ops=(1, 4)   [容许误差下降值，切分的最少样本数]
    Returns:
        retTree    决策树最后的结果
    """
    # 1、选择最优的切分方式：feature是最优属性的列，val是最优切分值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)

    # 如果 splitting 达到一个停止条件，那么返回 val
    # 最后的返回结果是值为val的叶子节点
    if feat is None:
        return val
    retTree = {}
    retTree['spIndex'] = feat
    retTree['spVal'] = val

    # 大于在右边，小于在左边，分为2个数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)

    # 递归的进行调用，在左右子树中继续递归生成树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

# 将数据集按照第feature列的value值进行二元划分
def binSplitDataSet(dataSet, featureIndex, value):
    """
    Args:
        dataMat 数据集
        feature 待切分的特征列
        value 特征列要比较的值
    Returns:
        mat0 小于等于 value 的数据集在左边
        mat1 大于 value 的数据集在右边
    Raises:
    """
    # nonzero,用于获取非零元素的索引
    mat0 = dataSet[np.nonzero(dataSet[:, featureIndex] <= value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, featureIndex] > value)[0], :]
    return mat0, mat1


# 从上而下找到叶节点，用测试数据集来判断将这些叶节点合并是否能降低测试误差
def prune(tree, testData):
    """
    Args:
        tree -- 待剪枝的树
        testData -- 剪枝所需要的测试数据 testData
    Returns:
        tree -- 剪枝完成的树
    """
    # 判断测试数据集s是否为空
    if np.shape(testData)[0] == 0:
        return utils.getMean(tree)

    # 判断分枝是否是dict字典（是否是叶节点），如果是就将测试数据集进行切分
    if (utils.isTree(tree['right']) or utils.isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spIndex'], tree['spVal'])

    # 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if utils.isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)

    # 如果是右边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if utils.isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)


    # 如果左右两边同时都不是dict字典，即都是叶节点，那么分割测试数据集。
    # 1. 如果正确
    #   * 那么计算一下总方差 和 该结果集的本身不分枝的总方差比较
    #   * 如果 合并的总方差 < 不合并的总方差，那么就进行合并
    # 注意返回的结果： 如果可以合并，原来的dict就变为了 数值
    if not utils.isTree(tree['left']) and not utils.isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spIndex'], tree['spVal'])
        # power(x, y)表示x的y次方
        # tree['left']和tree['right']都是叶节点，保存具体数值
        # 不合并的误差
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        # 如果 合并的总方差 < 不合并的总方差，那么就进行合并
        if errorMerge < errorNoMerge:
            print "merging"
            # 进行合并
            return treeMean
        else:
            return tree
    else:
        return tree


def giniTest():
    myDat = utils.loadDataSet('./data/oneFeatureIndex.txt')
    myMat = np.mat(myDat)
    myDatTest = utils.loadDataSet('./data/oneFeatureIndexTest.txt')
    myDatTest = np.mat(myDatTest)
    # 回归树
    # ---------------------------------------------
    # 全是连续属性,且只有一列
    #myTree = createTree(myMat)
    #print myTree

    # ---------------------------------------------
    # 预剪枝：减小误差和元素数的阈值，ops=(0, 1)表示不剪枝
    #myTree = createTree(myMat, ops=(0, 1))
    #print myTree

    # ---------------------------------------------
    # 后剪枝：通过测试数据，对预测模型进行合并判断

    # 不剪枝
    #myTree = createTree(myMat, ops=(0, 1))
    #print myTree
    #myFinalTree = prune(myTree, myDatTest)
    #print myFinalTree

    # ---------------------------------------------
    # 模型树
    #myTree = createTree(myMat, utils.modelLeaf, utils.modelErr)
    #print myTree

    # ---------------------------------------------
    # 回归树 VS 模型树 VS 线性回归
    """
    trainMat = np.mat(utils.loadDataSet('./data/bikeSpeedIQTrain.txt'))
    testMat = np.mat(utils.loadDataSet('./data/bikeSpeedIQTest.txt'))

    # 回归树
    myTree1 = createTree(trainMat, ops=(1, 20))
    print myTree1
    # 第0列是输入，第一列是输出
    yHat1 = utils.createForeCast(myTree1, testMat[:, 0])
    trueVal = testMat[:, 1]
    # 计算线性相关系数
    print "regTree:", np.corrcoef(yHat1, trueVal,rowvar=0)[0, 1]

    # 模型树
    myTree2 = createTree(trainMat, utils.modelLeaf, utils.modelErr, ops=(1, 20))
    yHat2 = utils.createForeCast(myTree2, testMat[:, 0], utils.modelTreeEval)
    print "modelTree:", np.corrcoef(yHat2, testMat[:, 1],rowvar=0)[0, 1]

    # 线性回归
    ws, X, Y = utils.linearSolve(trainMat)
    m = len(testMat[:, 0])
    yHat3 = np.mat(np.zeros((m, 1)))
    for i in range(np.shape(testMat)[0]):
        yHat3[i] = testMat[i, 0]*ws[1, 0] + ws[0, 0]
    print "lr:", np.corrcoef(yHat3, testMat[:, 1],rowvar=0)[0, 1]
    """