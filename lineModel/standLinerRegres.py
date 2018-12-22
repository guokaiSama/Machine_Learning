#!/usr/bin/python
# coding:utf8
import numpy as np
import matplotlib.pylab as plt
import utils
# 采用梯度下降法
def gradAscent(dataMatrix, classLabels,alpha = 0.01,maxCycles = 500):
    """
    :param dataMatrix:
        dataMatIn 是一个2维NumPy数组，每列分别代表每个不同的特征，每行则代表每个训练样本
        alpha代表向目标移动的步长
        maxCycles表示迭代次数
    alpha = 0.001
    :param classLabels:
        classLabels 是类别标签，它是一个 1*100 的行向量。为了便于矩阵计算，需要将该行向量转换为列向量，做法是将原向量转置，再将它赋值给labelMat。
    :return:
    """

    # 转化为矩阵[[0,1,0,1,0,1.....]]，并转置[[0],[1],[0].....]
    labelMat = classLabels.transpose()

    # dataMatrix的类型为什么变了
    dataMatrix = np.mat(dataMatrix)
    # m->数据量，样本数 n->特征数
    m,n = np.shape(dataMatrix)

    # 初始化权重weights都为1，然后进优化
    # ones((n,1)) 创建n行1列的数组，元素值均为1
    weights = np.ones((n,1))

    for k in range(maxCycles):
        # 矩阵乘法
        hypothesis = np.dot(dataMatrix,weights)

        # labelMat是实际值
        loss = (hypothesis-labelMat)
        # 梯度
        gradient = np.dot(dataMatrix.transpose(),loss)/m

        weights = weights - alpha * gradient
        # 损失
        cost = 0.5 * m * np.sum(np.square(np.dot(dataMatrix, weights) - classLabels))
        print "cost: ",cost
    return np.array(weights)

# 随机梯度上升
def stocGradAscentOri(dataMatrix, classLabels,alpha = 0.01):
    """
    随机梯度上升，只使用一个样本点来更新回归系数
    :param dataMatrix: 输入数据的数据特征（除去最后一列）,ndarray
    :param classLabels: 输入数据的类别标签（最后一列数据）
    :return: 得到的最佳回归系数
    """
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    classLabels=classLabels.tolist()
    # 随机梯度上升，只使用一个样本点来更新回归系数
    for i in range(m):
        # 此处求出的 h 是一个具体的数值，而不是一个矩阵
        hypothesis = np.dot(dataMatrix[i], weights)
        error = hypothesis - classLabels[0][i]
        weights = weights - np.dot((alpha * error),dataMatrix[i])
        # 损失
        cost = np.sum(np.square(np.dot(dataMatrix, weights) - classLabels))
        print "cost: ", cost
    weights=weights.reshape(2,1)
    return weights


# 标准线性回归
def standRegres(xArr, yArr):
    '''
    Args:
        xArr ：样本数据
        yArr ：对应于输入数据的类别标签
    Returns:
        ws：回归系数
    '''

    # mat()函数将xArr，yArr转换为矩阵 mat().T 代表的是对矩阵进行转置操作
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    # x的转置乘以x
    xTx = xMat.T * xMat

    # 因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
    # linalg.det() 函数是用来求得矩阵的行列式的，如果矩阵的行列式为0，则这个矩阵是不可逆的，就无法进行接下来的运算
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # xTx.I表示xTx的逆矩阵。最小二乘法，求得w的最优解
    ws = xTx.I * (xMat.T * yMat)
    return ws

def standRegresLeastSquare():
    xArr, yArr = utils.loadDataSet("./data/fullRankData.txt")
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    ws = standRegres(xArr, yArr)

    fig = plt.figure()
    # add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
    ax = fig.add_subplot(111)
    # scatter 的x是xMat中的第二列，y是yMat的第一列
    ax.scatter([xMat[:, 1].flatten()], [yMat.T[:, 0].flatten().A[0]])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

def standRegresGrand():
    xArr, yArr = utils.loadDataSet("./data/fullRankData.txt")
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    # 标准梯度下降
    #ws = gradAscent(xArr, yMat,alpha = 0.01,maxCycles = 1000)
    # 步长太小，导致无法拟合
    #ws = gradAscent(xArr, yMat, alpha=0.0001, maxCycles=1000)

    #随机梯度下降
    ws = stocGradAscentOri(xArr, yMat,alpha = 0.05)

    fig = plt.figure()
    # add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
    ax = fig.add_subplot(111)
    # scatter 的x是xMat中的第二列，y是yMat的第一列
    ax.scatter([xMat[:, 1].flatten()], [yMat.T[:, 0].flatten().A[0]])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
