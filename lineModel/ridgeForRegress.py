#!/usr/bin/python
# coding:utf8
import numpy as np
import standLinerRegres
import matplotlib.pylab as plt
import utils

# 按列进行规范化
def regularize(xMat):
    inMat = xMat.copy()
    # 计算平均值然后减去它
    inMeans = np.mean(inMat, 0)
    # 计算除以X_i的方差
    inVar = np.var(inMat, 0)
    inMat = (inMat - inMeans) / inVar
    return inMat

# 梯度下降法
def ridgeRegresGrand(xMat,yMat,lam=0.2):
    '''
        Desc：
            这个函数实现了给定 lambda 下的岭回归求解。
        Args：
            xMat：样本的特征数据，即 feature
            yMat：每个样本对应的类别标签，即目标变量，实际值
            lam：引入的一个λ值，使得矩阵非奇异
        Returns：
            经过岭回归公式计算得到的回归系数
    '''
    alpha = 0.8
    maxCycles = 500
    # 转化为矩阵[[0,1,0,1,0,1.....]]，并转置[[0],[1],[0].....]
    labelMat = yMat

    # dataMatrix的类型为什么变了
    dataMatrix = np.mat(xMat)
    # m->数据量，样本数 n->特征数
    m, n = np.shape(dataMatrix)

    # 初始化权重weights都为1，然后进优化
    # ones((n,1)) 创建n行1列的数组，元素值均为1
    weights = np.ones((n, 1))

    for k in range(maxCycles):
        # 矩阵乘法
        hypothesis = np.dot(dataMatrix, weights)

        # labelMat是实际值
        loss = (hypothesis - labelMat)
        # 梯度
        gradient = np.dot(dataMatrix.transpose(), loss) / m
        weights = weights - (alpha * gradient+np.dot(lam,weights))
        # 损失
        cost = 0.5 * m * np.sum(np.square(np.dot(dataMatrix, weights) - yMat))
        print "cost: ", cost
    return np.array(weights)


# 最小二乘法
def ridgeRegresLeastSquare(xMat,yMat,lam=0.2):
    '''
        Desc：
            这个函数实现了给定 lambda 下的岭回归求解。
        Args：
            xMat：样本的特征数据，即 feature
            yMat：每个样本对应的类别标签，即目标变量，实际值
            lam：引入的一个λ值，使得矩阵非奇异
        Returns：
            经过岭回归公式计算得到的回归系数
    '''

    xTx = xMat.T*xMat

    # 岭回归就是在矩阵 xTx 上加一个 λI 从而使得矩阵非奇异，进而能对 xTx + λI 求逆
    denom = xTx + np.eye(np.shape(xMat)[1])*lam

    # 检查行列式是否为零，即矩阵是否可逆，行列式为0的话就不可逆，不为0的话就是可逆。
    if np.linalg.det(denom) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return

    ws = denom.I * (xMat.T*yMat)
    return ws

# 在一组lambda上测试岭回归结果
def ridgeTest(xArr,yArr,lam=0.2,method=ridgeRegresLeastSquare):
    '''
        Args：
            xArr：样本数据的特征，即 feature
            yArr：样本数据的类别标签，即真实数据
        Returns：
            wMat：将所有的回归系数输出到一个矩阵并返回
    '''

    xMat = np.mat(xArr)
    yMat=np.mat(yArr).T
    """
    
    # 计算Y的均值
    yMean = np.mean(yMat,0)
    # Y的所有的特征减去均值
    yMat = yMat - yMean

    # 标准化 xMat
    xMat = regularize(xMat)

    """
    ws = method(xMat,yMat,lam)
    return ws

# 测试岭回归，最小二乘法
def testRidgeRegresLeastSquare():
    abX,abY = utils.loadDataSet("./data/fullRankData.txt")
    xMat = np.mat(abX)
    yMat = np.mat(abY)
    ws = ridgeTest(abX, abY,lam=0.1)

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


# 测试岭回归，梯度下降法
def testRidgeRegresGrand():
    abX, abY = utils.loadDataSet("./data/fullRankData.txt")
    xMat = np.mat(abX)
    yMat = np.mat(abY)
    ws = ridgeTest(abX, abY, lam=0.01,method=ridgeRegresGrand)

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
