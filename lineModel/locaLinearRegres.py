#!/usr/bin/python
# coding:utf8
import numpy as np
import matplotlib.pylab as plt
import utils

# 局部加权线性回归
def localLinearRegres(testPoint,xArr,yArr,k=1.0):
    '''
        Description：
            在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归。
        Args：
            testPoint：某样本点
            xArr：所有样本的feature
            yArr：每个样本对应的lable
            k:关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关
        Returns:
            testPoint * ws：预测值
        Notes:
            算法思路：假设预测点为i（共m个样本点），遍历1到m个样本点（含第i个），算出每一个样本点与预测点的距离，
            用到计算权重的公式，w = e^((x^((i))-x) / -2k^2)

            理解：x为某个预测点，x^((i))为样本点，样本点距离预测点越小，贡献的误差越大（权值越大）.越大则贡献的误差越小（权值越小）。
    '''

    # mat() 函数是将array转换为矩阵的函数， mat().T 是转换为矩阵之后，再进行转置操作
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T

    # 获得xMat矩阵的行数
    m,n = np.shape(xMat)

    # eye()返回一个对角线元素为1，其他元素为0的二维数组，即初始化了一个权重矩阵weights
    weights = np.mat(np.eye((m)))

    for j in range(m):
        # 计算 testPoint 与输入样本点之间的距离
        diffMat = testPoint - xMat[j,:]
        # k控制衰减的速度
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))

    xTx = xMat.T * (weights * xMat)

    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return

    # 计算出回归系数的一个估计
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

# 测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
def lwlrTest(testArr,xArr,yArr,k=1.0):

    '''
        Args：
            testArr：测试所用的所有样本点
            xArr：样本的特征数据，即 feature
            yArr：每个样本对应的类别标签
            k：控制核函数的衰减速率
        Returns：
            yHat：预测点的估计值
    '''
    # 得到样本点的总数

    m,n = np.shape(testArr)
    # 构建一个全部都是 0 的 1 * m 的矩阵
    yHat = np.zeros(m)
    # 循环所有的数据点，并将lwlr运用于所有的数据点
    for i in range(m):
        yHat[i] = localLinearRegres(testArr[i],xArr,yArr,k)
    # 返回估计值
    return yHat

def localLinearRegresMethod(k):
    xArr, yArr = utils.loadDataSet("./data/fullRankData.txt")
    yHat = lwlrTest(xArr, xArr, yArr, k)
    xMat = np.mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort=xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0] , s=2, c='red')
    plt.show()