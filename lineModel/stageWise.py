#!/usr/bin/python
# coding:utf8
import numpy as np
import utils
import matplotlib.pylab as plt
# 计算分析预测误差的大小
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()


def stageWise(xMat,yMat,eps=0.01,numIt=100):
    yMat = yMat.T
    m,n=np.shape(xMat)
    # 初始化参数
    ws = np.zeros((n,1))

    wsMax = ws.copy()
    for i in range(numIt):
        lowestError = np.inf
        # 每个方向进行优化
        for j in range(n):
            # 尝试不同的系数
            for sign in [-1,1]:
                wsTest = ws.copy()
                #更新系数
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
    return ws

# 逐步线性回归
def stageWiseTest():
    xArr,yArr=utils.loadDataSet("./data/fullRankData.txt")
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    ws=stageWise(xMat,yMat,0.1,200)
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