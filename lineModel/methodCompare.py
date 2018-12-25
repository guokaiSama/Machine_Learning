#!/usr/bin/python
# coding:utf-8
import locaLinearRegres
import utils
import standLinerRegres
import numpy as np
# 计算分析预测误差的大小
def rssError(yArr, yHatArr):
    '''
        Args:
            yArr：真实的目标变量
            yHatArr：预测得到的估计值
        Returns:
            计算真实值和估计值得到的值的平方和作为最后的返回值
    '''
    return ((yArr - yHatArr) ** 2).sum()

# 对比标准线性回归与局部加权线性回归
def abaloneTest():
    # 加载数据
    abX, abY = utils.loadDataSet("./data/abalone.txt")

    # 使用不同的核进行预测
    oldyHat01 = locaLinearRegres.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    oldyHat1 = locaLinearRegres.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    oldyHat10 = locaLinearRegres.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

    # 打印出不同的核预测值与训练数据集上的真实值之间的误差大小
    print("old yHat01 error Size is :", rssError(abY[0:99], oldyHat01.T))
    print("old yHat1 error Size is :", rssError(abY[0:99], oldyHat1.T))
    print("old yHat10 error Size is :", rssError(abY[0:99], oldyHat10.T))

    # 打印出 不同的核预测值 与 新数据集（测试数据集）上的真实值之间的误差大小
    newyHat01 = locaLinearRegres.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    print("new yHat01 error Size is :", rssError(abY[0:99], newyHat01.T))
    newyHat1 = locaLinearRegres.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    print("new yHat1 error Size is :", rssError(abY[0:99], newyHat1.T))
    newyHat10 = locaLinearRegres.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print("new yHat10 error Size is :", rssError(abY[0:99], newyHat10.T))

    # 使用简单的 线性回归 进行预测，与上面的计算进行比较
    standWs = standLinerRegres.standRegres(abX[0:99], abY[0:99])
    standyHat = np.mat(abX[100:199]) * standWs
    print("standRegress error Size is:", rssError(abY[100:199], standyHat.T.A))
