#!/usr/bin/python
# coding:utf-8
"""
线性回归的算法实现
sklearn的线性回归接口使用

by:guoKaiSama
"""
import standLinerRegres
import locaLinearRegres
import ridgeForRegress
from numpy import mat
import logisticRegress
import utils
import methodCompare
import stageWise
import sklearnMethod
import legoPricePredict
if __name__=="__main__":
    # 标准线性回归，满秩,最小二乘法
    #standLinerRegres.standRegresLeastSquare()
    #standLinerRegres.standRegresGrand()

    # 局部加权线性回归
    #locaLinearRegres.localLinearRegresMethod(k=0.005)

    # 预测鲍鱼的年龄,对比标准线性回归和局部加权线性回归
    #methodCompare.abaloneTest()

    # 测试岭回归L2,最小二乘法
    #ridgeForRegress.testRidgeRegresLeastSquare()
    # 测试岭回归L2,梯度下降法
    #ridgeForRegress.testRidgeRegresGrand()

    # 前向逐步回归算法，与Lasso的结果相似
    #stageWise.stageWiseTest()

    # sklearn 的逻辑回归
    #sklearnMethod.sklearnLiner()

    # 线性回归综合示例,预测玩具的价格
    legoPricePredict.legoPricePredict()
    # 测试逻辑回归与梯度上升
    #logisticRegress.testLogisticRegress()
    #logisticRegress.fitLogistic()

