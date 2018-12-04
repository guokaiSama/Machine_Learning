#!/usr/bin/python
# coding:utf-8
"""
回归树
使用随机生成的样本点计算sin值，中间会添加一些噪声

by:guoKaiSama
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def gen_random_set():
    # 创建一个随机的数据集，参考 https://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.random.mtrand.RandomState.html
    rng = np.random.RandomState(1)

    # 以给定的shape创建一个数组，数据分布在0-1之间。
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    # 每隔5个样本点，对数据进行异常设置
    y[::5] += 3 * (0.5 - rng.rand(16))
    return X,y

def regressorTree():
    # 拟合回归模型，保持 max_depth=5 不变，增加 min_samples_leaf=6 的参数，效果进一步提升了
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_2 = DecisionTreeRegressor(min_samples_leaf=6)

    X, y = gen_random_set()

    # 训练模型
    regr_2.fit(X, y)

    # 生成测试样本
    # np.newaxis的作用就是在增加一维
    X_test = np.arange(0.0, 5.0, 0.01)
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    # 预测
    y_2 = regr_2.predict(X_test)

    # 绘制结果
    plt.figure()
    plt.scatter(X, y, c="darkorange", label="data")

    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")

    plt.legend()
    plt.show()
