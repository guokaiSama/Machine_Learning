#!/usr/bin/python
# coding:utf-8
from sklearn.tree import DecisionTreeRegressor
import utils.utils as utils
import numpy as np
import matplotlib.pyplot as plt
# sklearn调用回归树
def regressorTree():
    # 拟合回归模型，保持 max_depth=5 不变，增加 min_samples_leaf=6 的参数，效果进一步提升了
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_2 = DecisionTreeRegressor(min_samples_leaf=6)

    X, y = utils.gen_random_set()

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
