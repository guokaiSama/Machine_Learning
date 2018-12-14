#!/usr/bin/python
# coding:utf-8
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

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
