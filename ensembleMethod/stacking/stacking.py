# coding:utf-8
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))

    # test_num行,n_fold列
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)

        # 产生次级学习器的特征（训练集）
        second_level_train_set[test_index] = clf.predict(x_tst)

        # 产生次级学习器的特征（测试集）
        test_nfolds_sets[:,i] = clf.predict(x_test)

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set


def stackingMethod():
    # 使用这5种分类算法进行融合
    rf_model = RandomForestClassifier()
    adb_model = AdaBoostClassifier()
    gdbc_model = GradientBoostingClassifier()
    et_model = ExtraTreesClassifier()
    svc_model = SVC()

    # 划分训练集与测试集
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2)

    train_sets = list()
    test_sets = list()
    for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
        train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
        train_sets.append(train_set)
        test_sets.append(test_set)

    meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

    # 使用决策树作为我们的次级分类器

    dt_model = DecisionTreeClassifier()
    dt_model.fit(meta_train, train_y)
    df_predict = dt_model.predict(meta_test)
    print(df_predict)


stackingMethod()