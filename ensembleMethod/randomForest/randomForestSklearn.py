# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
def loadData():
    train = pd.read_csv('./data/train_modified.csv')
    # Disbursed就是lable
    target = 'Disbursed'
    IDcol = 'ID'
    train['Disbursed'].value_counts()

    # 获取属性列
    x_columns = [x for x in train.columns if x not in [target, IDcol]]
    X = train[x_columns]
    y = train['Disbursed']
    return X,y

def demoRF():
    X, y=loadData()
    # 使用默认参数训练
    rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    rf0.fit(X, y)

    # 袋外分数
    print rf0.oob_score_

    rf1 = RandomForestClassifier(n_estimators=60, max_depth=13, min_samples_split=110,
                                 min_samples_leaf=20, max_features='sqrt', oob_score=True, random_state=10)
    rf1.fit(X, y)
    print (rf1.oob_score_)

    rf2 = RandomForestClassifier(n_estimators=60, max_depth=13, min_samples_split=120,
                                 min_samples_leaf=20, max_features=7, oob_score=True, random_state=10)
    rf2.fit(X, y)
    print (rf2.oob_score_)
