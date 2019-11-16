#!/usr/bin/python
# coding:utf-8
from sklearn.datasets import load_iris
from sklearn import tree
import sys
import os
import pydotplus
from IPython.display import Image
"""
scikit-learn中决策树的可视化一般需要安装graphviz。
主要包括graphviz的安装和python的graphviz插件的安装。
第一步是安装graphviz。下载地址在：http://www.graphviz.org/。将bin目录加到PATH
第二步是安装python插件graphviz： pip install graphviz
第三步是安装python插件pydotplus； pip install pydotplus

有时候python会graphviz，这时，可以在代码里面加入这一行：
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

"""
def sklearnGraph():
    iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    # 限制树的深度为4
    # clf = DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(iris.data, iris.target)
    with open("iris.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

    # 1、用graphviz的dot命令生成决策树的可视化文件
    # 当前目录会生成决策树的可视化文件iris.pdf
    # dot -Tpdf iris.dot -o iris.pdf

    # 2、用pydotplus生成iris.pdf

    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("iris.pdf")

    # 3、直接把图产生在ipython的notebook
    """
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=iris.feature_names,class_names=iris.target_names,filled=True, rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())
    """

