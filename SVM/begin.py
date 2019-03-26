#!/usr/bin/python
# coding:utf-8
"""
SVD算法的具体应用
by:guoKaiSama
"""
import svmSklearn
if __name__=="__main__":
    # 高斯核(RBF)
    #svmSklearn.sklearnSVM()

    # 简化版的SMO
    #svmSklearn.simpleSVM()

    # 完整版的SMO,选择合适的优化变量
    #svmSklearn.completeSVM()

    # 带核函数的SVM
    #svmSklearn.rbfSVM()

    # 手写数字识别：SVM
    svmSklearn.digitsSVM()


