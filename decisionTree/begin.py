#!/usr/bin/python
# coding:utf-8
"""
决策树的算法实现
sklearn的决策树接口使用

by:guoKaiSama
"""
import classifierTree.ID3 as ID3
import classifierTree.C4_5 as C45
import classifierTree.sklearnClassfierTree as SKT
if __name__=="__main__":

    # begin *****************决策树****************** begin #
    # 利用信息增益生成决策树(ID3)
    #ID3.xiguaTest()

    # 眼镜数据与蘑菇数据的实验
    #ID3.glassTest()
    #ID3.mushroomTest()


    # 利用信息增益率生成决策树(C4_5)
    #C45.xiguaTest()

    # 分类树，sklearn
    SKT.classifierTree()

    # end *****************决策树****************** end #


    # begin *****************回归树***************** begin #

    # regressorTree.regressorTree()
    # 测试数据集
    #testMat = mat(eye(4))
    #print(testMat)
    #print(type(testMat))
    #mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    #print(mat0, '\n-----------\n', mat1)

    # # 回归树
    # myDat = loadDataSet('./data/data1.txt')
    # print 'myDat=', myDat
    # myMat = mat(myDat)
    # print 'myMat=',  myMat
    # myTree = createTree(myMat)
    # print myTree

    # # 1. 预剪枝就是：提起设置最大误差数和最少元素数
    # myDat = loadDataSet('./data/data3.txt')
    # myMat = mat(myDat)
    # myTree = createTree(myMat, ops=(0, 1))
    # print myTree

    # # 2. 后剪枝就是：通过测试数据，对预测模型进行合并判断
    # myDatTest = loadDataSet('./data/data3test.txt')
    # myMat2Test = mat(myDatTest)
    # myFinalTree = prune(myTree, myMat2Test)
    # print '\n\n\n-------------------'
    # print myFinalTree

    # # --------
    # # 模型树求解
    # myDat = loadDataSet('./data/data4.txt')
    # myMat = mat(myDat)
    # myTree = createTree(myMat, modelLeaf, modelErr)
    # print myTree

    # # # 回归树 VS 模型树 VS 线性回归
    # trainMat = mat(loadDataSet('./data/bikeSpeedVsIq_train.txt'))
    # testMat = mat(loadDataSet('./data/bikeSpeedVsIq_test.txt'))
    # # # 回归树
    # myTree1 = createTree(trainMat, ops=(1, 20))
    # print myTree1
    # yHat1 = createForeCast(myTree1, testMat[:, 0])
    # print "--------------\n"
    # # print yHat1
    # # print "ssss==>", testMat[:, 1]
    # # corrcoef 返回皮尔森乘积矩相关系数
    # print "regTree:", corrcoef(yHat1, testMat[:, 1],rowvar=0)[0, 1]

    # # 模型树
    # myTree2 = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    # yHat2 = createForeCast(myTree2, testMat[:, 0], modelTreeEval)
    # print myTree2
    # print "modelTree:", corrcoef(yHat2, testMat[:, 1],rowvar=0)[0, 1]

    # # 线性回归
    # ws, X, Y = linearSolve(trainMat)
    # print ws
    # m = len(testMat[:, 0])
    # yHat3 = mat(zeros((m, 1)))
    # for i in range(shape(testMat)[0]):
    #     yHat3[i] = testMat[i, 0]*ws[1, 0] + ws[0, 0]
    # print "lr:", corrcoef(yHat3, testMat[:, 1],rowvar=0)[0, 1]