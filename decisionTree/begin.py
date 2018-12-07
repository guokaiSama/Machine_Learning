#!/usr/bin/python
# coding:utf-8
"""
决策树的算法实现
sklearn的决策树接口使用

by:guoKaiSama
"""
import classifierTree
import regressorTree
if __name__=="__main__":
    # 分类树，sklearn
    #classifierTree.classifierTree()
    #手动实现决策树,鱼数据
    #classifierTree.fishTest()
    # 手动实现决策树,眼镜数据
    #classifierTree.glassTest()
    # 手动实现决策树,蘑菇数据集
    #classifierTree.mushroomTest()


    # 回归树，sklearn
    # regressorTree.regressorTree()
    # 测试数据集
    testMat = mat(eye(4))
    print(testMat)
    print(type(testMat))
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print(mat0, '\n-----------\n', mat1)

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