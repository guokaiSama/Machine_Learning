#!/usr/bin/python
# coding:utf-8
# 加载文本中的数据
import numpy as np
import math
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

# 计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

# 为给定数据集构建一个包含 k 个随机质心的集合。随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小和最大值来完成。然后生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
def randCent(dataMat, k):
    n = np.shape(dataMat)[1]  # 列的数量
    centroids = np.mat(np.zeros((k, n)))  # 创建k个质心矩阵
    for j in range(n):  # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataMat[:, j])  # 最小值
        rangeJ = float(max(dataMat[:, j]) - minJ)  # 范围 = 最大值 - 最小值
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))  # 随机生成
    return centroids


def kMeans(dataMat, k, distMeas=distEclud, createCent=randCent):
    # 行数
    m = np.shape(dataMat)[0]

    # m*2的矩阵，用来保存划分结果
    clusterAssment = np.mat(np.zeros((m, 2)))

    # 随机初始化k个质心
    centroids = createCent(dataMat, k)

    clusterChanged = True
    while clusterChanged:
        clusterChanged = False

        # 循环每一个样本，分配到最近的质心
        for i in range(m):
            minDist = np.inf
            minIndex = -1

            # 计算到每个质心的距离
            for j in range(k):
                distJI = distMeas(centroids[j, :],dataMat[i, :])

                #判断是否需要更新
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            # 改变簇分配结果
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        # 更新质心
        for cent in range(k):
            ptsInClust = dataMat[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


# 二分 KMeans 聚类算法, 基于 kMeans 基础之上的优化，以避免陷入局部最小值
def biKMeans(dataMat, k, distMeas=distEclud):
    """
    该算法首先将所有点作为一个簇，然后将该簇一分为二。
    之后选择其中一个簇继续进行划分，选择哪一个簇进行划分
    取决于对其划分时候可以最大程度降低 SSE（平方和误差）的值。
    上述基于 SSE 的划分过程不断重复，直到得到用户指定的簇数目为止。
    """
    m = np.shape(dataMat)[0]

    # m*2的矩阵，用来保存划分结果
    clusterAssment = np.mat(np.zeros((m, 2)))

    # 质心初始化为所有数据点的均值
    centroid0 = np.mean(dataMat, axis=0).tolist()[0]

    # 初始化只有 1 个质心的 list
    centList = [centroid0]

    # 计算所有数据点到初始质心的距离平方误差
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataMat[j, :])**2

    # 当质心数量小于 k 时
    while (len(centList) < k):
        lowestSSE = np.inf

        # 对每一个质心
        for i in range(len(centList)):
            # 获取当前簇 i 下的所有数据点
            ptsInCurrCluster = dataMat[np.nonzero(clusterAssment[:, 0].A == i)[0], :]

            # 将当前簇 i 进行二分 kMeans 处理
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)

            # 将二分 kMeans 结果中的平方和的距离进行求和
            sseSplit = np.sum(splitClustAss[:, 1])

            # 将未参与二分 kMeans 分配结果中的平方和的距离进行求和
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0],1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # 找出最好的簇分配结果
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)

        # 更新为最佳质心
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))

        # 更新原质心 list 中的第 i 个质心为使用二分 kMeans 后 bestNewCents 的第一个质心
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]

        # 添加 bestNewCents 的第二个质心
        centList.append(bestNewCents[1, :].tolist()[0])

        # 重新分配最好簇下的数据（质心）以及SSE
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment



def kMeansMethod():
    # 加载测试数据集
    dataMat = np.mat(loadDataSet('./data/dataSet.txt'))

    #  该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
    #  运行结果（多次运行结果可能会不一样，可以试试，原因为随机质心的影响，但总的结果是对的， 因为数据足够相似）
    myCentroids, clustAssing = kMeans(dataMat, 4)

    print('centroids=', myCentroids)


def biKMeansMethod():
    # 加载测试数据集
    dataMat = np.mat(loadDataSet('./data/dataSet.txt'))

    centList, myNewAssments = biKMeans(dataMat, 3)

    print('centList=', centList)
