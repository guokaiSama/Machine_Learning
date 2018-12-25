#!/usr/bin/python
# coding:utf8
import numpy as np
from bs4 import BeautifulSoup
import os
import sys
# 分析html页面，生成retX和retY列表
def scrapePage(retX, retY, inFile, year, numPce, origPrc):
    # 打开并读取HTML文件
    fr = open(inFile)
    soup = BeautifulSoup(fr.read())
    i=1

    # 根据HTML页面结构进行解析
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text.lower()

        # 查找是否有全新标签
        if (title.find('new') > -1) or (title.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0

        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print "item #%d did not sell" % i
        else:
            # 解析页面获取当前价格
            price = currentRow[0].findAll('td')[4].text.replace('$','').replace(',','')
            if len(price)>1:
                price = price.replace('Free shipping', '')
            price = float(price)

            # 去掉不完整的套装价格
            if price > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (year,numPce,newFlag,origPrc, price)
                    retX.append([year, numPce, newFlag, origPrc])
                    retY.append(price)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)

# 依次读取六种乐高套装的数据，并生成数据矩阵
def setDataCollect(filePath):
    retX=list()
    retY=list()
    # 8288是乐高的某一个套装型号,,,,49.99是原价
    scrapePage(retX, retY, './data/setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, './data/setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, './data/setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, './data/setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, './data/setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, './data/setHtml/lego10196.html', 2009, 3263, 249.99)
    return retX, retY

# 交叉验证测试岭回归
def crossValidation(xArr,yArr,numVal=10):
    # 获得数据点个数，xArr和yArr具有相同长度
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal,30))

    # 主循环 交叉验证循环
    for i in range(numVal):
        # 随机拆分数据，将数据分为训练集（90%）和测试集（10%）
        trainX=[]; trainY=[]
        testX = []; testY = []

        # 对数据进行混洗操作
        random.shuffle(indexList)

        # 切分训练集和测试集
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])

        # 获得回归系数矩阵
        wMat = ridgeTest(trainX,trainY)

        # 循环遍历矩阵中的30组回归系数
        for k in range(30):
            # 读取训练集和数据集
            matTestX = mat(testX); matTrainX=mat(trainX)
            # 对数据进行标准化
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain

            # 测试回归效果并存储
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)

            # 计算误差
            errorMat[i,k] = ((yEst.T.A-array(testY))**2).sum()

    # 计算误差估计值的均值
    meanErrors = mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]

    # 不要使用标准化的数据，需要对数据进行还原来得到输出结果
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX

    # 输出构建的模型
    print "the best model from Ridge Regression is:\n",unReg
    print "with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat)


# 预测乐高玩具的价格
def legoPricePredict():
    lgX,lgY=setDataCollect(".\data\setHtml")
    crossValidation(lgX, lgY, 10)
