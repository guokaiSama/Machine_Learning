#!/usr/bin/python
# coding:utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# sawtooth 波浪方框
decisionNode = dict(boxstyle="sawtooth", fc="0.8")

# round4 矩形方框 , fc表示字体颜色的深浅 0.1~0.9 依次变浅
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# 西瓜数据集2.0 P77
def createXiGuaDataSet():
    """
    创建西瓜数据集
    color：青绿：0，乌黑：1，浅白：2
    pedicle:蜷缩：0，稍蜷：1，硬挺：2,
    sound：浊响：0，沉闷：1，清脆：2
    texture：清晰：0，稍糊：1，模糊：2
    navel：凹陷：0，稍凹：1，平坦：2
    touch：赢滑：0，软粘：1,
    """
    dataSet = [[0, 0, 0,0, 0, 0,'yes'],
               [1, 0, 1,0, 0, 0,'yes'],
               [1, 0, 0,0, 0, 0,'yes'],
               [0, 0, 1,0, 0, 0,'yes'],
               [2, 0, 0,0, 0, 0,'yes'],
               [0, 1, 0,0, 1, 1, 'yes'],
               [1, 1, 0,1, 1, 1, 'yes'],
               [1, 1, 0,0, 1, 0, 'yes'],
               [1, 1, 1,1, 1, 0, 'no'],
               [0, 2, 2,0, 2, 1, 'no'],
               [2, 0, 0,2, 2, 1, 'no'],
               [0, 1, 0,1, 0, 0, 'no'],
               [2, 1, 1,1, 0, 0, 'no'],
               [1, 1, 0,0, 1, 1, 'no'],
               [2, 0, 0,2, 2, 0, 'no'],
               [0, 0, 1,1, 1, 0, 'no']
               ]
    featureName = ['color', 'pedicle','sound','texture','navel','touch']
    return dataSet, featureName


# 加载眼镜数据集
def loadGlassDataSet():
    fr=open("./data/lenses.txt","rb")
    lecses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lecses,lensesLabels

# 加载蘑菇数据集
def loadMushroomDataSet():
    datas = pd.read_csv('./data/mushrooms_new.csv')
    labels=list(datas.columns)[1:]

    # 需要将lable放到最后，所以这里是将第一列移到第零列的过程
    chang_lable_postion=list(labels)
    chang_lable_postion.append("class")
    datas = datas.loc[:, chang_lable_postion]
    dataSet=np.array(datas).tolist()
    return dataSet,labels


# 获取叶节点个数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是否为dict, 不是+1
        if type(secondDict[key]) is dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]

    # 根节点开始遍历
    for key in secondDict.keys():
        # 判断子节点是不是dict, 求分枝的深度
        # ----------写法1 start ---------------
        if type(secondDict[key]) is dict:
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        maxDepth = max(maxDepth, thisDepth)
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

# 绘制节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotTree(myTree, parentPt, nodeTxt):
    # 获取叶子节点的数量
    numLeafs = getNumLeafs(myTree)

    # 找出第1个中心点的位置，然后与 parentPt定点进行划线
    cntrPt = (plotTree.xOff + (1 + numLeafs) / 2 / plotTree.totalW, plotTree.yOff)

    # 并打印输入对应的文字
    plotMidText(cntrPt, parentPt, nodeTxt)

    firstStr = list(myTree.keys())[0]

    # 可视化Node分支点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)

    # 根节点的值
    secondDict = myTree[firstStr]

    # y值 = 最高点-层数的高度[第二个节点位置]
    plotTree.yOff = plotTree.yOff - 1 / plotTree.totalD
    for key in secondDict.keys():
        # 判断该节点是否是Node节点
        if type(secondDict[key]) is dict:
            # 如果是就递归调用[recursion]
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 如果不是，就在原来节点一半的地方找到节点的坐标
            plotTree.xOff = plotTree.xOff + 1 / plotTree.totalW
            # 可视化该节点位置
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # 并打印输入对应的文字
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1 / plotTree.totalD


def createPlot(inTree):
    # 设置背景图片的颜色
    fig = plt.figure(1, facecolor='white')
    # 刷新当前图片
    fig.clf()

    axprops = dict(xticks=[], yticks=[])

    # 将生成的图片分成一行一列，createPlot.ax1 为第 1 个子图
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))

    # 半个节点的长度
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

