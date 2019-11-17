#!/usr/bin/python
# coding:utf-8
import numpy as np
# 对输入数据进行格式化处理，在原数据矩阵上增加第0列，元素的值都是1，
# 也就是增加偏移值，和我们之前的简单线性回归是一个套路，增加一个偏移量
def modelTreeEval(model, inDat):
    """
    Args:
        model -- 输入的模型，可选值为 回归树模型 或者 模型树模型
        inDat -- 输入的测试数据
    Returns:
        float(X * model) -- 将测试数据乘以 回归系数 得到一个预测值 ，转化为 浮点数 返回
    """
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1: n+1] = inDat
    # 计算线性模型的输出
    return float(X * model)


# 回归树测试案例，为了和 modelTreeEval() 保持一致，保留两个输入参数
def regTreeEval(model, inDat):
    """
    Args:
        model -- 指定模型，可选值为 回归树模型 或者 模型树模型，这里为回归树
        inDat -- 输入的测试数据
    Returns:
        float(model) -- 将输入的模型数据转换为 浮点数 返回
    """
    return float(model)


# 计算预测的结果:自顶向下遍历整棵树，直到命中叶节点为止
def treeForeCast(tree, inData, modelEval=regTreeEval):
    """
    Args:
        tree -- 已经训练好的树的模型
        inData -- 输入的测试数据，只有一行
        modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树）
    Returns:
        返回预测值
    """
    # 叶节点
    if not isTree(tree):
        return modelEval(tree, inData)

    # 只适合inData只有一列的情况，否则会产生异常
    if inData[0, tree['spIndex']] <= tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 对特定模型的树进行预测，可以是 回归树 也可以是 模型树
def createForeCast(tree, testData, modelEval=regTreeEval):
    """
    Args:
        tree -- 已经训练好的树的模型
        testData -- 输入的测试数据
        modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
    Returns:
        返回预测值矩阵
    """
    m = len(testData)
    # np.zeros 生成m行n列的全0矩阵
    yHat = np.mat(np.zeros((m, 1)))

    # 遍历每一行，计算其预测值，保存在yHat
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


# 将数据集格式化成目标变量Y和自变量X，执行简单的线性回归，得到ws
def linearSolve(dataSet):
    """
    Args:
        dataSet -- 输入数据
    Returns:
        ws -- 执行线性回归的回归系数
        X -- 格式化自变量X
        Y -- 格式化目标变量Y
    """
    m, n = np.shape(dataSet)
    # 产生一个m行，n列全是1的矩阵
    # X的0列为1，常数项，1-n保存输入值，不包括输出值
    X = np.mat(np.ones((m, n)))
    X[:, 1: n] = dataSet[:, 0: n-1]

    # 保存输出值
    Y = dataSet[:, -1]


    # 转置矩阵*矩阵
    xTx = X.T * X
    # 如果矩阵的逆不存在，会造成程序异常
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')

    # 最小二乘法求最优解:  w0*1+w1*x1=y
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


# 计算所有样本的值减去均值的平方和（\sum(y_i-c_1)^2）。可以通过方差*样本数求出
# 即通过决策树划分，可以让靠近的数据分到同一类中去
def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

# regLeaf 是产生叶节点的函数，叶节点的值是lable的均值，即用聚类中心点来代表这类数据
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

# 得到模型的ws系数：f(x) = x0 + x1*featrue1+ x2*featrue2 ...
def modelLeaf(dataSet):
    """
    Desc:
        当数据不再需要切分的时候，生成叶节点的模型。
    Args:
        dataSet -- 输入数据集
    Returns:
        调用 linearSolve 函数，返回得到的 回归系数ws
    """
    ws, X, Y = linearSolve(dataSet)
    return ws


# 计算线性模型的误差值
def modelErr(dataSet):
    """
    Args:
        dataSet -- 输入数据集
    Returns:
        调用 linearSolve 函数，返回 yHat 和 Y 之间的平方误差。
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    # 计算平方误差
    return sum(np.power(Y - yHat, 2))


# 从上往下遍历树直到叶节点为止，如果找到两个叶节点则计算它们的平均值。
def getMean(tree):
    """
    Args:
        tree -- 输入的树
    Returns:
        返回 tree 节点的平均值
    """
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

# 测试输入变量是否是一棵树,即是否是一个字典
def isTree(obj):
    return (type(obj).__name__ == 'dict')

# 默认解析的数据是用tab分隔，并且是数值类型
def loadDataSet(fileName):
    # 假定最后一列是结果值
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将所有的元素转化为float类型
        # 根据提供的函数对指定序列做映射
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat


# 生成随机的数据集
def gen_random_set():
    # 创建一个随机的数据集
    rng = np.random.RandomState(1)

    # 以给定的shape创建一个数组，数据分布在0-1之间。
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()

    # 每隔5个样本点，对数据进行异常设置
    y[::5] += 3 * (0.5 - rng.rand(16))
    return X,y