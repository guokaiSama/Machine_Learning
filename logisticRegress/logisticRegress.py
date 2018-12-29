#!/usr/bin/python
# coding:utf8
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# 解析以tab键分隔的文件中的浮点数
def loadDataSet(fileName):
    """
    Returns：
        dataMat ：  feature 对应的数据集
        labelMat ： feature 对应的分类标签，即类别标签
    """
    dataMat = []
    labelMat = []
    with open(fileName, 'r') as f:
        for line in f.readlines():
            line_arr = line.strip().split()
            # 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
            dataMat.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
            labelMat.append(int(line_arr[2]))
    return dataMat, labelMat


# sigmod函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# 画出数据集和 Logistic 回归最佳拟合直线的函数
def plotBestFit(dataArr, labelMat, weights):
    '''
    dataArr:样本数据的特征
    labelMat:样本数据的类别标签，即目标变量
    weights:回归系数
    '''
    n = np.shape(dataArr)[0]
    x1Lable0 = []
    x2Lable0 = []
    x1Lable1 = []
    x2Lable1 = []

    for i in range(n):
        if int(labelMat[i])== 1:
            x1Lable0.append(dataArr[i,1])
            x2Lable0.append(dataArr[i,2])
        else:
            x1Lable1.append(dataArr[i,1])
            x2Lable1.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1Lable0, x2Lable0, s=30, c='red', marker='s')
    ax.scatter(x1Lable1, x2Lable1, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    """
    f(x) = w0+w1*x1+w2*x2
    x1对应x坐标，x2对应y坐标，而f(x)被磨合误差给算到了w0,w1,w2上
    """
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# 采用梯度上升法(损失函数未取反，所以使用梯度上升)
# 找到 Logistic 分类器在此数据集上的最佳回归系数
def gradAscent(dataMatIn, classLabels):
    """
    ataMatIn 样本取值
    classLabels 是类别标签
    """
    # 转化为矩阵[[1,1,2],[1,1,2]....]
    dataMatrix = np.mat(dataMatIn)

    # 转化为矩阵[[0,1,0,1,0,1.....]]，并转置[[0],[1],[0].....]
    labelMat = np.mat(classLabels).transpose()

    # m:样本数 n:特征数
    m,n = np.shape(dataMatrix)

    # alpha代表向目标移动的步长
    alpha = 0.001

    # 迭代次数
    maxCycles = 500

    # 初始化 weights， n个长度为1的向量，值均为1
    weights = np.ones((n,1))

    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return np.array(weights)


# 随机梯度上升
def stocGradAscentOri(data_mat, class_labels):
    """
    随机梯度上升，只使用一个样本点来更新回归系数
    :param data_mat: 输入数据的数据特征（除去最后一列）,ndarray
    :param class_labels: 输入数据的类别标签（最后一列数据）
    :return: 得到的最佳回归系数
    """
    m, n = np.shape(data_mat)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # sum(data_mat[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,
        # 此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(data_mat[i] * weights))
        error = class_labels[i] - h
        # 还是和上面一样，这个先去看推导，再写程序
        weights = weights + alpha * error * data_mat[i]
    return weights


# 改进版随机梯度上升
def stocGradAscentPro(data_mat, class_labels, num_iter=150):
    """
    改进版的随机梯度上升，使用随机的一个样本来更新回归系数
    :param data_mat: 输入数据的数据特征（除去最后一列）,ndarray
    :param class_labels: 输入数据的类别标签（最后一列数据
    :param num_iter: 迭代次数
    :return: 得到的最佳回归系数
    """
    m, n = np.shape(data_mat)
    weights = np.ones(n)
    for j in range(num_iter):
        # 这里必须要用list，不然后面的del没法使用
        data_index = list(range(m))
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机产生一个 0～len()之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(data_mat[data_index[rand_index]] * weights))
            error = class_labels[data_index[rand_index]] - h
            weights = weights + alpha * error * data_mat[data_index[rand_index]]
            del(data_index[rand_index])
    return weights


# 最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
def classifyVector(in_x, weights):
    """
    :param in_x: 特征向量，features
    :param weights: 根据梯度下降/随机梯度下降 计算得到的回归系数
    :return:
    """
    prob = sigmoid(np.sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    return 0.0


# 从疝气病症预测病马的死亡率
# 疝病是描述马胃肠痛的术语。然而，这种病不一定源自马的胃肠问题，其他问题也可能引发马疝病
def horseColicTest():
    f_train = open('./data/HorseColicTraining.txt', 'r')
    f_test = open('./data/HorseColicTest.txt', 'r')
    training_set = []
    training_labels = []
    # 解析训练数据集中的数据特征和Labels
    # trainingSet 中存储训练数据集的特征，trainingLabels 存储训练数据集的样本对应的分类标签
    for line in f_train.readlines():
        curr_line = line.strip().split('\t')
        # 这里如果就一个空的元素，则跳过本次循环
        if len(curr_line) == 1:
            continue
        line_arr = [float(curr_line[i]) for i in range(21)]
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    # 使用 改进后的 随机梯度上升算法 求得在此数据集上的最佳回归系数 trainWeights
    train_weights = stocGradAscentPro(np.array(training_set), training_labels, 500)
    error_count = 0
    num_test_vec = 0.0
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in f_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1:
            continue    # 这里如果就一个空的元素，则跳过本次循环
        line_arr = [float(curr_line[i]) for i in range(21)]
        if int(classifyVector(np.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = error_count / num_test_vec
    print('the error rate is {}'.format(error_rate))
    return error_rate


def fitLogistic():
    """
    调用 colicTest() 10次并求结果的平均值
    """
    num_tests = 10
    error_sum = 0
    for k in range(num_tests):
        error_sum += horseColicTest()
    print('after {} iteration the average error rate is {}'.format(num_tests, error_sum / num_tests))


def testLogisticRegress():
    # 加载数据
    dataMat, labelMat = loadDataSet("./data/logisticRegressTestData.txt")

    # 训练模型
    dataArr = np.array(dataMat)
    #weights = gradAscent(dataArr, labelMat)

    # 优化后的随机梯度上升
    weights = stocGradAscentPro(dataArr, labelMat)

    # 数据可视化
    plotBestFit(dataArr, labelMat, weights)


# 个人收入预测（是否大于50k）
def personalIncome():
    datas = pd.read_csv('./data/personalIncome.csv')
    datas.columns = ["age","date","sex","worthRising","worthLoss","workTime","income"]
    datas['age'] = datas['age'].apply(lambda x: float(x))
    datas['date'] = datas['date'].apply(lambda x: float(x))
    datas['sex'] = datas['sex'].apply(lambda x: 1.0 if str(x)=="Male" else 0.0)
    datas['worthRising'] = datas['worthRising'].apply(lambda x: float(x))
    datas['worthLoss'] = datas['worthLoss'].apply(lambda x: float(x))
    datas['workTime'] = datas['workTime'].apply(lambda x: float(x))
    datas['income'] = datas['income'].apply(lambda x: 1.0 if str(x)=="<=50K" else 0.0)
    X,Y=datas.ix[:,0:-2],datas.ix[:,-1]
    training_set,test_set,training_labels,test_labels=train_test_split(X,Y,test_size=0.3,random_state=0)
    train_weights = stocGradAscentPro(np.array(training_set), training_labels, 500)
    # 使用 改进后的 随机梯度上升算法 求得在此数据集上的最佳回归系数 trainWeights
    error_count = 0
    num_test_vec = 0.0

    """"
    
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in f_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1:
            continue  # 这里如果就一个空的元素，则跳过本次循环
        line_arr = [float(curr_line[i]) for i in range(21)]
        if int(classifyVector(np.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = error_count / num_test_vec
    print('the error rate is {}'.format(error_rate))
    return error_rate
    """
