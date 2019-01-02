#!/usr/bin/python
# coding:utf-8

"""
协同过滤：是通过将用户和其他用户的数据进行对比来实现推荐的。
当知道了两个用户或两个物品之间的相似度，我们就可以利用已有的数据来预测未知用户的喜好。

基于物品的相似度和基于用户的相似度：物品比较少则选择物品相似度，用户比较少则选择用户相似度。【矩阵还是小一点好计算】

基于物品的相似度：计算物品之间的距离。由于物品A和物品C 相似度(相关度)很高，所以给买A的人推荐C。

基于用户的相似度：计算用户之间的距离。由于用户A和用户C 相似度(相关度)很高，所以A和C是兴趣相投的人，对于C买的物品就会推荐给A。

算法流程：
寻找用户没有评级的菜肴，即在用户-物品矩阵中的0值。
在用户没有评级的所有物品中，对每个物品预计一个可能的评级分数。这就是说：我们认为用户可能会对物品的打分（这就是相似度计算的初衷）。
对这些物品的评分从高到低进行排序，返回前N个物品。


基于内容的推荐有以下特点：
（1）通过各种标签来标记菜肴
（2）将这些属性作为相似度计算所需要的数据


问题
1）在大规模的数据集上，SVD分解会降低程序的速度
2）存在其他很多规模扩展性的挑战性问题，比如矩阵的表示方法和计算相似度得分消耗资源。
3）如何在缺乏数据时给出好的推荐-称为冷启动【简单说：用户不会喜欢一个无效的物品，而用户不喜欢的物品又无效】

建议
1）在大型系统中，SVD分解(可以在程序调入时运行一次)每天运行一次或者其频率更低，并且还要离线运行。
2）在实际中，另一个普遍的做法就是离线计算并保存相似度得分。(物品相似度可能被用户重复的调用)
3）冷启动问题，解决方案就是将推荐看成是搜索问题，通过各种标签／属性特征进行基于内容的推荐。
"""
import numpy as np
def loadExData():
    # 原始矩阵
    return[[0, -1.6, 0.6],
           [0, 1.2, 0.8],
           [0, 0, 0],
           [0, 0, 0]]
def recommendData():
    # 利用SVD提高推荐效果，菜肴矩阵
    """
    行：代表人
    列：代表菜肴名词
    值：代表人对菜肴的评分，0表示未评分
    """
    return np.mat([[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
            [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]])

# 相似度计算，假定inA和inB 都是列向量
# 基于欧氏距离
def ecludSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))


# pearsSim()函数会检查是否存在3个或更多的点。
# corrcoef直接计算皮尔逊相关系数，范围[-1, 1]，归一化后[0, 1]
def pearsSim(inA, inB):
    # 如果不存在，该函数返回1.0，此时两个向量完全相关。
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


# 计算余弦相似度，如果夹角为90度，相似度为0；如果两个向量的方向相同，相似度为1.0
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5 + 0.5*(num/denom)

# 基于物品相似度的推荐引擎
"""
（1）求评论过物品i的用户集合
（2）求评论过物品j的用户集合
（3）求用户集合的交集，计算欧氏距离
（4）当新来用户评论过wupini，那么我们就可以用相似度预测该用户对物品j的评分
"""
def standEst(dataMat, user, simMeas, item):
    """
    Args:
        dataMat         训练数据集
        user            用户编号
        simMeas         相似度计算方法
        item            未评分的物品编号
    Returns:
        ratSimTotal/simTotal     评分（0～5之间的值）
    """
    # 得到数据集中的物品数目
    m,n = np.shape(dataMat)
    # 初始化两个评分值
    simTotal = 0.0
    ratSimTotal = 0.0

    # 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
    for j in range(n):
        userRating = dataMat[user, j]
        # 如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0:
            continue

        #找到item和j均不为0（有评分）的用户
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # 如果相似度为0，则两着没有任何重合元素，终止本次循环
        if len(overLap) == 0:
            similarity = 0
        else:
            # 如果存在重合的物品，则基于这些重合物重新计算相似度。
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])

        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity  用户相似度，   userRating 用户评分
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0

    # 通过除以所有的评分总和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal/simTotal

# 基于SVD的评分估计
# 在recommend() 中，这个函数用于替换对standEst()的调用，该函数对给定用户给定物品构建了一个评分估计值
def svdEst(dataMat, user, simMeas, item):
    """svdEst(计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分)

    Args:
        dataMat         训练数据集
        user            用户编号
        simMeas         相似度计算方法
        item            未评分的物品编号
    Returns:
        ratSimTotal/simTotal     评分（0～5之间的值）
    """
    # 物品数目
    n = np.shape(dataMat)[1]
    # 对数据集进行SVD分解
    simTotal = 0.0
    ratSimTotal = 0.0

    # 在SVD分解之后，我们只利用包含了90%能量值的奇异值，这些奇异值会以NumPy数组的形式得以保存
    U, Sigma, VT = np.linalg.svd(dataMat)

    # 如果要进行矩阵运算，就必须要用这些奇异值构建出一个对角矩阵
    Sig4 = np.mat(np.eye(4) * Sigma[: 4])

    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品(物品+4个主要的特征)
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    # 对于给定的用户，for循环在用户对应行的元素上进行遍历，
    # 这和standEst()函数中的for循环的目的一样，只不过这里的相似度计算时在低维空间下进行的。
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        # 相似度的计算方法也会作为一个参数传递给该函数
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        # for 循环中加入了一条print语句，以便了解相似度计算的进展情况。如果觉得累赘，可以去掉
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        # 对相似度不断累加求和
        simTotal += similarity
        # 对相似度及对应评分值的乘积求和
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 计算估计评分
        return ratSimTotal / simTotal

# N=3表示输出top3个推荐结果
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 每行一个用户，user用来表示用户所在的行数
    # .A表示将matrix的数组转换成ndarray
    # unratedItems所有没有评分的菜品的下标
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]

    # 如果不存在未评分物品，那么就退出函数
    if len(unratedItems) == 0:
        return 'you rated everything'

    # 物品的编号和评分值
    itemScores = []
    # 在未评分物品上进行循环
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 寻找前N个未评级物品，调用standEst()来产生该物品的预测得分，该物品的编号和估计值会放在一个元素列表itemScores中
        itemScores.append((item, estimatedScore))

    # 按照估计得分，对该列表进行排序并返回。列表逆排序，第一个值就是最大值
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]

# 图像压缩函数
# 加载并转换数据
def imgLoadData(filename):
    myl = []
    # 打开文本文件，并从文件以数组方式读入字符
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    # 矩阵调入后，就可以在屏幕上输出该矩阵
    myMat = np.mat(myl)
    return myMat


# 实现图像压缩，允许基于任意给定的奇异值数目来重构图像
def imgCompress(numSV=3, thresh=0.8):
    """imgCompress( )
    Args:
        numSV       Sigma长度
        thresh      判断的阈值
    """
    # 构建一个列表
    myMat = imgLoadData('./data/imageData.txt')

    # Sigma是一个对角矩阵，因此需要建立一个全0矩阵，然后将前面的那些奇异值填充到对角线上。
    U, Sigma, VT = np.linalg.svd(myMat)

    # 分析插入的 Sigma 长度
    analyse_data(Sigma, 20)

    SigRecon = np.mat(np.eye(numSV) * Sigma[: numSV])
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    return reconMat


def analyse_data(Sigma, loopNum=20):
    """analyse_data(分析 Sigma 的长度取值)
    Args:
        Sigma         Sigma的值
        loopNum       循环次数
    """
    # 总方差的集合（总能量值）
    Sig2 = Sigma ** 2
    SigmaSum = sum(Sig2)
    for i in range(loopNum):
        SigmaI = sum(Sig2[:i + 1])
        '''
        根据自己的业务情况，就行处理，设置对应的 Singma 次数
        通常保留矩阵 80% ～ 90% 的能量，就可以得到重要的特征并取出噪声。
        '''
        print('主成分：%s, 方差占比：%s%%' % (format(i + 1, '2.0f'), format(SigmaI / SigmaSum * 100, '4.2f')))

def recommendTest():
    # 计算相似度的方法
    myMat = recommendData()
    print recommend(myMat, 1)
    print recommend(myMat, 1, estMethod=svdEst)
    print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

def sklearnSVD():
    # 使用sklearn对矩阵进行SVD分解
    Data = loadExData()
    print('Data:', Data)
    U, Sigma, VT = np.linalg.svd(Data)

    print('U:', U)
    print('Sigma', Sigma)
    print('VT:', VT)

    # 重构一个3x3的矩阵Sig3
    Sig3 = np.mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    print(U[:, :3] * Sig3 * VT[:3, :])