## 机器学习算法的随机数据生成
#### 1、numpy随机数据生成API
&emsp;&emsp;1) rand(d0, d1, ..., dn) 用来生成d0xd1x...dn维的数组。数组的值在[0,1)之间   
例如：np.random.rand(3,2,2)，输出如下3x2x2的数组   
&emsp;&emsp;2) randn((d0, d1, ..., dn), 也是用来生成d0xd1x...dn维的数组。不过数组的值服从N(0,1)的标准正态分布。   
例如：np.random.randn(3,2)，输出如下3x2的数组，这些值是N(0,1)的抽样数据。   
&emsp;&emsp;如果需要服从N(μ,σ2)的正态分布，只需要在randn上每个生成的值x上做变换σx+μ即可，例如：   
例如：2*np.random.randn(3,2) + 1，输出如下3x2的数组，这些值是N(1,4)的抽样数据。
&emsp;&emsp;3)randint(low[, high, size])，生成随机的大小为size的数据，size可以为整数，为矩阵维数，或者张量的维数。值位于半开区间 [low, high)   
例如：np.random.randint(3, size=[2,3,4])返回维数维2x3x4的数据。取值范围为最大值为3的整数。
&emsp;&emsp;4) random_integers(low[, high, size]),和上面的randint类似，区别在与取值范围是闭区间[low, high]。   
&emsp;&emsp;5) random_sample([size]), 返回随机的浮点数，在半开区间 [0.0, 1.0)。如果是其他区间[a,b),可以加以转换(b - a) * random_sample([size]) + a

#### 2、scikit-learn随机数据生成API介绍
&emsp;&emsp;1) 用make_regression 生成回归模型的数据   
&emsp;&emsp;2) 用make_hastie_10_2，make_classification或者make_multilabel_classification生成分类模型数据   
&emsp;&emsp;3) 用make_blobs生成聚类模型数据   
&emsp;&emsp;4) 用make_gaussian_quantiles生成分组多维正态分布的数据   

#### 3、scikit-learn随机数据生成实例
##### 3.1、回归模型随机数据
&emsp;&emsp;这里我们使用make_regression生成回归模型数据。几个关键参数有n_samples（生成样本数），n_features（样本特征数），noise（样本随机噪音）和coef（是否返回回归系数）。例子代码如下：   
```
X, y, coef =make_regression(n_samples=1000, n_features=1,noise=10, coef=True)
```


##### 3.2、分类模型随机数据
&emsp;&emsp;这里我们用make_classification生成三元分类模型数据。几个关键参数有n_samples（生成样本数），n_features（样本特征数），n_redundant（冗余特征数）和n_classes（输出的类别数），例子代码如下：   
```
X1, Y1 = make_classification(n_samples=400, n_features=2, n_redundant=0,n_clusters_per_class=1, n_classes=3)
```

#### 3.3、聚类模型随机数据
&emsp;&emsp;这里我们用make_blobs生成聚类模型数据。几个关键参数有n_samples（生成样本数），n_features（样本特征数），centers(簇中心的个数或者自定义的簇中心)和cluster_std（簇数据方差，代表簇的聚合程度）。例子如下：
```
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [1,1], [2,2]], cluster_std=[0.4, 0.5, 0.2])
```

#### 3.4、分组正态分布混合数据
&emsp;&emsp;我们用make_gaussian_quantiles生成分组多维正态分布的数据。几个关键参数有n_samples（生成样本数）， n_features（正态分布的维数），mean（特征均值）， cov（样本协方差的系数）， n_classes（数据在正态分布中按分位数分配的组数）。 例子如下：
```
X1, Y1 = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, mean=[1,2],cov=2)
```
