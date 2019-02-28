# 实现了
## 1、Stacking方法
&emsp;&emsp;代码入口：ID3.xiguaTest()     
&emsp;&emsp;还有其他一些数据集的实验,眼镜数据集：ID3.glassTest()，蘑菇数据集：ID3.mushroomTest

## 2、AdaBoost方法
&emsp;&emsp;scikit-learn中Adaboost类库主要有两个，分别是AdaBoostClassifier和AdaBoostRegressor。AdaBoostClassifier用于分类，AdaBoostRegressor用于回归。    
#### 2.1 框架调参
&emsp;&emsp; 1）base_estimator：AdaBoostClassifier和AdaBoostRegressor都有。理论上可以选择任何一个分类或者回归学习器，不过需要支持样本权重。我们常用的是CART决策树或者神经网络MLP。AdaBoostClassifier默认使用CART分类树，而AdaBoostRegressor默认使用CART回归树。另外有一个要注意的点是，如果我们选择的AdaBoostClassifier算法是SAMME.R，则我们的弱分类学习器还需要支持概率预测，也就是在scikit-learn中弱分类学习器对应的预测方法除了predict还需要有predict_proba。    
emsp;&emsp; 2）algorithm：这个参数只有AdaBoostClassifier有。主要原因是scikit-learn实现了两种Adaboost分类算法，SAMME和SAMME.R。两者的主要区别是弱学习器权重的度量，SAMME使用了和我们的原理篇里二元分类Adaboost算法的扩展，即用对样本集分类效果作为弱学习器权重，而SAMME.R使用了对样本集分类的预测概率大小来作为弱学习器权重。由于SAMME.R使用了概率度量的连续值，迭代一般比SAMME快，因此AdaBoostClassifier的默认算法algorithm的值也是SAMME.R。我们一般使用默认的SAMME.R就够了，但是要注意的是使用了SAMME.R， 则弱分类学习器参数base_estimator必须限制使用支持概率预测的分类器。SAMME算法则没有这个限制。 
emsp;&emsp; 3）loss：这个参数只有AdaBoostRegressor有，Adaboost.R2算法需要用到。有线性‘linear’, 平方‘square’和指数 ‘exponential’三种选择, 默认是线性，一般使用线性就足够了，除非你怀疑这个参数导致拟合程度不好。这个值对应了我们对第k个弱分类器的中第i个样本的误差的处理.   
emsp;&emsp;4) n_estimators： AdaBoostClassifier和AdaBoostRegressor都有，就是我们的弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是50。在实际调参的过程中，我们常常将n_estimators和下面介绍的参数learning_rate一起考虑。    
emsp;&emsp;5) learning_rate: AdaBoostClassifier和AdaBoostRegressor都有，即每个弱学习器的权重缩减系数ν。ν的取值范围为0<ν≤1。对于同样的训练集拟合效果，较小的ν意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的ν开始调参，默认是1。
