# 实现了
## 1、Stacking方法
&emsp;&emsp;代码入口：ID3.xiguaTest()     
&emsp;&emsp;还有其他一些数据集的实验,眼镜数据集：ID3.glassTest()，蘑菇数据集：ID3.mushroomTest

## 2、AdaBoost方法
&emsp;&emsp;scikit-learn中Adaboost类库主要有两个，分别是AdaBoostClassifier和AdaBoostRegressor。AdaBoostClassifier用于分类，AdaBoostRegressor用于回归。    
&emsp;&emsp;AdaBoostClassifier使用了两种Adaboost分类算法的实现，SAMME和SAMME.R。而AdaBoostRegressor则使用了我们原理篇里讲到的Adaboost回归算法的实现，即Adaboost.R2。 当我们对Adaboost调参时，主要要对两部分内容进行调参，第一部分是对我们的Adaboost的框架进行调参， 第二部分是对我们选择的弱分类器进行调参。两者相辅相成。下面就对Adaboost的两个类：AdaBoostClassifier和AdaBoostRegressor从这两部分做一个介绍。
