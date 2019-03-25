# 1、scikit-learn SVM算法库使用概述
scikit-learn中SVM的算法库分为两类,相关的类都包裹在sklearn.svm模块之中:     
&emsp;&emsp;一类是分类算法库，包括SVC， NuSVC，和LinearSVC    
&emsp;&emsp;一类是回归算法库，包括SVR， NuSVR，和LinearSVR    
&emsp;&emsp;对于SVC， NuSVC，和LinearSVC 3个分类的类，SVC和 NuSVC差不多，区别仅仅在于对损失的度量方式不同，而LinearSVC从名字就可以看出，他是线性分类，也就是不支持各种低维到高维的核函数，仅仅支持线性核函数，对线性不可分的数据不能使用。    
&emsp;&emsp;同样的，对于SVR， NuSVR，和LinearSVR 3个回归的类， SVR和NuSVR差不多，区别也仅仅在于对损失的度量方式不同。LinearSVR是线性回归，只能使用线性核函数。   
&emsp;&emsp;我们使用这些类的时候，如果有经验知道数据是线性可以拟合的，那么使用LinearSVC去分类 或者LinearSVR去回归，它们不需要我们去慢慢的调参去选择各种核函数以及对应参数， 速度也快。如果我们对数据分布没有什么经验，一般使用SVC去分类或者SVR去回归，这就需要我们选择核函数以及对核函数调参了。   
&emsp;&emsp;什么特殊场景需要使用NuSVC分类 和 NuSVR 回归呢？如果我们对训练集训练的错误率或者说支持向量的百分比有要求的时候，可以选择NuSVC分类 和 NuSVR 。它们有一个参数来控制这个百分比。

# 2、SVM分类算法库调参小结
https://www.cnblogs.com/pinard/p/6117515.html
