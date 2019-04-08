# 简述
&emsp;&emsp;在scikit-learn中，一共有3个朴素贝叶斯的分类算法类。分别是GaussianNB，MultinomialNB和BernoulliNB。其中GaussianNB就是先验为高斯分布的朴素贝叶斯，MultinomialNB就是先验为多项式分布的朴素贝叶斯，而BernoulliNB就是先验为伯努利分布的朴素贝叶斯。    
&emsp;&emsp;这三个类适用的分类场景各不相同，一般来说，如果样本特征的分布大部分是连续值，使用GaussianNB会比较好。如果如果样本特征的分大部分是多元离散值，使用MultinomialNB比较合适。而如果样本特征是二元离散值或者很稀疏的多元离散值，应该使用BernoulliNB。

# GaussianNB类
&emsp;&emsp;GaussianNB类的主要参数仅有先验概率priors一个，对应各个类别的先验概率P_k。这个值默认不给出，如果不给出此时P_k=m_k/m。其中m为训练集样本总数量，m_k为输出为第k类别的训练集样本数。如果给出的话就以priors 为准。   
&emsp;&emsp;在使用GaussianNB的fit方法拟合数据后，我们可以进行预测。此时预测有三种方法，包括predict，predict_log_proba和predict_proba。   
&emsp;&emsp;predict方法就是我们最常用的预测方法，直接给出测试集的预测类别输出。   
&emsp;&emsp;predict_proba则不同，它会给出测试集样本在各个类别上预测的概率。容易理解，predict_proba预测出的各个类别概率里的最大值对应的类别，也就是predict方法得到类别。
&emsp;&emsp;predict_log_proba和predict_proba类似，它会给出测试集样本在各个类别上预测的概率的一个对数转化。转化后predict_log_proba预测出的各个类别对数概率里的最大值对应的类别，也就是predict方法得到类别。   


# MultinomialNB类
&emsp;&emsp;MultinomialNB参数一共有3个。其中，参数alpha没有特别的需要，用默认的1即可。如果发现拟合的不好，需要调优时，可以选择稍大于1或者稍小于1的数。布尔参数fit_prior表示是否要考虑先验概率，如果是false,则所有的样本类别输出都有相同的类别先验概率。否则可以自己用第三个参数class_prior输入先验概率，或者不输入第三个参数class_prior让MultinomialNB自己从训练集样本来计算先验概率，此时的先验概率为P_k=m_k/m   
&emsp;&emsp;在使用MultinomialNB的fit方法或者partial_fit方法拟合数据后，我们可以进行预测。此时预测有三种方法，包括predict，predict_log_proba和predict_proba。和GaussianNB完全一样。　


# BernoulliNB类
&emsp;&emsp;BernoulliNB一共有4个参数，其中3个参数的名字和意义和MultinomialNB完全相同。唯一增加的一个参数是binarize。这个参数主要是用来帮BernoulliNB处理二项分布的，可以是数值或者不输入。如果不输入，则BernoulliNB认为每个数据特征都已经是二元的。否则的话，小于binarize的会归为一类，大于binarize的会归为另外一类。
&emsp;&emsp;在使用BernoulliNB的fit或者partial_fit方法拟合数据后，我们可以进行预测。此时预测有三种方法，包括predict，predict_log_proba和predict_proba。由和GaussianNB完全一样