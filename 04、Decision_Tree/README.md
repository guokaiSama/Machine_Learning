# 实现了
## 1、经典的决策树ID3
&emsp;&emsp;代码入口：ID3.xiguaTest()     
&emsp;&emsp;还有其他一些数据集的实验,眼镜数据集：ID3.glassTest()，蘑菇数据集：ID3.mushroomTest

## 2、基于C4.5决策树
&emsp;&emsp;代码入口：C45.xiguaTest()   

## 3、基于Gini系数的决策树
&emsp;&emsp;待实现  

## 4、sklearn的决策树接口
&emsp;&emsp;代码入口：SKC.classifierTree()     
&emsp;&emsp;该接口底层是基于Gini系数实现的

## 5、回归树(Gini)接口
&emsp;&emsp;代码入口：createTree(trainMat, ops=(1, 20))    

## 6、模型树接口
&emsp;&emsp;代码入口：createTree(trainMat, utils.modelLeaf, utils.modelErr, ops=(1, 20))

## 7、回归树sklearn接口
&emsp;&emsp;代码入口：SKR.regressorTree()       
&emsp;&emsp;该接口底层是基于Gini系数实现的

# 决策树调参
&emsp;&emsp;[参见文档](./sklearn调参.md)