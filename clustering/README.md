# 实现了
## 1、kMeans方法
&emsp;&emsp;1）n_clusters: 即我们的k值，一般需要多试一些值以获得较好的聚类效果。    
&emsp;&emsp;2）max_iter： 最大的迭代次数，一般如果是凸数据集的话可以不管这个值，如果数据集不是凸的，可能很难收敛，此时可以指定最大的迭代次数让算法可以及时退出循环。   
&emsp;&emsp;3）n_init：用不同的初始化质心运行算法的次数。由于K-Means是结果受初始值影响的局部最优的迭代算法，因此需要多跑几次以选择一个较好的聚类效果，默认是10，一般不需要改。如果你的k值较大，则可以适当增大这个值。   
&emsp;&emsp;4）init： 即初始值选择的方式，可以为完全随机选择'random',优化过的'k-means++'或者自己指定初始化的k个质心。一般建议使用默认的'k-means++'。  
&emsp;&emsp;5）algorithm：有auto, full or elkan三种选择。"full"就是我们传统的K-Means算法， “elkan”是我们原理篇讲的elkan K-Means算法。默认的"auto"则会根据数据值是否是稀疏的，来决定如何选择"full"和“elkan”。一般数据是稠密的，那么就是 “elkan”，否则就是"full"。一般来说建议直接用默认的"auto"

## 2、MiniBatchKMeans参数
&emsp;&emsp;1) n_clusters: 即我们的k值，和KMeans类的n_clusters意义一样。   
&emsp;&emsp;2）max_iter：最大的迭代次数， 和KMeans类的max_iter意义一样。   
&emsp;&emsp;3）n_init：用不同的初始化质心运行算法的次数。这里和KMeans类意义稍有不同，KMeans类里的n_init是用同样的训练集数据来跑不同的初始化质心从而运行算法。而MiniBatchKMeans类的n_init则是每次用不一样的采样数据集来跑不同的初始化质心运行算法。   
&emsp;&emsp;4）batch_size：即用来跑Mini Batch KMeans算法的采样集的大小，默认是100.如果发现数据集的类别较多或者噪音点较多，需要增加这个值以达到较好的聚类效果。   
&emsp;&emsp;5）init： 即初始值选择的方式，和KMeans类的init意义一样。   
&emsp;&emsp;6）init_size: 用来做质心初始值候选的样本个数，默认是batch_size的3倍，一般用默认值就可以了。    
&emsp;&emsp;7）reassignment_ratio: 某个类别质心被重新赋值的最大次数比例，这个和max_iter一样是为了控制算法运行时间的。这个比例是占样本总数的比例，乘以样本总数就得到了每个类别质心可以重新赋值的次数。如果取值较高的话算法收敛时间可能会增加，尤其是那些暂时拥有样本数较少的质心。默认是0.01。如果数据量不是超大的话，比如1w以下，建议使用默认值。如果数据量超过1w，类别又比较多，可能需要适当减少这个比例值。具体要根据训练集来决定。    
&emsp;&emsp;8）max_no_improvement：即连续多少个Mini Batch没有改善聚类效果的话，就停止算法， 和reassignment_ratio， max_iter一样是为了控制算法运行时间的。默认是10.一般用默认值就足够了。
