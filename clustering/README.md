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



## 3、DBSCAN参数
&emsp;&emsp;1）eps： DBSCAN算法参数，即我们的`$\epsilon -$`邻域的距离阈值。默认值是0.5.一般需要通过在多组值里面选择一个合适的阈值。eps过大，则更多的点会落在核心对象的`$\epsilon -$`邻域，此时我们的类别数可能会减少， 本来不应该是一类的样本也会被划为一类。反之则类别数可能会增大，本来是一类的样本却被划分开。   
&emsp;&emsp;2）min_samples： DBSCAN算法参数，即样本点要成为核心对象所需要的`$\epsilon -$`邻域的样本数阈值。默认值是5. 一般需要通过在多组值里面选择一个合适的阈值。通常和eps一起调参。在eps一定的情况下，min_samples过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多。反之min_samples过小的话，则会产生大量的核心对象，可能会导致类别数过少。    
&emsp;&emsp;3）metric：最近邻距离度量参数。可以使用的距离度量较多，一般来说DBSCAN使用默认的欧式距离（即p=2的闵可夫斯基距离）。   
&emsp;&emsp;4）algorithm：最近邻搜索算法参数，算法一共有三种，第一种是蛮力实现，第二种是KD树实现，第三种是球树实现。这三种方法在K近邻法(KNN)原理小结中都有讲述，如果不熟悉可以去复习下。对于这个参数，一共有4种可选输入，‘brute’对应第一种蛮力实现，‘kd_tree’对应第二种KD树实现，‘ball_tree’对应第三种的球树实现， ‘auto’则会在上面三种算法中做权衡，选择一个拟合最好的最优算法。需要注意的是，如果输入样本特征是稀疏的时候，无论我们选择哪种算法，最后scikit-learn都会去用蛮力实现‘brute’。个人的经验，一般情况使用默认的 ‘auto’就够了。 如果数据量很大或者特征也很多，用"auto"建树时间可能会很长，效率不高，建议选择KD树实现‘kd_tree’，此时如果发现‘kd_tree’速度比较慢或者已经知道样本分布不是很均匀时，可以尝试用‘ball_tree’。而如果输入样本是稀疏的，无论你选择哪个算法最后实际运行的都是‘brute’。    
&emsp;&emsp;5）leaf_size：最近邻搜索算法参数，为使用KD树或者球树时， 停止建子树的叶子节点数量的阈值。这个值越小，则生成的KD树或者球树就越大，层数越深，建树时间越长，反之，则生成的KD树或者球树会小，层数较浅，建树时间较短。默认是30. 因为这个值一般只影响算法的运行速度和使用内存大小，因此一般情况下可以不管它。   
&emsp;&emsp;6） p: 最近邻距离度量参数。只用于闵可夫斯基距离和带权重闵可夫斯基距离中p值的选择，p=1为曼哈顿距离， p=2为欧式距离。如果使用默认的欧式距离不需要管这个参数。

## 4、BIRCH参数
&emsp;&emsp;1) threshold:即叶节点每个CF的最大样本半径阈值T，它决定了每个CF里所有样本形成的超球体的半径阈值。一般来说threshold越小，则CF Tree的建立阶段的规模会越大，即BIRCH算法第一阶段所花的时间和内存会越多。但是选择多大以达到聚类效果则需要通过调参决定。默认值是0.5.如果样本的方差较大，则一般需要增大这个默认值。      
&emsp;&emsp;2) branching_factor：即CF Tree内部节点的最大CF数B，以及叶子节点的最大CF数L。这里scikit-learn对这两个参数进行了统一取值。也就是说，branching_factor决定了CF Tree里所有节点的最大CF数。默认是50。如果样本量非常大，比如大于10万，则一般需要增大这个默认值。选择多大的branching_factor以达到聚类效果则需要通过和threshold一起调参决定   
&emsp;&emsp;3）n_clusters：即类别数K，在BIRCH算法是可选的，如果类别数非常多，我们也没有先验知识，则一般输入None，此时BIRCH算法第4阶段不会运行。但是如果我们有类别的先验知识，则推荐输入这个可选的类别值。默认是3，即最终聚为3类。   
&emsp;&emsp;4）compute_labels：布尔值，表示是否标示类别输出，默认是True。一般使用默认值挺好，这样可以看到聚类效果。

## 5、SpectralClustering参数
&emsp;&emsp;在scikit-learn的类库中，sklearn.cluster.SpectralClustering实现了基于Ncut的谱聚类，没有实现基于RatioCut的切图聚类。同时，对于相似矩阵的建立，也只是实现了基于K邻近法和全连接法的方式，没有基于ϵ-邻近法的相似矩阵。最后一步的聚类方法则提供了两种，K-Means算法和 discretize算法。    
&emsp;&emsp;对于SpectralClustering的参数，我们主要需要调参的是相似矩阵建立相关的参数和聚类类别数目，它对聚类的结果有很大的影响。当然其他的一些参数也需要理解，在必要时需要修改默认参数。    
SpectralClustering重要参数与调参注意事项   
&emsp;&emsp;1）n_clusters：代表我们在对谱聚类切图时降维到的维数（原理篇第7节的k1），同时也是最后一步聚类算法聚类到的维数(原理篇第7节的k2)。也就是说scikit-learn中的谱聚类对这两个参数统一到了一起。简化了调参的参数个数。虽然这个值是可选的，但是一般还是推荐调参选择最优参数。
&emsp;&emsp;2) affinity: 也就是我们的相似矩阵的建立方式。可以选择的方式有三类，第一类是 'nearest_neighbors'即K邻近法。第二类是'precomputed'即自定义相似矩阵。选择自定义相似矩阵时，需要自己调用set_params来自己设置相似矩阵。第三类是全连接法，可以使用各种核函数来定义相似矩阵，还可以自定义核函数。最常用的是内置高斯核函数'rbf'。其他比较流行的核函数有‘linear’即线性核函数, ‘poly’即多项式核函数, ‘sigmoid’即sigmoid核函数。如果选择了这些核函数， 对应的核函数参数在后面有单独的参数需要调。自定义核函数我没有使用过，这里就不多讲了。affinity默认是高斯核'rbf'。一般来说，相似矩阵推荐使用默认的高斯核函数。    
&emsp;&emsp;3) 核函数参数gamma: 如果我们在affinity参数使用了多项式核函数 'poly'，高斯核函数‘rbf’, 或者'sigmoid'核函数，那么我们就需要对这个参数进行调参。   
多项式核函数中这个参数对应K(x,z)=（γx∙z+r)d中的γ。一般需要通过交叉验证选择一组合适的γ,r,d    
高斯核函数中这个参数对应K(x,z)=exp(−γ||x−z||2)中的γ。一般需要通过交叉验证选择合适的γ    
sigmoid核函数中这个参数对应K(x,z)=tanh（γx∙z+r)中的γ。一般需要通过交叉验证选择一组合适的γ,r    
γ默认值为1.0，如果我们affinity使用'nearest_neighbors'或者是'precomputed'，则这么参数无意义。   
&emsp;&emsp;4）核函数参数degree：如果我们在affinity参数使用了多项式核函数 'poly'，那么我们就需要对这个参数进行调参。这个参数对应K(x,z)=（γx∙z+r)d中的d。默认是3。一般需要通过交叉验证选择一组合适的γ,r,d    
&emsp;&emsp;5）核函数参数coef0: 如果我们在affinity参数使用了多项式核函数 'poly'，或者sigmoid核函数，那么我们就需要对这个参数进行调参。    
多项式核函数中这个参数对应K(x,z)=（γx∙z+r)d中的r。一般需要通过交叉验证选择一组合适的γ,r,d   
sigmoid核函数中这个参数对应K(x,z)=tanh（γx∙z+r)中的r。一般需要通过交叉验证选择一组合适的γ,r   
coef0默认为1.   
&emsp;&emsp;6）kernel_params：如果affinity参数使用了自定义的核函数，则需要通过这个参数传入核函数的参数。   
&emsp;&emsp;7 )n_neighbors: 如果我们affinity参数指定为'nearest_neighbors'即K邻近法，则我们可以通过这个参数指定KNN算法的K的个数。默认是10.我们需要根据样本的分布对这个参数进行调参。如果我们affinity不使用'nearest_neighbors'，则无需理会这个参数。   
&emsp;&emsp;8）eigen_solver:1在降维计算特征值特征向量的时候，使用的工具。有 None, ‘arpack’, ‘lobpcg’, 和‘amg’4种选择。如果我们的样本数不是特别大，无需理会这个参数，使用''None暴力矩阵特征分解即可,如果样本量太大，则需要使用后面的一些矩阵工具来加速矩阵特征分解。它对算法的聚类效果无影响。    
&emsp;&emsp;9）eigen_tol：如果eigen_solver使用了arpack’，则需要通过eigen_tol指定矩阵分解停止条件。    
&emsp;&emsp;10）assign_labels：即最后的聚类方法的选择，有K-Means算法和 discretize算法两种算法可以选择。一般来说，默认的K-Means算法聚类效果更好。但是由于K-Means算法结果受初始值选择的影响，可能每次都不同，如果我们需要算法结果可以重现，则可以使用discretize。    
&emsp;&emsp;11）n_init：即使用K-Means时用不同的初始值组合跑K-Means聚类的次数，这个和K-Means类里面n_init的意义完全相同，默认是10，一般使用默认值就可以。如果你的n_clusters值较大，则可以适当增大这个值。
