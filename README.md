# FunnyPyML
This is python machine learning project, just for fun and deeper understanding. There are some mechine learning algorithms(SVM, LR, GBDT...) and 8 optimizers(SGD, LBFGS, Adam...) .Again,  I write it just for deeper my understanding so that I dont guarantee that the code is right. But I think it will be helpful to someone like me. The following part will explain all files(what is and how to use) in this project. 


这是一个用python写的机器学习小项目，仅仅是好玩和为了加深我的理解。主要实现了一些常见的机器学习算法（比如支持向量机，逻辑回归，梯度提升树）和8个迭代优化算法（随机梯度下降，LBFGS，Adam）。有必要强调的是，这些代码仅仅是为了加深理解写的，并不保证一定正确，但是应该还是对一些刚刚学习机器学习的同学有帮助。 后面会介绍这个项目中的所有文件，包括它是什么和怎么用。


###所有文件夹


**base** : 一些基本操作的python模块

**dataset** : 几个数据集

**learner** : 一些分类、聚类、回归算法

**loss** : 几个损失函数

**material** : 写算法时的参照，不是所有算法都有

**optimizer** : 8个迭代优化算法

**preprocessor** : 数据与特征预处理算法，只有一个pca

**util** : 别的算法实现时需要用到的一些小工具或者算法


###主要的功能文件

####base/_logging.py
仅仅是用来生成一个日志对象。


####base/common_function.py
一些公共使用的函数，比如计算sigmoid，欧式距离，条件熵等。


####base/dataset.py
数据集对象类，包含了一个切分数据集为训练和测试的方法。其中categorical是存储数据集每个特征是数值型还是离散型，原本是用来做数据集检查用的，因为不同算法要求输入的数据集不一样，有些必须是数值型特征，有些必须是离散型，不过等自己实现了一些以后才想起来所有都忘了检查。。所以这个变量其实暂时没用到。


####base/dataloader.py
用于加载数据的模块，只可以加载arff和csv数据集，需要指定数据集的目标列名，如果不指定或者指定的列名不在数据集中，则取最后一列作为目标列。
```
path = 'dataset/iris.arff'
loader = DataLoader(path)
dataset = loader.load(target_col_name='binaryClass')
trainset, testset = dataset.cross_split(ratio=0.1, shuffle=True)
```

####base/metric.py
评价准则的模块，包含准确率（分类），平均误差（回归），F-measure（聚类），所有函数都包含2个参数，第一个真实输出，第二个预测输出。


####base/time_scheduler.py
用于测试程序运行时间的模块，对比不同算法或者算法的不同参数的运行时间时可能会用到。
```
lr = LogisticRegression()
scheduler = TimeScheduler()
scheduler.tic_tac(task_name='test1', func=lr.fit, X=trainset[0], y=trainset[1])  #后面的参数X和y是lr.fit的参数
scheduler.print_task_schedule(task_name='test1')

# 和上面的方式作用一样
scheduler.start_task(task_name='test2')
lr.fit(X=trainset[0], y=trainset[1])
scheduler.stop_task(task_name='test2')
scheduler.print_task_schedule(task_name='test2')
```


####learner/abstract_learner.py
仅仅是定义了一些抽象类


####learner/logistic_regression.py
逻辑回归，除了一些最大迭代次数和批大小等基础参数，还可以设置是否需要打印损失图。
```
lr = LogisticRegression(max_iter=1000, batch_size=100, learning_rate=0.01, is_plot_loss=True)
lr.fit(trainset[0], trainset[1])
predict = lr.predict(testset[0])
acc = accuracy_score(testset[1], predict)
print 'test accuracy:', acc
```


####learner/linear_regressor.py
线性回归，和上面的逻辑回归基本一样，只不过一个分类算法，一个回归算法


####learner/naive_bayes.py
朴素贝叶斯，使用方法和上面一样


####learner/kmeans.py
Kmeans聚类，当特征数为2时，可以打印出聚类图。
```
kmeans = Kmeans(k=2, is_plot=True)
kmeans.fit(trainset[0])   # 参数只有一个X， 不需要传入y
prediction = kmeans.predict(testset[0])
performance = cluster_f_measure(testset[1], prediction)
print 'F-measure:', performance
```

####learner/decision_tree.py
决策树，里面包含了分类树（ID3算法）和回归树（CART），都有剪枝的过程，但是不知道是我实现的有问题还是算法就不对，剪枝以后效果反而更差。剪枝目前用的方式是划分一个剪枝集，然后从下往上开始对比，到底是剪了好还是不剪好，使用剪枝集来判断，留下好的那种情况。分类树中可以打印出决策树图，打印的代码是使用机器学习实战里的代码，但是不建议打印，因为太乱，实在想看的可以试试。分类树中预测概率是通过最终叶子节点中包含各个类别的比例实现的。
```
dtc = DecisionTreeClassifier(min_split=1, is_prune=False)
dtc.fit(trainset[0], trainset[1])
predict = dtc.predict(testset[0])
performance = accuracy_score(testset[1], predict)
print 'test accuracy:', performance
dt.createPlot()   # 很乱，不要看

dtr = DecisionTreeRegressor(is_prune=False)
dtr.fit(trainset[0], trainset[1])
predict = dtr.predict(testset[0])
performance = mean_error(testset[1], predict)
print 'mean error:', performance
```


####learner/random_forest.py
随机森林，树集成算法，主要过程就是行列采样，并且让决策树完全分裂，行是有放回采样，列是无放回采样，最后让所有树进行投票（预测的类别概率加和），但是算法的效果居然还不如一颗决策树，要是大侠知道了是哪块错了请务必告诉我。
```
rf = RandomForest(k=100, column_sample_ratio=0.9)
rf.fit(trainset[0], trainset[1])
predict = rf.predict(testset[0])
print 'RandomForest accuracy', accuracy_score(testset[1], predict)
```

####learner/gbdt.py
梯度提升树，同样也是树集成算法，不过上面那个实现的是分类，这个是回归，主要思想是后一颗树预测的目标是前一颗树的残差，预测时将所有树的预测结果加和，整体像一条流水线，是纵向的，而随机森林是横向。GBDT效果确实很好，把树的数量设置高一些，拟合能力比一棵树简直强了太多（可能过拟合）。
```
gbdt = GradientBoostingDecisionTree(k=10)    # 由于实现的cart回归树的效率不高，导致这里树的个数也不敢设置太高，不然时间太久
gbdt.fit(trainset[0], trainset[1])
predict = gbdt.predict(testset[0])
print 'GBDT mean error:', mean_error(testset[1], predict)
```


####learner/adaboost.py
提升（不知怎么翻译），是集成算法，不过基分类器可以自己指定，当基分类器比较弱时，提升很明显。和gbdt一样是基于前向分步算法，对每一个样本都有一个权重（会根据训练结果动态调整），根据权重抽样，并且每一个基分类器也有一个权重（预测效果好的权重高）。
```
base_learner = NaiveBayes()
ada = AdaBoost(base_learner=base_learner, k=100)
ada.fit(trainset[0], trainset[1])
predict = ada.predict(testset[0])
performance = accuracy_score(testset[1], prediction)
print 'AdaBoost accuracy:', performance
```

####learner/svm.py
支持向量机，只支持二分类，可以调整核函数，有3种核函数，一种linear（其实就是没有），一种poly（多项式核，用p控制多项式的次数），一种rbf（高斯核，将特征映射到无穷维空间，sigma设置很小时，有非常强的拟合能力，注意过拟合）。当特征数为2时，可以打印出分类的决策边界，这个值得打印出来看一下。算法的迭代优化是使用smo，每次挑选2个拉格朗日乘子更新，一个违反约束，另一个使间隔最大（这样一次更新下降的才最快）。
```
svm = SVM(kernel_type='rbf', sigma=0.3)
svm.fit(trainset[0], trainset[1])
predict = svm.predict(testset[0])
print 'test accuracy:', accuracy_score(testset[1], predict)
svm.plot(testset[0])   # 特征数一定要为2，否则无法打印

svm = SVM(kernel_type='poly', p=2)
svm.fit(trainset[0], trainset[1])
predict = svm.predict(testset[0])
print 'test accuracy:', accuracy_score(testset[1], predict)
svm.plot(testset[0]) 
```

####learner/perceptron.py
感知机，只支持二分类。使用梯度下降更新参数，但是值得注意的是，一次更新一个误分类点，所以必须将随机梯度下降的批次设置为1。同样可以打印出分类的决策边界，可以和svm的边界对比，感知机的边界是凑活能用型，svm边界要求间隔最大（个人感觉，svm边界是两边的支持向量取中点，这样是不是有点草率，应该还和支持向量周围的点的密度有关，仅感勿喷）。
```
model = Perceptron(max_iter=200, learning_rate=0.01, is_plot_loss=True)
model.fit(trainset[0], trainset[1])
predict = model.predict(testset[0])
acc = accuracy_score(testset[1], predict)
print 'test accuracy:', acc
model.plot(testset[0])     # 同样feature数必须是2
```

####learner/knn.py
k近邻，包含了分类和回归，方法都是一样。实现了2种，暴力搜和KD树搜索。KD树之前读统计学习方法时一直不理解如何构建超球体，然后判断待搜索区域是否和超球体相交，这样实现起来不是超级麻烦，后来看了别人的实现，发现用一个小技巧就能实现判断。
```
knn = KNNClassifier(search_mode='kd_tree')
knn.fit(trainset[0], trainset[1])
predict = knn.predict(testset[0])
print accuracy_score(testset[1], predict)
```


####preprocessor/pca.py
主成分分析，用于数据去相关或降维的算法。支持白化（使方差为1），可以将变换后的矩阵打印出来。
```
pca = PCA(2, whiten=False)
pca.fit(trainset[0])   # feature数必须为2
pca.plot(trainset[0])
```

####optimizer/sgd.py
随机梯度下降，支持每次迭代随机打乱，支持学习率衰减（3种策略，step，exponential，anneal），支持加入梯度噪声（高斯噪声）。迭代函数需要传入一个计算损失和梯度的函数feval，以及数据集X和y，还有相对应的参数。
```
optimizer = StochasticGradientDescent(learning_rate=self._learning_rate, batch_size=self._batch_size,
                                        decay_strategy='anneal', max_iter=self._max_iter, is_plot_loss=True,
                                        add_gradient_noise=True)
parameter = optimizer.optim(feval=self.feval, X=_X, y=_y, parameter=parameter)
```

####optimizer/cg.py
共轭梯度法，每次梯度下降的方向都与之前的正交（施密特正交化），学习步长使用非精确一维搜索确定。
```
optimizer = ConjugateGradientDescent()
```


####optimizer/lbfgs.py
限存拟牛顿法（Limited-memory BFGS），仅仅保存最近k次方向和步长，步长使用非精确一维搜索确定，这里和cg不一样的是，一维搜索时，不需要那么迭代那么多次，几次就够了，它的收敛速度不是非常依赖于学习率。
```
optimizer = LBFGS()
```

####optimizer/momentum.py
动量梯度下降，这个包括后面几个都是随机梯度下降派别的算法。加入动量的概念，梯度变化更加平滑，更容易冲破局部极小值
```
optimizer = MomentumSGD(learning_rate=self._learning_rate, batch_size=self._batch_size, momentum=0.9,
                        momentum_type='nesterov', max_iter=self._max_iter, is_plot_loss=True,
                        add_gradient_noise=True)
```


####optimizer/adagrad.py
adagrad，这个相对于sgd，学习率随着历史的梯度变化量逐渐减小（这也是缺点。。），并且不同的参数的学习率衰减程度不一样。
```
optimizer = Adagrad(learning_rate=self._learning_rate, batch_size=self._batch_size, max_iter=self._max_iter,
                    is_plot_loss=True, add_gradient_noise=True)
```


####optimizer/adadelta.py
adadelta, 对adagrad的改进，不是对历史所有梯度变化量累加，而是历史的和新增的有相对应的权重，并且不需要人为设置学习率了，学习率根据历史和前一次参数变化量（注意，这里是参数变化量，前面是梯度变化量）来估计，所以当曲线逐渐平缓接近极值点时，学习率自动就会调整。
```
optimizer = Adadelta(batch_size=self._batch_size, max_iter=self._max_iter, is_plot_loss=True,
                    add_gradient_noise=True)
```

####optimizer/rmsprop.py
rmsprop，hinton大神的算法，和adadelta很像，只不过学习率还是人为设置的，不是估计的，看似好像参数需要人为设置，这是缺点，但是事实上有的时候固定的学习率能下降的更快。
```
optimizer = RMSProp(learning_rate=self._learning_rate, batch_size=self._batch_size, max_iter=self._max_iter,
                    is_plot_loss=True, add_gradient_noise=True)
```

####optimizer/adam.py
adam，这个算法应该是综合起来最好的算法，下降速度非常快，学习率也是需要人为设置，梯度采用动量，衰减方式和前面两种一样，还有一个校正的过程（没看原始论文，不明白这一步是怎么回事）。
```
optimizer = Adam(learning_rate=self._learning_rate, batch_size=self._batch_size, max_iter=self._max_iter,
                is_plot_loss=True, add_gradient_noise=True)
```

####loss/cross_entropy.py
交叉熵损失（二分类时的对数损失）


####loss/mean_square_error.py
均方差损失，主要用于回归问题，有些人将这个误用到逻辑回归（这是二分类）中，这是不对的（不信自己试试）。


####loss/zero_one_loss.py
0-1损失，感知机中用到。


####util/freq_dict.py
一个统计频率的模块，输入是一个列表，统计出列表中每个元素出现的频率，还包含一个plot_pie用来打印出频率的饼图。
```
testdata = [1, 2, 3, 1, 2, 5, 1, 9, 2, 5, 6, 2]
fd = FreqDict(testdata)
fd.print_freq_dict()
fd.plot_pie()
```

####util/heap.py
堆，用于建立大根堆，小根堆或者堆排序使用


####util/kd_tree.py
kd树，用于加速knn搜索近邻。主要维持2个队列，一个待搜索队列，一个已经搜过的元素的队列。遍历树时，如果根据条件搜索右子树，则将左子树加入待搜队列，反之亦然，循环这个过程。之所以将很多不满足条件的子树加入待搜队列，是因为在那些不满足条件的子树中有可能有更近的节点存在（至于为什么，看一下统计学习方法或者画个图），然后再通过一个小技巧就能大大减少在待搜队列里搜索的次数（也就是关于那个超球体的），具体难描述，大家自己看代码，就一行。


####util/line_search.py
一维搜索，或者叫线搜索。包含2个非精确线搜索算法，armijo和wolfe。整体过程就是要寻找一个步长，这个步长能使目标函数有足够的下降。armijo和wolfe这两算法都是包含2个条件，并且第一个条件还一样，由于第二个条件的差别，wolfe比armijo要好一些，因为不会排除掉极小值点。


####util/sampler.py
采样器，支持根据概率采样。根据概率采样时，有2种方式，一种是正常的，另一种是快速但是会损失一定的精度


####util/shuffle_split.py
用于打乱并划分数据的模块，和dataset模块里面的cross_split工作一样。

##结束
可以看到，上面很多实现我自己也不确定是否正确，所以各位看官需要抱着怀疑的态度。如果发现了哪里有BUG或者算法本身有问题，告诉我一下，感谢。
