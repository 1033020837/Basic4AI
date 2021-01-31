# 说明

机器学习、深度学习、自然语言处理基础知识总结。

目前主要参考李航老师的《统计学习方法》一书，也有一些内容例如**XGBoost**、**LSTM+CRF**、**深度学习相关内容**等是书中未提及的。

由于github的markdown解析器不支持latex，因此笔记部分需要在本地使用Typora才能正常浏览，也可以直接访问下面给出的博客链接。

Document文件夹下为笔记，Code文件夹下为代码，Data文件夹下为某些代码所使用的数据集，Image文件夹下为笔记部分所用到的图片。

由于时间和精力原因，部分代码来自github开源项目，如Seq2Seq、Transformer等部分的代码。

# 机器学习

- 线性回归（[笔记](https://www.cnblogs.com/lyq2021/p/14353781.html)）
- 感知机（[笔记](https://www.cnblogs.com/lyq2021/p/14253768.html)+[代码](Code/perceptron.py)）
- KNN（[笔记](https://www.cnblogs.com/lyq2021/p/14253756.html)+[代码](Code/knn.py)）
- 朴素贝叶斯（[笔记](https://www.cnblogs.com/lyq2021/p/14253771.html)+[代码](Code/naive_bayes.py)）
- 决策树（[笔记](https://www.cnblogs.com/lyq2021/p/14253778.html)+[代码](Code/decision_tree.py)）
- 逻辑回归（[笔记](https://www.cnblogs.com/lyq2021/p/14253818.html)+[代码](Code/logistic_regression.py)）
- 最大熵（[笔记](https://www.cnblogs.com/lyq2021/p/14253820.html)+[代码](Code/max_entropy.py)）
- SVM（[笔记](https://www.cnblogs.com/lyq2021/p/14253858.html)+[代码](Code/svm.py)）
- AdaBoost（[笔记](https://www.cnblogs.com/lyq2021/p/14253860.html)+[代码](Code/adaboost.py)）
- GBDT（[笔记](https://www.cnblogs.com/lyq2021/p/14253863.html)+[代码](Code/gbdt.py)）
- EM算法（[笔记](https://www.cnblogs.com/lyq2021/p/14253869.html)+[代码](Code/em.py)）
- 隐马尔可夫模型（[笔记](https://www.cnblogs.com/lyq2021/p/14253871.html)+[代码](Code/hmm.py)）
- 条件随机场（[笔记](https://www.cnblogs.com/lyq2021/p/14253872.html)）
- 随机森林（[笔记](https://www.cnblogs.com/lyq2021/p/14253876.html)+[代码](Code/random_forest.py)）
- XGBoost（[笔记](https://www.cnblogs.com/lyq2021/p/14253885.html)）
- 聚类（[笔记](https://www.cnblogs.com/lyq2021/p/14341111.html)）

# 深度学习

- 神经网络（[笔记](https://www.cnblogs.com/lyq2021/p/14269424.html)+[代码](Code/neural_network.py)）
- RNN([笔记](https://www.cnblogs.com/lyq2021/p/14295398.html))
- LSTM和GRU([笔记](https://www.cnblogs.com/lyq2021/p/14302282.html))
- CNN([笔记](https://www.cnblogs.com/lyq2021/p/14321103.html))
- 深度学习中的最优化方法（[笔记](https://www.cnblogs.com/lyq2021/p/14336242.html)）

# 自然语言处理

- 词嵌入之Word2Vec([笔记](https://www.cnblogs.com/lyq2021/p/14308673.html))
- 词嵌入之GloVe([笔记](https://www.cnblogs.com/lyq2021/p/14312830.html))
- 词嵌入之FastText([笔记](https://www.cnblogs.com/lyq2021/p/14313968.html))
- TextCNN（[笔记](https://www.cnblogs.com/lyq2021/p/14317291.html)+[代码](Code/textcnn.py)）
- Seq2Seq（[笔记](https://www.cnblogs.com/lyq2021/p/14325262.html)+[代码](https://github.com/1033020837/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb)）
- Transformer（[笔记](https://www.cnblogs.com/lyq2021/p/14330534.html)+[代码](https://github.com/1033020837/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)）
- BERT（[笔记](https://www.cnblogs.com/lyq2021/p/14347124.html)）
- LSTM+CRF进行序列标注（[笔记](https://www.cnblogs.com/lyq2021/p/14253897.html)）


# 待添加部分

- 降维算法
- 特征选择方法
- 主题模型
- LightGBM

