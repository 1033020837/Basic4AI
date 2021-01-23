

1. 什么是TextCNN

   Yoon Kim在论文(2014 EMNLP) [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)提出TextCNN，该模型将卷积神经网络CNN应用到文本分类任务，是卷积神经网络应用到文本分析的开创性工作之⼀。

2. TextCNN的结构

   TextCNN的结构图如下：

   [![sTMltJ.png](https://s3.ax1x.com/2021/01/23/sTMltJ.png)](https://imgchr.com/i/sTMltJ)

   具体包含如下结构：

   - Embedding层

     将词的One-hot表示映射为稠密向量表示。

   - 一维卷积层

     宽度设为词嵌入维度，高度为卷积核大小（超参数），在word-level上进行一维卷积。虽然文本经过词嵌入后是二维数据，但是在embedding-level上的二维卷积没有意义。同一卷积核大小一般设置多个卷积核来提取不同的特征。

   - 时序最大池化层

     对一个卷积核得到的feature map取最大值，由于一个卷积核是在word-level即按照时序进行卷积的，所以称为时序最大池化（max-over-time pooling）。

   - 全连接层

     将各个卷积、池化后的结果拼接后经过最后一层或多层全连接层将特征转化为label的概率分布。

3. TextCNN学到了什么

   TextCNN不同大小的卷积核学习到的是卷积核大小n对应的某个n-gram特征，时序最大池化层提取句子中该特征的最大取值，最后的全连接层组合这些n-gram特征进行分类。因此，TextCNN能够学习到很多用于分类的局部的特征，适用于短文本的分类，而对于有较长依赖关系的长文本分类效果较差。