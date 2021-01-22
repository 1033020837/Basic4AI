1. 什么是FastText

   FastText是Facebook于2016年开源的一个词向量计算和文本分类工具，它提出了子词嵌入的方法，试图在词嵌入向量中引入构词信息。一般情况下，使用fastText进行文本分类的同时也会产生词的embedding，即embedding是fastText分类的产物。

2. FastText流程

   FastText的架构图为：

   [![sIBYI1.png](https://s3.ax1x.com/2021/01/22/sIBYI1.png)](https://imgchr.com/i/sIBYI1)

   分为输入层、隐含层、输出层，下面分别介绍这三层：

   - 输入层

     输入层包含三类特征：

     - 全词特征，也就是每个词的完整词嵌入向量；
     - 字符n-gram特征，例如对于单词$where$，首先在其首尾添加开始和结束的符号得到$<where>$，其trigram特征为$<wh,whe,her,ere,re>$，每个字符n-gram都会被映射成对应嵌入向量；
     - 词n-gram特征，例如对于句子I like machine learning，其bigram特征为I like，like machine，machine learning，每个词n-gram特征都会被映射成对应嵌入向量；

   - 隐藏层

     对所有输入特征取均值。

   - 输出层

     使用Word2Vec一节中介绍的层次Softmax输出文档类别，霍夫曼树的构造基于每个类别出现的频数。

   FastText架构与CBOW非常相似，不同的是：

   - CBOW的输入是目标单词的上下文，FastText的输入是多个单词及其n-gram特征，这些特征用来表示单个文档；
   - CBOW的输出是目标词汇，fastText的输出是文档对应的类标。

   FastText的核心思想就是：将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。

3. FastText的优点

   - 充分利用了构词信息，能够提升英语、德语等利用构词法进行构词的语言的嵌入效果；
   - 能够很好的解决未登录词（OOV）的问题，解决方法是将未登录词表示为其字符嵌入的均值；
   - 由于在分类时加入了两类n-gram信息，分类性能得到了提升；
   - 速度很快。

   