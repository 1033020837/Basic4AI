1. Transformer是什么

   Transformer是Google在2017年的论文《Attention Is All You Need》中所提出的一种Seq2Seq的模型，该模型完全的抛弃了以往深度学习所使用的CNN、RNN等结构而全部使用Attention结构。Transformer的效果和并行性都非常好，其作为一个整体能被用于机器翻译、语音识别、文本摘要等传统Seq2Seq被应用的领域，基于其Encoder部分所构建的Bert、基于其Decoder部分所构建的GPT都是目前NLP领域十分热门的模型。

2. Transformer的结构

   - 总体结构

     Transformer采用Seq2Seq架构，分为Encoder和Decoder模块，Encoder由6个Encoder子模块堆叠而成，Decoder由6个Decoder子模块堆叠而成。下图为其总览图：

     [![sXnwQJ.png](https://s3.ax1x.com/2021/01/26/sXnwQJ.png)](https://imgchr.com/i/sXnwQJ)

     每一个Encoder子模块和Decoder子模块的内部结构如下图：

     [![sXut0I.png](https://s3.ax1x.com/2021/01/26/sXut0I.png)](https://imgchr.com/i/sXut0I)

     可以看到，每个Encoder子模块包含两层，一个self-attention层和一个前馈神经网络；每个Decoder子模块包含三层，在self-attention层和前馈神经网络间还有一层attention用于融合Encoder所编码的信息。

   - Encoder结构

     首先，模型需要对输入的数据进行一个embedding操作将词表示为嵌入向量，enmbedding结束之后，输入到encoder层，self-attention处理完数据后把数据送给前馈神经网络，前馈神经网络的计算可以并行，得到的输出会输入到下一个encoder。

     - Embedding

       [![sXU3Ie.png](https://s3.ax1x.com/2021/01/26/sXU3Ie.png)](https://imgchr.com/i/sXU3Ie)

       如上图所示，Embedding包括两个部分，绿色部分就跟传统的Embedding一样，将词的id映射为稠密向量；黄色部分为位置编码，因为Transformer未使用RNN这类时序结构，因此需要添加位置编码来表示单词的位置。具体地，Transformer使用如下方式进行位置编码：
       $$
       PE(pos,k)=\left\{\begin{aligned}
       \sin(\frac{pos}{10000^{2i/d_{model}}}) \qquad \quad k=2i\\
       \cos(\frac{pos}{10000^{2i/d_{model}}}) \qquad k=2i+1
       \end{aligned}\right.
       $$
       关于位置编码的解释可以参考：[https://www.zhihu.com/question/347678607/answer/864217252](https://www.zhihu.com/question/347678607/answer/864217252)。

       将两种编码方式相加即得到词的嵌入向量。

     - self-attention

       假设我们想要翻译这个句子：

       “The animal didn't cross the street because it was too tired”

       那么it在这句话中是是指animal还是street，人类好理解这句话，但是对机器来说就很困难了。当模型处理这个单词“it”的时候，自注意力机制会允许“it”与“animal”建立联系。可视化如下图：

       [![sXwDIO.png](https://s3.ax1x.com/2021/01/26/sXwDIO.png)](https://imgchr.com/i/sXwDIO)

       self-attention的具体工作流程为：

       1. 为每个单词向量生成3个新的向量Query、Key、Value（通过乘以三个参数矩阵实现）；

          [![sXBaE6.png](https://s3.ax1x.com/2021/01/26/sXBaE6.png)](https://imgchr.com/i/sXBaE6)

       2. 计算self-attention的分数值，该分数值决定了当我们在某个位置encode一个词时，对输入句子的其他部分的关注程度。

          该分数的计算方法是Query与Key做点乘。以下图为例，我们针对Thinking这个词，计算出其他词对于该词的一个分数值，首先是针对于自己本身即q1·k1，然后是针对于第二个词即q1·k2。

          [![sXD6WF.png](https://s3.ax1x.com/2021/01/26/sXD6WF.png)](https://imgchr.com/i/sXD6WF)

       3. 接下来，把点成的结果除以一个常数，这里我们除以8，这个值一般是采用上文提到的矩阵的第一个维度的开方即64的开方8，当然也可以选择其他的值，这样会使得梯度更加稳定。此方法成为scaled dot-product attention。

          然后把得到的结果做一个Softmax的计算进行归一化。得到的结果即是每个词对于当前位置的词的相关性大小，当然，当前位置的词相关性肯定会会很大，但有时关注另一个与当前单词相关的单词也会有帮助。

          [![sXrl6J.png](https://s3.ax1x.com/2021/01/26/sXrl6J.png)](https://imgchr.com/i/sXrl6J)

       4. 下一步就是把Value和softmax得到的值进行相乘，并相加，得到的结果即是self-attetion在当前节点的值。

          [![sXsaEq.png](https://s3.ax1x.com/2021/01/26/sXsaEq.png)](https://imgchr.com/i/sXsaEq)

       5. 实际实现时，上述操作是通过矩阵运算实现的。

          [![sXyasH.png](https://s3.ax1x.com/2021/01/26/sXyasH.png)](https://imgchr.com/i/sXyasH)

       6. Multi-Head Attention

          论文中使用了Multi-Head Attention（多头注意力机制），每一个头都是一组上述操作，一共8组。

          [![sXcA3T.png](https://s3.ax1x.com/2021/01/26/sXcA3T.png)](https://imgchr.com/i/sXcA3T)

          [![sXceu4.png](https://s3.ax1x.com/2021/01/26/sXceu4.png)](https://imgchr.com/i/sXceu4)

       7. 拼接每个头的输出并经过一个矩阵变换后得到self-attention的最终输出。

          [![sXc6Kg.png](https://s3.ax1x.com/2021/01/26/sXc6Kg.png)](https://imgchr.com/i/sXc6Kg)

       用一张图来表示self-attention的完整过程：

       [![sXc5GV.png](https://s3.ax1x.com/2021/01/26/sXc5GV.png)](https://imgchr.com/i/sXc5GV)

     - Add & Nomorlize

       在Transformer中，每一个子层（self-attetion，Feed Forward Neural Network）之后都会接一个ResNet模块（即在输出和输入间连接一条通路），并且有一个Layer normalization。

       Layer normalization与在神经网络一节介绍的Batch normalization类似，不过Batch normalization是在每一批数据的每一个维度上进行归一化，而Layer normalization是在每一个样本上进行归一化。

       [![sX2mkR.png](https://s3.ax1x.com/2021/01/26/sX2mkR.png)](https://imgchr.com/i/sX2mkR)

     - 前馈神经网络（Feed Forward Neural Network）

       由两个全连接和一个ReLU函数实现，即：
       $$
       FFN(x)=W_2ReLU(W_1x+b_1)+b_2
       $$
       论文中前馈神经网络模块输入和输出的维度均为512，其内层的维度为2048。

   - Decoder结构

     Decoder结构与Encoder大致相同，但有两点不同需要指出：

     - Masked Multi-Head Attention

       与Encoder在一开始就给定了完整的句子不同，Decoder在预测的时候存在一个解码过程，即输出序列是一个接一个生成的，因此在处理序列中的第t个单词时，模型只能看到第t个单词和它之前的单词。训练过程为了保持和测试过程的一致性，也应当遵循这一原则。

       具体的做法是在处理第t个词时，将self-ttention向量的第t维之后的值全部设为0。

     - Encoder-Decoder Attention

       类似于传统的Seq2Seq，Transformer的Decoder部分的每一个子模块都会有一个self-attention部分来接受Encoder最后一个子模块的输出，self-attention的Key、Value来自Encoder，Query来自Decoder。

3. Transformer的优点

   - 更好的并行能力，Decoder部分训练时可并行；
   - 更强的特征抽取能力，网络层叠式的设计、self-attention机制能够更好地学习到词与词之间的依赖关系比如长距离依赖；
   - 自注意力可以产生更具可解释性的模型。

4. Transformer的其他知识点

   - Position Encoding的好处

     编码数值范围在[-1,1]；

     正弦函数周期性的变化使得相邻单词的位置编码不依赖于序列长度；

     位置编码的变化主要发生在低维度，这可能是可以将位置编码直接与词嵌入向量相加的原因之一；

     $PE_{pos+k}$可以表示为$PE_{pos}$的线性函数，让模型更容易学习到相对位置信息。

   - 多头注意力机制的好处

     进行 Multi-head Attention 的原因是将模型分为多个头，形成多个子空间，可以让模型去关注不同方面的信息，最后再将各个方面的信息综合起来。

   - Query和Key为什么不使用相同矩阵

     避免单词与自身的attention分数过高而占据绝对主导地位，使得模型有机会关注到其他单词；

     打破对称性，A对B的重要度可能与B对A的不同；

     增强了表达能力，提高了泛化能力。

   - Softmax之前为什么要对Attention分数进行scaled

     参考：[https://www.zhihu.com/question/339723385/answer/782509914](https://www.zhihu.com/question/339723385/answer/782509914)，太大的输入会让梯度消失，同时点积结果的方差与输入维度有关。

5. 参考链接

   [https://github.com/NLP-LOVE/ML-NLP/tree/master/NLP/16.7%20Transformer](https://github.com/NLP-LOVE/ML-NLP/tree/master/NLP/16.7%20Transformer)

   [史上最全Transformer面试题：灵魂20问帮你彻底搞定Transformer](https://github.com/DA-southampton/NLP_ability/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/Transformer/%E7%AD%94%E6%A1%88%E8%A7%A3%E6%9E%90(1)%E2%80%94%E5%8F%B2%E4%B8%8A%E6%9C%80%E5%85%A8Transformer%E9%9D%A2%E8%AF%95%E9%A2%98%EF%BC%9A%E7%81%B5%E9%AD%8220%E9%97%AE%E5%B8%AE%E4%BD%A0%E5%BD%BB%E5%BA%95%E6%90%9E%E5%AE%9ATransformer.md)

   [https://blog.csdn.net/orangerfun/article/details/104851834](https://blog.csdn.net/orangerfun/article/details/104851834)

   [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

   [https://www.zhihu.com/question/347678607/answer/864217252](https://www.zhihu.com/question/347678607/answer/864217252)

   [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)

   

   

