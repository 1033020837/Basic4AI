1. 为什么使用LSTM+CRF进行序列标注

   直接使用LSTM进行序列标注时只考虑了输入序列的信息，即单词信息，没有考虑输出信息，即标签信息，这样无法对标签信息进行建模，所以在LSTM的基础上引入一个标签转移矩阵对标签间的转移关系进行建模。这一点和传统CRF很像，CRF中存在两类特征函数，一类是针对观测序列与状态的对应关系，一类是针对状态间关系。在LSTM+CRF模型中，前一类特征函数由LSTM的输出给出，后一类特征函数由标签转移矩阵给出。

2. 由输入序列x计算条件概率p(y|x)

   设输入序列x长度为n，即$x=(x_1,x_2,...,x_n)$，可能的标签个数为m，即存在$m^n$种可能的输出序列$y=(y_1,y_2,...,y_n)$。

   设LSTM输出的各个时刻各标签的概率为$E\in\mathbb{R}^{n*m}$，转移矩阵为$T\in\mathbb{R}{m*m}$，任意序列y的得分为score(y)，则：
   $$
   score(y)=\sum_{i=1}^n({E[i,y_i]+T[y_{i-1},y_i]})
   $$
   利用Softmax进行归一化得到序列y的概率：
   $$
   P(y|x)=\frac{e^{score(y)}}{Z(x)},
   \\ 其中 Z(x)=\sum_{y}e^{score(y)}
   $$
   
   取对数得：
   $$
   \ln{P(y|x)}=score(y)-\ln{Z(x)}
   $$
   所以关键是求取上式中的后面部分即$\ln{Z(x)}$，直接求取的时间复杂度为$O(m^n)$，考虑使用动态规划来求解。
   
   记$\alpha(y_i=t_j)=\sum_{y_i=t_j}e^{score(y_i=t_j)}$为第i时刻输出第j个标签的所有路径得分取取指数的和，则：
   $$
   \alpha(y_{i+1}=t_j)=\sum_{k=1}^m\sum_{y_i=t_k}e^{score(y_i=t_k)+E(i+1,t_j)+T(t_k,t_j)}
   \\=e^{E(i+1,t_j)}*\sum_{k=1}^{m}(e^{T(t_k,t_j)}*\sum_{y_i=t_k}e^{score(y_i=t_k)})
   $$
   取对数得：
   $$
   \ln\alpha(y_{i+1}=t_j)=E(i+1,t_j)+\ln{\sum_{k=1}^me^{T(t_k,t_j)}*\alpha(y_i=t_k)}
   \\=E(i+1,t_j)+\ln{\sum_{k=1}^me^{T(t_k,t_j)}*e^{\ln\alpha(y_i=t_k)}}
   \\=E(i+1,t_j)+\ln{\sum_{k=1}^me^{T(t_k,t_j)+\ln\alpha(y_i=t_k)}}
   $$
   令$\beta_i=[\ln\alpha(y_i=t_1),\ln\alpha(y_i=t_2),...,\ln\alpha(y_i=t_m)]\in\mathbb{R}^m$，则：
   $$
   \beta_{i+1}=[\ln\sum_{k=1}^me^{\beta_{i,k}+T(t_k,t_0)}+E(i+1,t_0),\ln\sum_{k=1}^me^{\beta_{i,k}+T(t_k,t_1)}+E(i+1,t_1),
   \\...,\ln\sum_{k=1}^me^{\beta_{i,k}+T(t_k,t_m)}+E(i+1,t_m)]
   $$
   使用一个m维数组存储$\beta$即可编程实现。
   
   通过使用$-P(y|x)$作为Loss即可实现端到端的训练。

3. 使用维特比算法得到最优路径

   推理时如果直接计算每条路径的得分然后取得分最大的路径则时间复杂度为$m^n$，再次考虑使用动态规划来求解。

   记$\delta_i\in\mathbb{R}^m$，其第j维$\delta_{i,j}$表示i时刻以标签$t_j$结尾的所有路径的得分中的最大得分，则：
   $$
   \delta_{i+1,j}=\max_{k}[\delta_{i,k}+T(t_k,t_j)+E(i+1,t_j)]
   \\=\max_{k}[\delta_{i,k}+T(t_k,t_j)]
   $$
   同时使用$Q\in\mathbb{R}^{n*m}$来方便进行路径回溯，矩阵第i行第j列对应元素$Q_{i,j}$表示第i个时刻以标签$t_j$结尾时得分最大路径的第i-1时刻所对应的标签，即：
   $$
   Q_{i+1,j}=\arg\max_{k}[\delta_{i,k}+T(t_k,t_j)]
   $$
   通过$\delta和Q$进行回溯即可求得最优路径。

4. 编程实现时的注意事项
   - 使用数值稳定版本的$\ln\sum\exp$函数。
   - 对于使用batch实现的批操作，注意针对长度不同的序列要使用mask，计算$P(y|x)$以及推理时均需要。


