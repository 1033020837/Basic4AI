1. 什么是AdaBoost

   标准AdaBoost关注二分类问题，AdaBoost通过训练一系列的弱分类器来组成一个强分类器，每一轮训练时会提高前一轮弱分类器错误分类样本的权值，而降低那些被正确分类的样本的权值。模型最后的预测结果为各弱分类器预测结果的加权多数表决结果。

   AdaBoost算法具体流程：

   - 输入：训练数据集$T={(x_1,y_1),(x_2,y_2),..,(x_N,y_N)}$，其中$x_i\in \mathbb{R}^n$，$Y_i\in \{-1,1\}$；弱分类器算法（一般为树桩）；
   
   - 输出：最终分类器$G(x)$.
   
   - 1. 初始化训练数据的权值分布为
   
        $D_1=(w_{11},w_{12},...,w_{1N}),w_{1i}=\frac{1}{N}$
   
     2. 对m=1,2,...,M（M为弱分类器数量）
   
        - 使用具有权值分布$D_m$的训练数据学习，得到第m个基分类器$G_m(x)$
   
        - 计算$G_m(x)$的分类误差率：
          $$
          e_m=\sum_{i=1}^Nw_{mi}I(G_m(x_i)\ne y_i)
          $$
   
        - 计算$G_m(x)$的系数（权重）
          $$
          \alpha_m=\frac{1}{2}\ln \frac{1-e_m}{e_m}
          $$
   
        - 更新训练集数据权值分布
          $$
          D_{m+1}=(w_{m1},w_{m2},...,w_{mN})
          \\ w_{mi}=\frac{w_{mi}e^{-\alpha_m y_iG(x_i)}}{Z_m}
          \\ Z_m=\sum_{i=1}^Nw_{mi}e^{-\alpha_m y_iG(x_i)}
          $$
   
        - 最终分类器
          $$
          G(x)=sign(\sum_{m=1}^M\alpha_mG_m(x))
          $$
   
2. AdaBoost算法的解释

   AdaBoost算法可解释为模型是加法模型、损失函数为指数函数、学习算法为前向分步算法时的二分类学习算法。

   - 前向分步算法

     加法模型：
     $$
     f(x)=\sum_{m=1}^M\beta_mb(x;\gamma_m)
     $$
     其中，$b(x;\gamma_m)$为基函数，$\gamma_m$为基函数参数，$beta_m$为基函数的系数，M为基函数的个数。

     学习加法模型$f(x)$即对经验风险极小化：
     $$
     \min_{\beta_m,\gamma_m}\sum_{i=1}^NL(y_i,\sum_{m=1}^M\beta_mb(x;\gamma_m))
     $$
     通常这是一个复杂的优化问题，前向分布算法求解这一优化问题的想法是：从前往后每一步只学习一个基函数及其系数，逐步逼近最优解。具体地，每一步需要优化如下损失函数：
     $$
     \min_{\beta,\gamma}\sum_{i=1}^NL(y_i,f_{m-1}(x_i)+\beta b(x;\gamma))
     $$

   - 前向分布算法与AdaBoost

     前向分步算法逐一学习基函数，这与AdaBoost算法逐一学习基本分类器的过程一致。

     当前向分步算法的损失函数为指数损失函数，即：
     $$
     L(y,f(x))=e^{-yf(x)}
     $$
     时，其学习的具体操作等价于AdaBoost算法学习的具体操作。

     假设经过m-1轮迭代前向分步算法已经得到：
     $$
     f_{m-1}(x)=\alpha_1G_1(x)+\alpha_2G_2(x)+...+\alpha_{m-1}G_{m-1}(x)
     $$
     在第m轮迭代得到$\alpha_m,G_m(x)和f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)$，目标是希望得到：
     $$
     (\alpha_m,G_m(x))=\arg\min_{\alpha_m,G_m(x)}\sum_{i=1}^Ne^{-y_i(f_{m-1}(x_i)+\alpha_mG_m(x_i))}
     $$
     令$\overline w_{mi}=e^{-y_if_{m-1}(x_i)}$，则上式子可以化为：
     $$
     (\alpha_m,G_m)=\arg\min_{\alpha_m,G_m(x)}\sum_{i=1}^N\overline w_{mi}e^{-\alpha_my_iG_m(x_i)}
     \\ =\arg\min_{\alpha_m,G_m}e^{-\alpha_m}\sum_{y_i=G_m(x_i)}\overline w_{mi}+e^{\alpha_m}\sum_{y_i\ne G_m(x_i)}\overline w_{mi}
     \\ =\arg\min_{\alpha_m,G_m}e^{-\alpha_m}(\sum_{i=1}^N\overline w_{mi}-\sum_{y_i\ne G_m(x_i)}\overline w_{mi})+e^{\alpha_m}\sum_{y_i\ne G_m(x_i)}\overline w_{mi}
     \\ =\arg\min_{\alpha_m,G_m}e^{-\alpha_m}\sum_{i=1}^N\overline w_{mi}+(e^{\alpha_m}-e^{-\alpha_m})\sum_{y_i\ne G_m(x_i)}\overline w_{mi}
     \\ =\arg\min_{\alpha_m,G_m}e^{-\alpha_m}\sum_{i=1}^N\overline w_{mi}+(e^{\alpha_m}-e^{-\alpha_m})\sum_{i=1}^N\overline w_{mi}I(y_i\ne G_m(x_i))
     $$
     对于固定的$\alpha_m$，上式中$e^{-\alpha_m}\sum_{i=1}^N\overline w_{mi}$和$e^{\alpha_m}-e^{-\alpha_m}$都是定值，则上式等价于：
     $$
     G_m^*=\arg\min_{G_m}\sum_{i=1}^N\overline w_{mi}I(y_i\ne G_m(x_i))
     $$
     这与AdaBoost中要寻找的基本分类器一致。

     然后对$\alpha_m$求导并使其等于0，得：
     $$
     \alpha_m^*=\frac{1}{2}\ln \frac{1-e_m}{e_m}
     \\ e_m=\frac{\sum_{i=1}^N\overline w_{mi}I(y_i\ne G_m(x_i))}{\sum_{i=1}^N\overline w_{mi}}
     $$
     
     令$w_{mi}=\frac{\overline w_{mi}}{\sum_{i=1}^N\overline w_{mi}}$，得$e_m=\sum_{i=1}^Nw_{mi}I(y_i\ne G_m(x_i))$，这与AdaBoost一致。特别地，当$m=0$时，$\overline w_{mi}=e^{-y_i*0}=1,w_{mi}=\frac{1}{N}=\frac{\overline w_{mi}}{\sum_{i=1}^N\overline w_{mi}}$。
     
     由$\overline w_{mi}=e^{-y_if_{m-1}(x_i)}$以及$f_m(x_i)==f_{m-1}(x)+\alpha_mG_m(x_i)$得：
     $$
     \overline w_{m+1,i}=\overline w_{mi}e^{-y_i\alpha_mG_m(x_i)}
     $$
     由$w_{m+1,i}=\frac{\overline w_{m+1,i}}{\sum_{i=1}^N\overline w_{m+1,i}}$以及上式得：
     $$
     w_{m+1,i}=\frac{\overline w_{mi}e^{-y_i\alpha_mG_m(x_i)}}{\sum_{i=1}^N\overline w_{mi}e^{-y_i\alpha_mG_m(x_i)}}
     \\ =\frac{\frac{\overline w_{mi}}{\sum_{i=1}^N\overline w_{mi}}e^{-y_i\alpha_mG_m(x_i)}}{\sum_{i=1}^N\frac{\overline w_{mi}}{\sum_{i=1}^N\overline w_{mi}}e^{-y_i\alpha_mG_m(x_i)}}
     \\ =\frac{w_{mi}e^{-y_i\alpha_mG_m(x_i)}}{\sum_{i=1}^Nw_{mi}e^{-y_i\alpha_mG_m(x_i)}}
     $$
     与AdaBoost一致。
     
     综上，模型是加法模型、损失函数为指数函数、学习算法为前向分步算法时可以推导出AdaBoost。

   ​         

   

   

