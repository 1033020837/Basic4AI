1. 导出SVM要优化的问题

   ![img](https://img-blog.csdn.net/20140829134124453)

   对于上图中这样一个二分类线性可分问题，期望找到一个分类超平面将正负类分开，SVM就是一个用来寻找这样的分类超平面的算法。

   定义正负类的标签分别为1、-1，分类超平面的表达式为$f(x)=w^Tx+b$，其中x为样本向量，w、b分别为超平面的权重以及偏置项。可以由$f(x)$的符号来区分样本的类别，当样本类别为1时$f(x)>0$，当样本类别为-1时$f(x)<0$，则$yf(x)>0$始终成立。

   对于任意一个样本点x，令其垂直投影到超平面上的对应点为 x<sub>0</sub> ，w 是垂直于超平面的一个向量(法向量)，![img](https://img-blog.csdn.net/20140829135315499)为样本x到超平面的距离（带符号），如下图所示：

   ![img](http://blog.pluskid.org/wp-content/uploads/2010/09/geometric_margin.png)

   则x可表示为：
   $$
   x=x_0+{\gamma}\frac{w}{||w||}
   $$
   将$f(x_0)=w^Tx_0+b=0$代入上式得：
   $$
   \gamma=\frac{f(x)}{||w||}
   $$
   对$\gamma$取绝对值，得到：
   $$
   |\gamma|=\frac{yf(x)}{||w||}
   $$
   将上式称作点x到超平面的间隔，将所有数据点到超平面间隔的最小值称为该数据集D到超平面的间隔$\tilde{\gamma}$，即:
   $$
   \tilde{\gamma}=\min_{x\in{D}}|\gamma|=\min_{x\in{D}}\frac{yf(x)}{||w||}
   $$
   对于超平面$f(x)=w^Tx+b$，对w以及b同时增大或者减少任意倍数时，平面位置不变，上式分子与分母改变的倍数相同，即$\tilde\gamma$的值不变。为了方便计算，可以将上式中分子部分的值调整为最小值等于1，即$yf(x)\ge1$，则上式可转化为：
   $$
   \tilde\gamma=\frac{1}{||w||}
   $$
   至此我们得到了数据集到超平面间隔的表达式，SVM的核心思想是希望最大化这个间隔即求解$\max\frac{1}{||w||}$。由于样本点距离超平面的远近可以代表该点被分类的难易程度，因此关于SVM的核心思想可以直观理解为希望把最难区分的点的分类效果做到最好。

   求解$\max\frac{1}{||w||}$等价于求解$\min\frac{1}{2}||w||^2$，问题转化为了求解一个凸函数的最小值。将前面的约束条件带上之后可以将SVM要优化的问题形式化表述为：
   $$
   \min\frac{1}{2}||w||^2\qquad s.t.,y_if(x_i)\ge1,i=1,2,...,n
   $$
   其中n为样本数。该问题为一个带不等式约束条件的最值问题。

2. 原始问题转化为对偶问题

   将上式中的每一个约束条件乘上一个拉格朗日乘子$\alpha$得到拉格朗日函数：
   $$
   L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^n\alpha{_i}(1-y_if(x_i))
   \\=\frac{1}{2}||w||^2+\sum_{i=1}^n\alpha{_i}(1-y_i(w^Tx_i+b))
   $$

   记$g_i=1-y_i(w^Tx_i+b)$，则：
   $$
   L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^n\alpha{_i}g_i
   $$

   由几何性质有$\alpha_i\ge0$(参考：https://www.cnblogs.com/liaohuiqiang/p/7805954.html，当不等式约束不起作用时$\alpha_i=0$，当不等式约束起作用时约束函数与原函数在最优解处梯度方向相反，$\alpha_i>0$)，直观理解为：若$\alpha_i<0$，由于$g_i\le0$，则$L(w,b,\alpha)$将不存在极值。

   由于$\alpha_i\ge0,g_i\le0$，因此$\sum_{i=1}^n\alpha_ig_i\le0$，因此$\max_{\alpha_i\ge0}L(w,b,\alpha)=\frac{1}{2}||w||^2$，因此求解$\min\frac{1}{2}||w||^2$可转化为求解：
   $$
   \min\max_{\alpha_i\ge0}L(w,b,\alpha)
   $$
   记$min_{w,b}\max_{\alpha}L(w,b,\alpha)$为原始问题，解为$w_1,b_1,\alpha_1$；$\max_{\alpha}\min_{w,b}L(w,b,\alpha)$为其对偶问题，解为$w_2,b_2,\alpha_2$。易得$L(w_2,b_2,\alpha_2)\le L(w_1,b_1,\alpha_2) \le L(w_1,b_1,\alpha_1)$，则$\max_{\alpha}\min_{w,b}L(w,b,\alpha)\le\min_{w,b}\max_{\alpha}L(w,b,\alpha)$恒成立，当L满足强对偶条件时取等号，强对偶条件即为下面所介绍的KKT条件。

   参考：https://www.pianshen.com/article/15821257925/，通过引入松弛变量将不等式转化为等式并求导后可以得到以下KKT条件：
   $$
   \frac{\partial{L(w,b,\alpha)}}{\partial{w}}=0
   \\ \frac{\partial{L(w,b,\alpha)}}{\partial{b}}=0
   \\ \alpha_i\ge0
   \\ g_i\le0
   \\ \alpha_ig_i=0
   $$
   至此，通过让原函数满足KKT条件，将原问题（极小极大问题）转化为了对偶问题（极大极小问题），即：
   $$
   \max_{\alpha}\min_{w,b}\frac{1}{2}||w||^2+\sum_{i=1}^n\alpha{_i}(1-y_i(w^Tx_i+b))
   \\s.t.\quad\alpha_i\ge 0
   \\1-y_i(w^Tx_i+b)\le 0
   $$

3. 对偶问题求解

   为了求解对偶问题，需要先求$L(w,b,\alpha)$对于$w,b$的极小，再求对$\alpha$的极大。

   - 求$\min_{w,b}L(w,b,\alpha)$

     对w和b求偏导：
     $$
     \frac{\partial L(w,b,\alpha)}{\partial w}=w-\sum_{i=1}^n\alpha{_i}y_ix_i
     \\\frac{\partial L(w,b,\alpha)}{\partial b}=-\sum_{i=1}^n\alpha{_i}y_i
     $$
     令偏导数等于0得：
     $$
     w=\sum_{i=1}^n\alpha{_i}y_ix_i
     \\\sum_{i=1}^n\alpha{_i}y_i=0
     $$
     代回到$L(w,b,\alpha)$的表达式得：
     $$
     L(w,b,\alpha)=\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha{_i}y_i\alpha{_j}y_j(x_i^Tx_j)+\sum_{i=1}^n\alpha{_i}-\sum_{i=1}^n\alpha{_i}y_i((\sum_{j=1}^n\alpha{_j}y_jx_j^T)x_i+b)
     \\=-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha{_i}y_i\alpha{_j}y_j(x_i^Tx_j)+\sum_{i=1}^n\alpha{_i}
     $$

   - 求$\min_{w,b}L(w,b,\alpha)$对$\alpha$的极大，即：
     $$
     \max_{\alpha}-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha{_i}y_i\alpha{_j}y_j(x_i^Tx_j)+\sum_{i=1}^n\alpha{_i}
     \\s.t.\quad \sum_{i=1}^n\alpha{_i}y_i=0
     \\\alpha_i\ge 0, i=1,2,...,n
     $$
     将极大问题转化为极小问题：
     $$
     \min_{\alpha}\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha{_i}y_i\alpha{_j}y_j(x_i^Tx_j)-\sum_{i=1}^n\alpha{_i}
     \\s.t.\quad \sum_{i=1}^n\alpha{_i}y_i=0
     \\\alpha_i\ge 0, i=1,2,...,n
     $$
     具体求解过程在后面介绍。假设求得$\alpha$的解为$\alpha^*$，则相应的w和b为：
     $$
     w^*=\sum_{i=1}^n\alpha{_i}^*y_ix_i
     \\b^*=y_j-\sum_{i=1}^n\alpha{_i}^*y_i(x_i^Tx_j),j满足\alpha_j^*>0
     $$
     $w^*和b^*$的求解只依赖于训练数据中对应于$\alpha_i^*>0$的样本点，将这些点称为支持向量。
     
     最后求得的决策函数为：
     $$
     f(x)=sign(\sum_{i=1}^n\alpha{_i}^*y_i(x_i^Tx)+b^*)
     $$

4. 线性不可分的支持向量机

   前面所述的SVM都是建立在数据是线性可分的基础上的，即存在一个超平面能够将数据按类别完全分开，这对于线性不可分数据是不适用的。线性不可分意味着某些样本点不能满足函数间隔大于等于1的约束条件。为此，对每一个样本点引进一个松弛变量$\xi_i\ge 0$，使得函数间隔加上松弛变量大于等于1。这样，约束条件变为了：
   $$
   y_if(x_i)=y_i(w^Tx_i+b)\ge1-\xi_i,i=1,2,...,n
   $$
   将目标函数修改为：
   $$
   \min \frac{1}{2}||w||^2+C\sum_{i=1}^n\xi_i
   $$
   这里C>0为惩罚参数，值越大对误分类的惩罚越大，越容易造成过拟合。上述目标函数包含两层含义：

   - 使$\frac{1}{2}||w||^2$尽量小即间隔尽量大；
   - 使误分类个数尽量少。

   将上面的优化目标称为软间隔最大化。这样，线性不可分的支持向量机的学习问题变为了：
   $$
   \min_{w,b,\xi} \frac{1}{2}||w||^2+C\sum_{i=1}^n\xi_i
   \\s.t.\quad y_i(w^Tx_i+b)\ge1-\xi_i,i=1,2,...,n
   \\\xi_i\ge 0,i=1,2,...,n
   $$
   该问题的拉格朗日函数为：
   $$
   L(w,b,\xi,\alpha,\mu)=\frac{1}{2}||w||^2+C\sum_{i=1}^n\xi_i-\sum_{i=1}^n\alpha_i(y_i(w^Tx_i+b)-1+\xi_i)-\sum_{i=1}^n\mu_i\xi_i\qquad\alpha_i\ge 0,\mu_i\ge 0,i=1,2,...,n
   $$
   仿照前面的推导，原问题变为:
   $$
   \min_{w,b,\xi} \max_{\alpha,\mu}L(w,b,\xi,\alpha,\mu)
   $$
   其对偶问题为：
   $$
   \max_{\alpha,\mu}\min_{w,b,\xi} L(w,b,\xi,\alpha,\mu)
   $$
   为了求解对偶问题，需要先求$L(w,b,\xi,\alpha,\mu)$对于$w,b,\xi$的极小，再求对$\alpha,\mu$的极大。

   求$L(w,b,\xi,\alpha,\mu)$对$w,b,\xi$的偏导并令其等于0：
   $$
   \frac{\partial L(w,b,\xi,\alpha,\mu)}{\partial w}=w-\sum_{i=1}^n\alpha{_i}y_ix_i=0
   \\\frac{\partial L(w,b,\xi,\alpha,\mu)}{\partial b}=-\sum_{i=1}^n\alpha{_i}y_i=0
   \\\frac{\partial L(w,b,\xi,\alpha,\mu)}{\partial \xi_i}=C-\alpha_i-\mu_i=0
   $$
   得：
   $$
   w=\sum_{i=1}^n\alpha{_i}y_ix_i
   \\\sum_{i=1}^n\alpha{_i}y_i=0
   \\C-\alpha_i-\mu_i=0
   $$
   代回到$L(w,b,\xi,\alpha,\mu)$的表达式得：
   $$
   L(w,b,\xi,\alpha,\mu)=-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha{_i}y_i\alpha{_j}y_j(x_i^Tx_j)+\sum_{i=1}^n\alpha{_i}
   $$
   求$L(w,b,\xi,\alpha,\mu)$对于$\alpha,\mu$的极大，即：
   $$
   \max -\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha{_i}y_i\alpha{_j}y_j(x_i^Tx_j)+\sum_{i=1}^n\alpha{_i}
   \\s.t.\quad \sum_{i=1}^n\alpha{_i}y_i=0
   \\C-\alpha_i-\mu_i=0
   \\\alpha_i\ge 0
   \\\mu_i\ge 0,i=1,2,...,n
   $$
   消去约束条件中的$\mu$并将极大问题转化为极小问题：
   $$
   \min_{\alpha}\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha{_i}y_i\alpha{_j}y_j(x_i^Tx_j)-\sum_{i=1}^n\alpha{_i}
   \\s.t.\quad \sum_{i=1}^n\alpha{_i}y_i=0
   \\0\le \alpha_i\le C, i=1,2,...,n
   $$
   具体求解过程在后面介绍。假设求得$\alpha$的解为$\alpha^*$，则相应的w和b为：
   $$
   w^*=\sum_{i=1}^n\alpha{_i}^*y_ix_i
   \\b^*=y_j-\sum_{i=1}^n\alpha{_i}^*y_i(x_i^Tx_j),j满足0<\alpha_j^*<C
   $$
   将$\alpha_i^*>0$对应的样本点称为支持向量。W

5. 非线性支持向量机与核函数

   前面所述的线性可分支持向量机和线性不可分支持向量机统称为线性支持向量机，对于非线性问题，需要使用非线性支持向量机，其主要特点是利用核技巧。

   - 核函数

     设$\chi$是输入空间（欧氏空间$\R^n$的子集或离散集合），又设$\mathcal H$为特征空间（希尔伯特空间（咱也不知道是啥，大概就是欧式空间的拓展吧）），如果存在一个从$\chi$到$\mathcal H$的映射
     $$
     \phi(x):\chi\rightarrow \mathcal H
     $$
     使得对所有的$x,z\in \chi$，函数$K(x,z)$满足条件
     $$
     K(x,z)=\phi(x)\phi(z)
     $$
     则称$K(x,z)$为核函数，$\phi(x)$为映射函数。

   核技巧的想法是：在学习和预测中只定义核函数$K(x,z)$而不显式定义映射函数$\phi(x)$。对于给定的核函数，映射函数并不唯一。

   我们注意到在线性支持向量机的对偶问题中，无论是目标函数还是决策函数都只涉及输入实例和实例之间的内积，使用核函数替代内积，此时对偶问题变为：
   $$
   \min_{\alpha}\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha{_i}y_i\alpha{_j}y_jK(x_i,x_j)-\sum_{i=1}^n\alpha{_i}
   \\s.t.\quad \sum_{i=1}^n\alpha{_i}y_i=0
   \\0\le \alpha_i\le C, i=1,2,...,n
   $$
   决策函数变为：
   $$
   f(x)=sign(\sum_{i=1}^n\alpha{_i}^*y_iK(x_i,x_j)+b^*)
   $$
   这等价于经过映射函数$\phi(x)$将原来的输入空间变换到一个新的特征空间，在新的特征空间里从训练样本中学习线性支持向量机。当映射函数是非线性函数时，学习到的含有核函数的支持向量机为非线性支持向量机。学习是隐式地在特征空间中进行的，不需要显示地定义特征空间和映射函数。这种技巧称为核技巧。

   在实际应用中，往往依赖于领域知识直接选择核函数，核函数选择的有效性需要通过实验验证。

   通常所说的核函数就是正定核，正定核存在以下充要条件：

   设$K:\chi * \chi \rightarrow \R$是对称函数，则$K(x,z)$为正定核函数的充要条件是对任意$x_i\in \chi,i=1,2,...,m$，$K(x,z)$对应的Gram矩阵：
   $$
   K=[K(x_i,x_j)]_{m*m}
   $$
   是半正定矩阵。

   由于对于一个具体的函数$K(x,z)$，检验对任意有限输入集$\{x_1,x_2,...,x_m\}$，K对应的Gram是否为半正定是很困难的，因此在实际问题中往往应用已有的核函数。

   下面介绍几个常用核函数：

   - 线性核
     $$
     K(x,z)=x^Tz
     $$

   - 多项式核
     $$
     K(x,z)=(x^Tz+1)^p
     $$

   - 高斯核
     $$
     K(x,z)=e^{-\frac{||x-z||^2}{2\sigma^2}}
     $$

6. 序列最小最优化算法（SMO算法）

   不失一般性，使用带核函数的目标函数来表示支持向量机的通用目标函数，最优化问题为：
   $$
   \min_{\alpha}\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n\alpha{_i}y_i\alpha{_j}y_jK(x_i,x_j)-\sum_{i=1}^n\alpha{_i}
   \\s.t.\quad \sum_{i=1}^n\alpha{_i}y_i=0
   \\0\le \alpha_i\le C, i=1,2,...,n
   $$
   在这个问题中，变量是拉格朗日乘子，一个变量$\alpha_i$对应一个样本点$(x_i,y_i)$。SMO算法是一种启发式算法，其解决上述问题的基本思路是：如果所有变量的解都满足此最优化问题的KKT条件，那么这个最优化问题的解就得到了，因为KKT条件是该最优化问题的充要条件。否则，选择两个变量，固定其他变量；针对这两个变量构建一个二次规划问题，这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始问题的目标函数变得更小。重要的是，这时子问题可以通过解析方法求解，这样就可以大大提高整个算法的计算速度。

   整个SMO算法包括两个部分：求解两个变量的二次规划的解析方法、选择变量的启发式方法。

   - 两个变量的二次规划的求解方法

     不失一般性，假设所选两个变量为$\alpha_1,\alpha_2$，其他变量$\alpha_i,i=3,...,n$是固定的。则SMO的待优化问题为：
     $$
     \min_{\alpha_1,\alpha_2} W(\alpha_1,\alpha_2)=\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2+y_1y_2K_{12}\alpha_1\alpha_2-(\alpha_1+\alpha_2)+y_1\alpha_1\sum_{i=3}^ny_i\alpha_iK_{i1}+y_2\alpha_2\sum_{i=3}^ny_i\alpha_iK_{i2}
     \\s.t.\quad\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^n\alpha{_i}y_i=\zeta
     \\0\le \alpha_i\le C, i=1,2
     $$
     其中，$K_{ij}=K(x_i,x_j)$，$\zeta$是常数，上式省略了不含$\alpha_1,\alpha_2$的常数项。

     约束条件$0\le \alpha_i\le C, i=1,2$使得$(\alpha_1,\alpha_2)$在$(0,0)$与$(C,C)$所构成的正方形内；约束条件$\alpha_1y_1+\alpha_2y_2=\zeta$使得$(\alpha_1,\alpha_2)$落在平行于该正方形的对角线的线段上。

     设上述问题的初始可行解为$\alpha_1^{old},\alpha_2^{old}$，最优解为$\alpha_1^{new},\alpha_2^{new}$。由于$\alpha_1=(\zeta-y_2\alpha_2)y_1$，因此可以将问题转化为仅含$\alpha_2$的优化问题。

     为了方便表示，设$v_i=\sum_{j=3}^ny_j\alpha_jK_{ij},i=1,2$，则得到的仅包含$\alpha_2$的目标函数为：
     $$
     W(\alpha_2)=\frac{1}{2}K_{11}(\zeta-y_2\alpha_2)^2+\frac{1}{2}K_{22}\alpha_2^2+y_2K_{12}(\zeta-y_2\alpha_2)\alpha_2-(\zeta-y_2\alpha_2)y_1+v_1(\zeta-y_2\alpha_2)+y_2\alpha_2v_2
     $$
     对$\alpha_2$求导并令其等于0：
     $$
     \frac{\partial W(\alpha_2)}{\partial \alpha_2}=K_{11}\alpha_2+K_{22}\alpha_2-2K_{12}\alpha_2-K_{11}\zeta y_2+K_{12}\zeta y_2+y_1y_2-1-v_1y_2+y_2v_2=0
     $$
     得：
     $$
     (K_{11}+K_{22}-2K_{12})\alpha_2=y_2(y_2-y_1+\zeta K_{11}-\zeta K_{12}+v_1-v_2)
     $$
     令$g(x_i)=\sum_{j=1}^n\alpha_jy_j(x_j^Tx_i)+b,E(x_i)=g(x_i)-y_i$，则：
     $$
     (K_{11}+K_{22}-2K_{12})\alpha_2=y_2(y_2-y_1+\zeta K_{11}-\zeta K_{12}+(g(x_1)-\sum_{j=1}^2y_j\alpha_j^{old}K_{1j}-b)-(g(x_2)-\sum_{j=1}^2y_j\alpha_j^{old}K_{2j}-b))
     $$
     将$\zeta=\alpha_1^{old}y_1+\alpha_2^{old}y_2$代入上式并化简得：
     $$
     (K_{11}+K_{22}-2K_{12})\alpha_2=(K_{11}+K_{22}-2K_{12})\alpha_2^{old}+y_2(E_1-E_2)
     $$
     令$\eta=K_{11}+K_{22}-2K_{12}$，于是，在未将$\alpha_2$限定在$(0,0)$与$(C,C)$所构成的正方形内（即未进行剪辑）时，$\alpha_2$的解为：
     $$
     \alpha_2^{new,unc}=\alpha_2^{old}+\frac{y_2(E_1-E_2)}{\eta}
     $$
     当将$\alpha_2$限定在$(0,0)$与$(C,C)$所构成的正方形内时，需要分情况考虑其上下界。设$L\le\alpha_2^{new}\le H$，则：

     - 当$y_1\neq y_2$时，$L=\max(0,\alpha_2^{old}-\alpha_1^{old}),H=\min(C,C+\alpha_2^{old}-\alpha_1^{old})$
     - 当$y_1=y_2$时，$L=\max(0,\alpha_2^{old}+\alpha_1^{old}-C),H=\min(C,\alpha_2^{old}+\alpha_1^{old})$

     因此：
     $$
     \alpha_2^{new}=\left\{\begin{aligned}
     H \qquad\qquad \alpha_2^{new,unc}>H\\
     \alpha_2^{new,unc} \quad L\le\alpha_2^{new,unc}\le H\\
    L \qquad\qquad \alpha_2^{new,unc}<L
     \end{aligned}\right.
     \\\alpha_1^{new}=\alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_2^{new})
     $$
     由KKT条件易得，当$0<\alpha_1^{new}<C$时，$\sum_{i=1}^n\alpha_iy_iK_{i1}+b=y_1$，即：
     $$
     b_1^{new}=y_1-\sum_{i=3}^n\alpha_iy_iK_{i1}-\alpha_1^{new}y_1K_{11}-\alpha_2^{new}y_2K_{12}
     $$
     因为：
     $$
     E_1=\sum_{i=3}^n\alpha_iy_iK_{i1}+\alpha_1^{old}y_1K_{11}+\alpha_2^{old}y_2K_{12}+b^{old}-y_1
     $$
     所以：
     $$
     b_1^{new}=-E_1-y_1K_{11}(\alpha_1^{new}-\alpha_1^{old})-y_2K_{12}(\alpha_2^{new}-\alpha_2^{old})+b^{old}
     $$
     同理，如果$0<\alpha_2^{new}<C$，那么：
     $$
     b_2^{new}=-E_2-y_1K_{12}(\alpha_1^{new}-\alpha_1^{old})-y_2K_{22}(\alpha_2^{new}-\alpha_2^{old})+b^{old}
     $$
     如果$\alpha_1^{new},\alpha_2^{new}$都在边界上，则处于$b_1^{new},b_2^{new}$中间部分的值都是符合KKT条件的阈值（此处存疑，为什么？），此时选择它们的中点作为更新值。
   
     综上得：
     $$
     b^{new}=\left\{\begin{aligned}
     b_1^{new} \qquad\qquad 0<\alpha_1^{new}<C\\
     b_2^{new} \qquad\qquad 0<\alpha_2^{new}<C\\
     (b_1^{new}+b_2^{new})/2 \qquad\qquad otherwise
     \end{aligned}\right.
     $$
   
   - 变量的选择方法
   
     - 第一个变量的选择
   
       SMO称第一个变量的选择为外层循环，选取训练样本中违反KKT条件最严重的样本点作为第一个变量。具体地，检验训练样本点$(x_i,y_i)$是否满足KKT条件，即：
       $$
       \alpha_i=0\leftrightarrows y_ig(x_i)\ge1
       \\0<\alpha_i<C\leftrightarrows y_ig(x_i) = 1
       \\\alpha_i=C\leftrightarrows y_ig(x_i) \le1
       $$
       该检验是在$\epsilon$范围内进行的（不等式取等号的原因）。
   
       SVM的原作者Platt 的文章里证明过一旦某个$\alpha$处于边界（0 或者 C）的时候，就很不容易变动了，因此，外层循环优先选择所有满足条件$0<\alpha_i<C$的样本点，检验他们是否满足KKT条件。如果它们都满足KKT条件，再遍历整个训练集寻找不满足KKT条件的样本点。
   
     - 第二个变量的选择
   
       SMO称第二个变量的选择为内层循环，假设在外层循环中找到的第一个变量为$\alpha_1$，在内层循环中要寻找的变量为$\alpha_2$，第二个变量的选择标准是使得$\alpha_2$有足够大的变化。
   
       由：
       $$
       \alpha_2^{new,unc}=\alpha_2^{old}+\frac{y_2(E_1-E_2)}{\eta}
       $$
       $\alpha_2$的更新量依赖于$|E_1-E_2|$，因此选择使得$|E_1-E_2|$最大的$\alpha_2$。

7. 使用合页损失（铰链损失）解释SVM

   线性支持向量机的原始最优化问题：
   $$
   \min_{w,b,\xi} \frac{1}{2}||w||^2+C\sum_{i=1}^n\xi_i
   \\s.t.\quad y_i(w^Tx_i+b)\ge1-\xi_i,i=1,2,...,n
   \\\xi_i\ge 0,i=1,2,...,n
   $$
   等价于最优化问题：
   $$
   \min_{w,b} \sum_{i=1}^n\max(0,1-y_i(w^Tx_i+b))+\lambda||w||^2
   $$
   证明如下：

   令$\max(0,1-y_i(w^Tx_i+b))=\xi_i$，则$\xi_i\ge0$。当$1-y_i(w^Tx_i+b)>0$时，$y_i(w^Tx_i+b)=1-\xi_i$；当$1-y_i(w^Tx_i+b)\le0$时，$\xi_i=0$，有$y_i(w^Tx_i+b)\ge1=1-\xi_i$。所以$y_i(w^Tx_i+b)\ge1-\xi_i$，即线性支持向量机的原始最优化问题的约束条件均满足。最优化问题可写成：
   $$
   \min_{w,b}\lambda||w||^2+\sum_{i=1}^n\xi_i
   $$
   取$\lambda=\frac{1}{2C}$，则
   $$
   \min_{w,b} \frac{1}{C}(\frac{1}{2}||w||^2+C\sum_{i=1}^n\xi_i)
   $$
   与线性支持向量机一致。反之也可以由线性支持向量机推导出使用合页损失表示的损失函数。因此二者等价。