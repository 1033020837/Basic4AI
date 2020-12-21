参考资料：https://blog.csdn.net/v_july_v/article/details/7624837

1. 导出SVM要优化的问题导出

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

2. SVM优化问题的转化

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

   由于$\alpha_i\ge0,g_i\le0$，因此$\sum_{i=1}^n\alpha_ig_i\le0$，因此$\max_{\alpha_i\ge0}L(w,b,\alpha)=\frac{1}{2}||w||^2$，因此原问题求解$\min\frac{1}{2}||w||^2$可转化为求解$\min\max_{\alpha_i\ge0}L(w,b,\alpha)$，令$\theta(\alpha)=\max_{\alpha}L(w,b,\alpha)$，$\theta(w,b)=\min_{w,b} L(w,b,\alpha)$，则$\theta(w,b)=\min_{w,b}L(w,b,\alpha)\le\min_{w,b}\max_{\alpha}L(w,b,\alpha)\le\min_{w,b}\theta(\alpha)\le\theta(\alpha)$，该式对所有的$w,b,\alpha$均成立，则$\max_{\alpha}\min_{w,b}L(w,b,\alpha)\le\min_{w,b}\max_{\alpha}L(w,b,\alpha)$恒成立，当L满足强对偶条件时取等号，强对偶条件即为下面所介绍的KKT条件。

   参考：https://www.pianshen.com/article/15821257925/，通过引入松弛变量将不等式转化为等式并求导后可以得到以下KKT条件：
   $$
   \frac{\partial{L(w,b,\alpha)}}{\partial{w}}=0
   \\ \frac{\partial{L(w,b,\alpha)}}{\partial{b}}=0
   \\ \alpha_i\ge0
   \\ g_i\le0
   \\ \alpha_ig_i=0
   $$
   至此，通过让原函数满足KKT条件，将极小极大问题转化为了极大极小问题，即求解$\max_{\alpha}\min_{w,b}L(w,b,\alpha)$。

3. SVM优化问题求解

   