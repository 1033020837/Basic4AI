"""
EM算法
三硬币模型以及高斯混合模型

高斯混合模型主要参考：https://zhuanlan.zhihu.com/p/55826713
"""

import random
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.style.use('seaborn')

'''
三硬币模型
'''
class ThreeCoins(object):

    def __init__(self,pi,p,q) -> None:
        '''
        pi p q 分别对应硬币A B C正面朝上的概率
        '''
        super().__init__()
        self.pi = pi
        self.p = p
        self.q = q

    def gen_ys(self, n):
        '''
        生成长度为n的观测序列
        '''

        ys = []
        for _ in range(n):
            if random.random() < self.pi:    # A正面朝上，投掷B
                if random.random() < self.p:
                    ys.append(1)
                else:
                    ys.append(0)
            else:   # A反面朝上，投掷C
                if random.random() < self.q:
                    ys.append(1)
                else:
                    ys.append(0)

        return ys

    def estimate_params(self, ys, pi=0.4, p=0.4, q=0.4, max_step=100):
        '''
        使用EM算法估计三硬币模型的参数
        '''


        n = len(ys)
        e = 1e-5
        for _ in range(max_step):
            mu = []
            for y in ys:
                mu.append(pi*(p**y)*((1-p)**(1-y))/(pi*(p**y)*((1-p)**(1-y))+\
                            (1-pi)*(q**y)*((1-q)**(1-y))))
            
            mu_sum = sum(mu)

            new_pi = mu_sum / n
            new_p = sum([a*b for a,b in zip(mu,ys)]) / mu_sum
            new_q = sum([(1-a)*b for a,b in zip(mu,ys)]) / (n-mu_sum)


            if abs(pi-new_pi) < e and abs(p-new_p) < e and abs(q-new_q) < e:
                break

            pi, p, q = new_pi, new_p, new_q

        return pi,p,q

    



class GMM(object):
    '''
    混合高斯模型
    参考 https://zhuanlan.zhihu.com/p/55826713
    '''

    def __init__(self) -> None:
        super().__init__()

    def gen_data(self, params):
        '''
        生成二维高斯混合数据

        params 参数 列表  每一个元素为  [点的数量,[x均值,y均值],[x方差，y方差]]
        '''

        X = []
        for num,mu,var in params:
            temp = np.random.multivariate_normal(mu, np.diag(var), num)
            X.append(temp)

        # 合并在一起
        X = np.vstack(X)

        return X

    def estimate_params(self, X, K, init_params,  max_step=100):
        '''
        使用EM算法估计GMM的参数
        X 观测点
        K 高斯分布的个数
        init_params 参数的初始化值 列表 每一个元素为  [比重,[x均值,y均值],[x方差，y方差]]
        '''

        alphas = [x[0] for x in init_params]    # 各个分模型的比重
        mus = [x[1] for x in init_params]   # 各个分模型的均值
        vars = [x[2] for x in init_params]    # 各个分模型的方差
        n = len(X)  # 观测点的个数
        e = 1e-5    # 迭代变化不超过此值时停止
        pdfs = np.zeros(((n, K)))   # 各个分模型对观测数据的响应度  n * K
        for _ in range(max_step):
            for k in range(K):
                pdfs[:,k] = alphas[k] * multivariate_normal.pdf(X, mus[k], np.diag(vars[k]))
            pdfs = pdfs / np.sum(pdfs, 1, keepdims=True)

            pdf_sums = np.sum(pdfs,0)

            # 根据响应度更新参数
            new_alphas,new_mus, new_vars = [],[],[]
            for k in range(K):
                new_mus.append((np.sum(pdfs[:,k].reshape((-1,1))*X,0)/pdf_sums[k]).tolist())
                new_vars.append((np.sum(pdfs[:,k].reshape((-1,1))*(X-new_mus[-1])**2,0)/pdf_sums[k]).tolist())
                new_alphas.append(pdf_sums[k]/n)

            if abs(np.mean(np.array(mus)-np.array(new_mus))) < e and \
                abs(np.mean(np.array(alphas)-np.array(new_alphas))) < e and \
                abs(np.mean(np.array(vars)-np.array(vars))) < e:
                break    

            alphas, mus, vars = new_alphas, new_mus, new_vars


        return alphas, mus, vars

def plot_clusters(X, Mu, Var, Mu_true=None, Var_true=None):
    '''
    画图函数  参考：https://zhuanlan.zhihu.com/p/55826713
    Mu var 为EM算法估计的均值 方差
    Mu_true Var_true 为真实均值 方差

    虚线为EM算法估计的高斯分布
    实线为真实高斯分布
    '''
    colors = ['b', 'g', 'r']
    n_clusters = len(Mu)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5)
    ax = plt.gca()
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        ellipse = Ellipse(Mu[i], 3 * Var[i][0], 3 * Var[i][1], **plot_args)
        ax.add_patch(ellipse)
    if (Mu_true is not None) & (Var_true is not None):
        for i in range(n_clusters):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
            ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
            ax.add_patch(ellipse)         
    plt.show()

if __name__ == '__main__':

    # 测试三硬币
    # pi,p,q = 0.1,0.4,0.4
    # three_coins = ThreeCoins(pi,p,q)
    # ys = three_coins.gen_ys(100)
    # pi_hat,p_hat,q_hat = three_coins.estimate_params(ys, pi=0.1, p=0.1, q=0.1, max_step=1000)
    # print(pi_hat,p_hat,q_hat)


    # 测试高斯混合模型
    gmm = GMM()
    # 第一簇的数据
    num1, mu1, var1 = 400, [0.5, 0.5], [1, 3]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, [5.5, 2.5], [2, 2]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, [1, 7], [6, 2]
    params = [[num1,mu1,var1],[num2,mu2,var2],[num3,mu3,var3],]
    # 生成数据
    X = gmm.gen_data(params)
    K = 3
    init_params = [[1/K,[0, -1],[1, 1]],[1/K,[6, 0],[1, 1]],[1/K,[0, 9],[1, 1]]]
    alphas, mus, vars = gmm.estimate_params(X, K, init_params, max_step=100)
    # 画图
    plot_clusters(X, mus, vars, [mu1, mu2, mu3], [var1, var2, var3])

    


