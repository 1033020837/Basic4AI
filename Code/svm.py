"""
支持向量机
"""

import numpy as np
from sklearn.datasets import load_digits
import random
from tqdm import tqdm

class SVM(object):
    def __init__(self, C=10, epsilon=1e-4, kernel_type='gauss', sigma = 10, p = 3) -> None:
        '''
        C 惩罚参数  越大对误分类的惩罚越大
        epsilon 计算时的精度
        kernel_type 核函数类型 'gauss' 高斯和函数  'poly' 多项式核函数
        sigma 高斯核的参数
        p 多项式核的参数
        '''
        super().__init__()

        
        assert kernel_type in ['gauss','poly']

        self.C = C
        self.epsilon = epsilon
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.p = p

    def train(self, train_xs, train_ys, max_iter=10):
        '''
        训练
        train_xs 训练特征
        train_ys 训练标签
        max_iter 最大迭代次数
        '''

        cur_iter = 0    # 当前迭代轮次
        is_params_changed = True    # 上一轮迭代是否有参数改变
        n,m = train_xs.shape   # 训练集样本数目、特征维度

        # 将需要在其他函数中用到的变量保存为实例变量
        self.n = n
        self.m = m
        self.xs = train_xs
        self.ys = train_ys
        self.alphas = np.zeros((n,))    # 将所有拉格朗日乘子初始化为0
        self.kernel_values = np.zeros((n,n))    # 将所有样本间的核函数先提前计算好保存下来
        for i in range(n):
            for j in range(i,n):
                self.kernel_values[i,j] = self.kernel_values[j,i] = self._cal_kernel(self.xs[i],self.xs[j])
        self.b = 0  # SVM的偏置项
        self.gs = np.zeros((n,))  # 保存每一个样本使用当前参数预测并去除偏置项的值，加速误差计算
        self.is_error_modified = np.zeros((n,))  # 保存该样本的误差是否被修改过


        while cur_iter < max_iter and is_params_changed:
            
            is_params_changed = False
            cur_iter += 1

            # 外层循环选取一个违反KKT条件的样本
            for i in tqdm(range(n)):
                if not self._is_satisfy_kkt(i):

                    error_i = self._cal_ei(i)
                    j, error_j  = self._get_alpha_j(i, error_i)

                    # 将符号与书上对应方便理解
                    alpha1, alpha2 = self.alphas[i], self.alphas[j]
                    e1, e2 = error_i, error_j
                    y1, y2 = self.ys[i], self.ys[j]
                    k11,k22,k12 = self.kernel_values[i,i], self.kernel_values[j,j],self.kernel_values[i,j]

                    # 求alphas[j]更新的上下界
                    if y1 == y2:
                        l,h = max(0,alpha1 + alpha2 - self.C), min(self.C, alpha1 + alpha2)
                    else:
                        l,h = max(0, alpha2 - alpha1), min(self.C, self.C + alpha2 - alpha1)
                    
                    eta = k11 + k22 - 2 * k12

                    # 未经剪辑的更新后的alpha2
                    alpha2_new_unc = alpha2 + y2 * (e1 - e2) / eta
                    
                    # 剪辑alpha2
                    self.alphas[j] = max(min(alpha2_new_unc,h),l)

                    # 求更新后的alpha1
                    self.alphas[i] = alpha1 + y1 * y2 * (alpha2 - self.alphas[j])

                    b1_new = -e1 - y1 * k11 * (self.alphas[i] - alpha1) - y2 * k12 * (self.alphas[j] - alpha2) + self.b
                    b2_new = -e2 - y1 * k12 * (self.alphas[i] - alpha1) - y2 * k22 * (self.alphas[j] - alpha2) + self.b


                    if self.alphas[i] > 0 and self.alphas[i] < self.C:
                        self.b = b1_new
                    elif self.alphas[j] > 0 and self.alphas[j] < self.C:
                        self.b = b2_new
                    else:
                        self.b = (b1_new + b2_new) / 2
                    

                    # alpha1 alpha2的更新量
                    delta_alpha1 = self.alphas[i] - alpha1
                    delta_alpha2 = self.alphas[j] - alpha2

                    # 更新gs
                    self.gs += delta_alpha1 * self.ys[i] * self.kernel_values[i]
                    self.gs += delta_alpha2 * self.ys[j] * self.kernel_values[j]

                    self.is_error_modified[i] = 1
                    self.is_error_modified[j] = 1

                    # 判断是否有参数更新
                    if delta_alpha1 != 0:
                        is_params_changed = True

    def test(self, test_xs, test_ys):
        '''
        测试
        '''
        
        predict_ys = []
        for i in range(test_xs.shape[0]):
            predict_value = self.b
            for j in range(self.n):
                if self.alphas[j] != 0:
                    predict_value += self.alphas[j] * self.ys[j] * self._cal_kernel(test_xs[i], self.xs[j])
            if predict_value >= 0:
                predict_ys.append(1)
            else:
                predict_ys.append(-1)
        predict_ys = np.array(predict_ys)
        accuracy = (predict_ys == test_ys).sum() / test_ys.shape[0]
        print('Accuracy:%.4f'%accuracy)

    def _is_satisfy_kkt(self, i):
        '''
        判断第i和alpha在episilon的精度下是否满足KKT条件
        '''
        gxi = self.gs[i]
        yi = self.ys[i]

        if (abs(self.alphas[i]) < self.epsilon) and (yi * gxi >= 1):
            return True
        elif (abs(self.alphas[i] - self.C) < self.epsilon) and (yi * gxi <= 1):
            return True
        elif (self.alphas[i] > -self.epsilon) and (self.alphas[i] < (self.C + self.epsilon)) \
                and (abs(yi * gxi - 1) < self.epsilon):
            return True
        return False

    def _cal_kernel(self, x1, x2):
        '''
        计算 x1 和 x2 的核函数
        '''

        if self.kernel_type == 'gauss':
            tmp = np.sum((x1 - x2) ** 2) / (2 * self.sigma ** 2)
            return np.exp(-tmp)
        else:
            return (np.sum(x1 * x2) + 1) ** self.p

    def _cal_ei(self, i):
        '''
        计算Ei，即使用当前参数计算xi的预测值与真实值间的差值
        '''
        return self.gs[i] + self.b - self.ys[i]

    def _get_alpha_j(self, i, error_i):
        '''
        选取使alpha[j]变化最大的j
        参考 https://github.com/wojiushimogui/SVM/blob/master/svm.py
        '''
        candidates = np.nonzero(self.is_error_modified)[0]
        max_e1_minus_e2 = 0
        j = 0
        error_j = 0
    
        # find the alpha with max iterative step  
        if len(candidates) > 1:  
            for k in candidates:  
                error_k = self._cal_ei(k)  
                if abs(error_k - error_i) > max_e1_minus_e2:  
                    max_e1_minus_e2 = abs(error_k - error_i)  
                    j = k  
                    error_j = error_k  
        # if came in this loop first time, we select alpha j randomly  
        else:             
            j = i  
            while j == i:  
                j = int(random.uniform(0, self.n))  
            error_j = self._cal_ei(j)  

        return j, error_j 
    

if __name__ == '__main__':

    # 加载sklearn自带的手写数字识别数据集
    digits = load_digits()
    features = digits.data
    targets = (digits.target > 4).astype(int)   
    targets[targets==0] = -1

    # 随机打乱数据
    shuffle_indices = np.random.permutation(features.shape[0])  
    features = features[shuffle_indices]
    targets = targets[shuffle_indices]

    # 划分训练、测试集
    train_count = int(len(features)*0.8)
    train_xs, train_ys = features[:train_count], targets[:train_count]
    test_xs, test_ys = features[train_count:], targets[train_count:]

    kernel_type='gauss'
    svm = SVM(kernel_type=kernel_type)

    svm.train(train_xs, train_ys, max_iter=5)
    svm.test(test_xs, test_ys)
