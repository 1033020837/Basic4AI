"""
逻辑回归
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn import preprocessing

def sigmoid(z):
    '''
    数值稳定版本的Sigmoid函数
    '''
    res = np.array([(1. / (1. + np.exp(-x))) if x >= 0 else (np.exp(x) / (1. + np.exp(x))) for x in z])

    return res

class LogisticRegression(object):
    def __init__(self, m):
        '''
        m 特征维度
        '''
        self.w = np.zeros((m+1,1))   # 包含偏置项

    def train(self, train_xs, train_ys, batch_size, \
             test_xs=None, test_ys=None, learning_rate = 1e-2, epochs=5):
        '''
        随机梯度下降训练
        batch_size 批大小
        '''
        n = train_xs.shape[0]   # 训练集大小
        train_xs = np.concatenate([train_xs,np.ones((n,1))],-1) # 增加一列全为1的特征，方便将偏置与权重统一处理
        if test_xs is not None:
            test_xs = np.concatenate([test_xs,np.ones((test_xs.shape[0],1))],-1)
        train_ys = train_ys.reshape((-1,1))
        if test_ys is not None:
            test_ys = test_ys.reshape((-1,1))
        
        # 迭代训练
        for epoch in range(epochs):
            cur = 0
            while cur < n:
                batch_xs = train_xs[cur:cur+batch_size,:]
                batch_ys = train_ys[cur:cur+batch_size,:]

                p = sigmoid(np.matmul(batch_xs, self.w)) # batch_size * 1

                delta = np.mean((p - batch_ys) * batch_xs,0).reshape((-1,1))    # 梯度

                self.w -= learning_rate * delta # 更新权重

                cur += batch_size

            # 每个epoch结束测试一次
            if test_xs is not None and test_ys is not None:
                accuracy = self.test(test_xs, test_ys)
                print('Epoch:%d, accuracy:%.4f'%(epoch+1,accuracy))

    def test(self, test_xs, test_ys):
        '''
        测试函数
        '''
        p = sigmoid(np.matmul(test_xs, self.w))
        predict_ys = (p > 0.5).astype(int)
        accuracy = (predict_ys == test_ys).sum() / test_ys.shape[0]

        return accuracy

if __name__ == '__main__':

    # 加载sklearn自带的手写数字识别数据集
    digits = load_digits()
    features = digits.data
    targets = (digits.target > 4).astype(int)   # 0~4为类别0，5~9为类别1

    # 随机打乱数据
    shuffle_indices = np.random.permutation(features.shape[0])  
    features = features[shuffle_indices]
    targets = targets[shuffle_indices]

    # 划分训练、测试集
    train_count = int(len(features)*0.8)
    train_xs, train_ys = features[:train_count], targets[:train_count]
    test_xs, test_ys = features[train_count:], targets[train_count:]

    lr = LogisticRegression(train_xs.shape[1])

    batch_size = 128
    learning_rate = 1e-3
    epochs = 20

    # 20轮准确率87.78%
    lr.train(train_xs, train_ys, batch_size, test_xs, test_ys, learning_rate, epochs)

