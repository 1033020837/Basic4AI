"""
感知机
"""

import numpy as np
from sklearn.datasets import load_digits


class Perceptron(object):
    def __init__(self, m):
        '''
        m 特征维度
        '''
        self.w = np.zeros((m,))   # 权重
        self.b = 0 # 偏置

    def train(self, train_xs, train_ys, batch_size, \
             test_xs=None, test_ys=None, learning_rate = 1e-1, epochs=5):
        '''
        随机梯度下降训练
        batch_size 批大小
        '''
        n = train_xs.shape[0]   # 训练集大小
        
        # 迭代训练
        for epoch in range(epochs):
            cur = 0
            while cur < n:
                batch_xs = train_xs[cur:cur+batch_size,:]
                batch_ys = train_ys[cur:cur+batch_size]

                p = np.matmul(batch_xs, self.w) + self.b # batch_size * 1
                p = p.reshape((-1,))
                p[p >= 0] = 1
                p[p < 0] = -1

                flags = p != batch_ys

                self.w += np.mean(batch_ys[flags].reshape((-1,1)) * batch_xs[flags], 0)
                self.b += np.mean(batch_ys[flags])

                cur += batch_size

            # 每个epoch结束测试一次
            if test_xs is not None and test_ys is not None:
                accuracy = self.test(test_xs, test_ys)
                print('Epoch:%d, accuracy:%.4f'%(epoch+1,accuracy))

    def test(self, test_xs, test_ys):
        '''
        测试函数
        '''
        p = np.matmul(test_xs, self.w) + self.b
        predict_ys = (p >= 0).astype(int)
        predict_ys[predict_ys==0] = -1
        accuracy = (predict_ys == test_ys).sum() / test_ys.shape[0]

        return accuracy

if __name__ == '__main__':

    # 加载sklearn自带的手写数字识别数据集
    digits = load_digits()
    features = digits.data
    # 0~4为类别0，5~9为类别1
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

    perceptron = Perceptron(train_xs.shape[1])

    batch_size = 64
    learning_rate = 1e-3
    epochs = 20

    # 20轮准确率87.50%
    perceptron.train(train_xs, train_ys, batch_size, test_xs, test_ys, learning_rate, epochs)