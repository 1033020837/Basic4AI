"""
bp全连接神经网络
"""

import numpy as np
from sklearn.datasets import *

class ActFunc(object):
    '''
    激活函数
    '''
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z):
        '''
        前向计算
        z 输入
        '''
        NotImplemented

    def derivation(self, z, a):
        '''
        反向求导
        z 输入
        a 输出
        '''
        NotImplemented


class Sigmoid(ActFunc):

    def forward(self, z):
        shape = z.shape
        z = z.reshape((-1,))
        res = np.array([(1. / (1. + np.exp(-x))) if x >= 0 else (np.exp(x) / (1. + np.exp(x))) for x in z])
        return res.reshape(shape)

    def derivation(self, z, a):
        return a*(1-a)

class Tanh(ActFunc):
    def forward(self, z):
        shape = z.shape
        z = z.reshape((-1,))
        res = np.array([(2. / (1. + np.exp(-2*x)) -1 ) if x >= 0 else (2 * np.exp(2*x) / (1. + np.exp(2*x)) - 1) for x in z])
        return res.reshape(shape)  

    def derivation(self, z, a):
        return 1 - a**2

class ReLU(ActFunc):
    
    def forward(self, z):
        return np.where(z<0,0,z)
    
    def derivation(self, z, a):
        return np.where(z<0,0,1)


class Softmax(ActFunc):

    def forward(self, z):
        _max = np.max(z,axis=1,keepdims=True)
        z = np.exp(z - _max)
        return z / np.sum(z,axis=1,keepdims=True)

    def derivation(self, z, a):
        '''
        Softmax一般放在输出层，与损失函数作为一个整体计算导数
        '''
        NotImplemented

class LossFunc(object):
    '''
    损失函数
    '''
    def forward(self, y_true, y_pred):
        '''
        计算损失
        '''
        NotImplemented
    
    def derivation(self, y_true, y_pred):
        '''
        反向求导
        '''
        NotImplemented


class CrossEntropy(LossFunc):
    '''
    交叉熵
    '''
    def forward(self, y_true, y_pred):
        y_true = y_true.reshape((-1,))
        return -np.mean(np.log(y_pred[np.arange(len(y_true)),y_true]))

    def derivation(self, y_true, y_pred, act_func):
        if isinstance(act_func, Softmax):
            y_true = y_true.reshape((-1,))
            res = y_pred
            res[np.arange(len(y_true)),y_true] = res[np.arange(len(y_true)),y_true] - 1
            return res
        else:
            raise NotImplementedError('CrossEntropy must be combined with Softmax yet!')

def init_weight(input_num, output_num):
    '''
    随机初始化权重
    '''
    a = np.sqrt(6) / np.sqrt(input_num + output_num)
    return np.random.uniform(-a, a, size=(output_num, input_num))

class NeuralNetwork(object):

    def __init__(self, layers) -> None:
        '''
        input_num 输入结点个数
        layers 输入层、隐藏层、输出层 list 每一个元素为[神经元个数，激活函数] 
        acts 每一层的激活函数
        '''
        super().__init__()

        self.layer_num = len(layers)    # 总网络层数（包括输出层，不包括输入层）
        
        self.layers = []    # 神经网络的所有层，包括权重、偏置、激活函数

        for i in range(self.layer_num):
            layer = []
            if i == 0:
                layer = [None] * 3
            else:
                # 权重
                layer.append(init_weight(layers[i-1][0], layers[i][0]))
                # 偏置
                layer.append(np.zeros((layers[i][0], 1)))
                # 激活函数
                layer.append(layers[i][1])
            self.layers.append(layer)

    def forward(self, inputs):
        '''
        前向计算
        inputs 输入   bacth_size * feature_num
        return 每一层激活后的输出
        '''
        outputs = []  # 每一层的输出,包括激活前和激活后的输出

        for i in range(self.layer_num):
            if i == 0:  # 输入层
                z,a = inputs,inputs
            else:
                w,b,act = self.layers[i]    # 权重、偏置、激活函数
                z = np.dot(w,outputs[-1][1].T) + b
                z = z.T
                a = act.forward(z)
            outputs.append([z,a])

        return outputs

    def backward(self, outputs, targets, loss_func, learning_rate):
        '''
        反向求导
        outputs 每一层输出
        targets 真实输出
        loss_func   损失函数
        learning_rate 学习率
        '''
        assert isinstance(loss_func, LossFunc)

        loss_value = loss_func.forward(targets, outputs[-1][1])

        delta = None                    # 损失函数对当前层未经激活的输出的导数
        batch_size = len(targets)
        for i in range(self.layer_num-1, 0, -1):
            act = self.layers[i][2]
            z,a = outputs[i]
            if i == self.layer_num-1:
                delta = loss_func.derivation(targets, a, act)
            else:
                delta = np.dot(delta, self.layers[i+1][0]) * act.derivation(z,a)
            gradient_w = np.dot(delta.T, outputs[i-1][1]) / batch_size  # 对权重的梯度
            gradient_b = np.mean(delta,0).reshape((-1,1))   # 对偏置的梯度
            

            # 更新参数
            self.layers[i][0] -= learning_rate * gradient_w
            self.layers[i][1] -= learning_rate * gradient_b

        return loss_value

    def train(self, train_xs, train_ys, batch_size, loss_func, \
             test_xs=None, test_ys=None, learning_rate = 1e-2, epochs=5):
        '''
        随机梯度下降训练
        batch_size 批大小
        '''
        n = train_xs.shape[0]   # 训练集大小
        
        # 迭代训练
        for epoch in range(epochs):
            cur = 0
            loss_values = []
            while cur < n:
                batch_xs = train_xs[cur:cur+batch_size,:]
                batch_ys = train_ys[cur:cur+batch_size]

                outputs = self.forward(batch_xs)
                loss_value = self.backward(outputs, batch_ys, loss_func, learning_rate)
                loss_values.append(loss_value)
                cur += batch_size

            # 每个epoch结束测试一次
            if test_xs is not None and test_ys is not None:
                accuracy = self.test(test_xs, test_ys)
                print('Epoch:%d, train loss:%.4f, test accuracy:%.4f'%(epoch+1,\
                    np.mean(loss_values),accuracy))

    def test(self, test_xs, test_ys):
        '''
        测试函数
        '''
        predict_ys = np.argmax(self.forward(test_xs)[-1][1],1)
        accuracy = (predict_ys == test_ys).sum() / test_ys.shape[0]

        return accuracy

if __name__ == '__main__':


    # 加载sklearn自带的手写数字识别数据集
    digits = load_digits()
    features = digits.data
    targets = digits.target

    # 对特征进行标准化
    mean, std = np.mean(features,0),np.std(features,0)
    features = (features - mean) / (std + 1e-4)

    np.random.seed(2021)
    # 随机打乱数据
    shuffle_indices = np.random.permutation(features.shape[0])  
    features = features[shuffle_indices]
    targets = targets[shuffle_indices]

    # 划分训练、测试集
    train_count = int(len(features)*0.8)
    train_xs, train_ys = features[:train_count], targets[:train_count]
    test_xs, test_ys = features[train_count:], targets[train_count:]


    batch_size = 64
    learning_rate = 1e-1
    epochs = 100

    layers = [[test_xs.shape[1],None],[32,ReLU()],[targets.max()+1,Softmax()]]
    nn = NeuralNetwork(layers)

    nn.train(train_xs,train_ys,batch_size,CrossEntropy(),test_xs,test_ys,learning_rate,epochs)




            


            





    










        