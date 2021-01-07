"""
梯度提升树（GBDT）
"""

from decision_tree import DecisionTree
import numpy as np
import threading
from sklearn.datasets import load_digits    # 手写数字类别预测，用于分类任务
from sklearn.datasets import load_boston    # 波士顿房价预测，用于回归任务
from logistic_regression import sigmoid     # 数值稳定版本的Sigmoid函数
from tqdm import tqdm

# 数值稳定版本的Softmax函数  z 二维数组  样本数*类别数
def softmax(z):
    _max = np.max(z,axis=1,keepdims=True)
    z = np.exp(z - _max)
    return z / np.sum(z,axis=1,keepdims=True)
    


class GBDT(object):

    def __init__(self, learning_rate = 1, tree_count = 50, predict_type='classification', split_count=10, max_depth=3) -> None:
        '''
        learning_rate 学习率
        tree_count 决策树数量
        predict_type 预测类型 classification 分类 regression 回归
        split_count 对于连续属性切分的次数
        max_depth 决策树最大深度
        '''
        super().__init__()

        assert predict_type in ['classification','regression']

        self.learning_rate = learning_rate
        self.tree_count = tree_count
        self.predict_type = predict_type
        self.split_count = split_count
        self.max_depth = max_depth

    def train(self, datas, targets, attr_type):
        '''
        训练函数
        '''

        # 对于分类任务，获取类别数
        if self.predict_type == 'classification':
            self.class_count = len(np.unique(targets))
            if self.class_count == 2:
                f_values = np.zeros((datas.shape[0],))
            else:
                # 对标签进行one-hot编码
                targets = self._one_hot_encoder(targets)

                f_values = np.zeros((datas.shape[0],self.class_count))
        else:
            f_values = np.zeros((datas.shape[0],))

        self.trees = [] # 保存所有子树

        for _ in tqdm(range(self.tree_count)):
            # 计算残差
            residuals = self._cal_residual(f_values, targets)
            
            if self.predict_type == 'classification' and self.class_count > 2:  # 多分类任务一次需要构建多棵树，每棵树拟合一个类别
                trees = []
                for i in range(self.class_count):
                    tree = DecisionTree('CART', 'regression', self.split_count, self.max_depth)
                    tree.tree = tree.build_tree(datas, residuals[:,i], attr_type, 0, np.arange(datas.shape[0]))
                    trees.append(tree)
                    f_values[:,i] += self.learning_rate * tree.predict_values
                self.trees.append(trees)
            else:
                tree = DecisionTree('CART', 'regression', self.split_count, self.max_depth)
                tree.tree = tree.build_tree(datas, residuals, attr_type, 0, np.arange(datas.shape[0]))
                self.trees.append(tree)
                f_values += self.learning_rate * tree.predict_values    # 更新f_values
            

    def test(self, datas, targets):
        '''
        测试函数
        '''
        predict_targets = []
        for i in range(datas.shape[0]):
            predict_target = self.predict(datas[i])
            predict_targets.append(predict_target)
        predict_targets = np.array(predict_targets)
        if self.predict_type == 'classification':   # 分类任务计算准确率
            accuracy = (predict_targets == targets).mean()
            print('Accuracy:%.4f'%accuracy)
        else:   # 回归任务计算均方误差
            mse = ((predict_targets - targets) ** 2).mean()
            print('MSE:%.4f'%mse)

    def predict(self, data):
        '''
        预测函数

        data 单个样本
        '''
        if self.predict_type == 'classification' and self.class_count > 2:  # 多分类任务一次需要构建多棵树，每棵树拟合一个类别
            predict_logits = np.zeros_like(data)
            for trees in self.trees:    # 多分类一轮有多棵树
                for i,tree in enumerate(trees):
                    predict_logits[i] += self.learning_rate * tree.predict(tree.tree, data)
            return np.argmax(predict_logits)
        elif self.predict_type == 'classification':
            predict_logit = 0   # 预测的几率
            for tree in self.trees:
                predict_logit += self.learning_rate * tree.predict(tree.tree, data)
            return (predict_logit >= 0).astype(int) # 几率大于等于0时预测为1，否则预测为0
        else:
            predict_target = 0
            for tree in self.trees:
                predict_target += self.learning_rate * tree.predict(tree.tree, data)
            return predict_target

    def _cal_residual(self, f_values, targets):
        '''
        计算残差
        f_values 当前所有树对所有训练样本的预测值
                 对于回归任务和二分类任务，f_values为一个长度为训练样本数的一维array
                 对于多分类任务，f_values为一个二维array，第一维为训练样本数， 第二维为类别数，
        '''
        if self.predict_type == 'classification' and self.class_count > 2:  # 多分类任务一次需要构建多棵树，每棵树拟合一个类别
            p_hat = softmax(f_values)
            return targets - p_hat
        elif self.predict_type == 'classification':
            p_hat = sigmoid(f_values) # 计算预测为类别1的概率
            return targets - p_hat
        else:
            return targets - f_values
    
    def _one_hot_encoder(self, sparse_target):
        '''
        one-hot编码
        '''
        one_hot_target= np.zeros((sparse_target.shape[0],self.class_count))  
        one_hot_target[np.arange(sparse_target.shape[0]), sparse_target] = 1
        return one_hot_target

if __name__ == '__main__':

    binary_classification = False   # 是否进行二分类
    predict_type = 'regression' # 分类还是回归任务  classification 分类 regression 回归

    if predict_type == 'classification':
        # 加载sklearn自带的手写数字识别数据集
        digits = load_digits()
        features = digits.data
        targets = digits.target
        if binary_classification:
            targets = (digits.target > 4).astype(int)   # 0-4 设为标签0 5-9 设为标签1
    else:
        boston = load_boston()
        features = boston.data
        targets = boston.target

    np.random.seed(2021)    # 为了比较决策树、随机森林、GBDT的效果，固定随机种子
    # 随机打乱数据
    shuffle_indices = np.random.permutation(features.shape[0])
    features = features[shuffle_indices]
    targets = targets[shuffle_indices]

    # 划分训练、测试集
    train_count = int(len(features)*0.8)
    train_datas, train_targets = features[:train_count], targets[:train_count]
    test_datas, test_targets = features[train_count:], targets[train_count:]

    attr_type = [1] * train_datas.shape[1]

    tree_count = 100
    learning_rate = 0.1
    max_depth = 3
    gbdt = GBDT(learning_rate = learning_rate, tree_count = tree_count,\
             predict_type=predict_type, max_depth=max_depth)
    gbdt.train(train_datas, train_targets, attr_type)
    gbdt.test(test_datas, test_targets)
