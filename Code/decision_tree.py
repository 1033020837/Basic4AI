"""
决策树
ID3 C4.5 CART
"""

import numpy as np
from sklearn.datasets import load_digits

class DecisionTree(object):

    def __init__(self, _type='ID3', predict_type='classification') -> None:
        '''
        _type 决策树类型
        predict_type 预测类型 classification 分类 regression 回归
        '''
        super().__init__()

        assert _type in ['ID3','C4.5','CART']
        assert predict_type in ['classification','regression']

        self.type = _type

        self.tree = None

    def build_tree(self, datas, targets, attr_type):
        '''
        递归构建树
        对于 ID3 C4.5 只支持所有属性均为离散类型
        attr_type   属性类型，0代表离散，1代表连续 list  (m,)
        '''

        m = len(attr_type)
        n = datas.shape[0]

        tree = {'is_leaf':False}

        # 判断是否符合终止条件
        # 样本数为0、所有样本属于同一类别、
        if len(datas) == 0 or len(np.unique(targets)) == 1 or self._is_all_attr_unique(datas):
            tree['is_leaf'] = True
            tree['label'] = self._majority_vote(targets)
            return tree

        # 计算原数据集的信息熵
        if self.type in ['ID3','C4.5']:
            origin_entropy = self._cal_entropy(targets)
        # 保存使用每一个属性分割后带来的信息增益
        gains = []
        # 对于C4.5还需要计算信息增益率
        gain_ratios = []

        for i in range(m):
            if attr_type[i] == 0:
                if self.type in ['ID3','C4.5']:
                    # 针对每一个属性计算分割后的信息熵
                    uniques,counts = np.unique(datas[:,i], return_counts=True)
                    entropy = 0
                    for j in range(uniques.shape[0]):
                        value,count = uniques[j],counts[j]
                        entropy += (count/n) * self._cal_entropy(targets[datas[:,i]==value])
                    # 计算信息增益
                    gain = origin_entropy - entropy
                    gains.append(gain)
                    if self.type == 'C4.5':
                        h = max(self._cal_entropy(datas[:,i]),1e-4)
                        gain_ratios.append(gain/h)
                elif self.type == 'CART':
                    pass
            else:
                pass
        
        if self.type in ['ID3','C4.5']:
            if self.type == 'ID3':
                best_feat = np.argmax(gains)    # 挑选信息增益最大的特征切分
            else:
                # 先计算平均信息增益
                mean_gain = np.mean(gains)
                best_gain_ratio = -1
                best_feat = -1
                for index,(gain,gain_ratio) in enumerate(zip(gains, gain_ratios)):
                    # 在信息增益大于均值的特征中挑选信息增益率最高的特征切分
                    if gain >= mean_gain and gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_feat = index
            tree['best_feat'] = best_feat
            tree['childs'] = {}
            uniques = np.unique(datas[:,best_feat])
            for value in uniques:
                indicies = datas[:,best_feat] == value
                sub_datas = np.hstack((datas[indicies,:best_feat],datas[indicies,best_feat+1:]))
                tree['childs'][value] = self.build_tree(sub_datas,targets[indicies],attr_type[:best_feat]+attr_type[best_feat+1:])
        else:
            pass

        return tree

    def test(self, datas, targets):
        '''
        测试
        '''
        predict_targets = []
        for i in range(datas.shape[0]):
            predict_target = self.predict(self.tree,datas[i])
            predict_targets.append(predict_target)
        predict_targets = np.array(predict_targets)
        accuracy = (predict_targets == targets).sum() / targets.shape[0]
        print('Accuracy:%.4f'%accuracy)

    def predict(self, tree, data):
        '''
        预测函数

        tree 决策树
        data 单个样本
        '''

        if tree['is_leaf']:
            return tree['label']
        
        if self.type in ['ID3','C4.5']:
            return self.predict(tree['childs'][data[tree['best_feat']]],np.hstack((data[:tree['best_feat']],data[tree['best_feat']+1:])))


    def prune(self, tree, datas, targets, alpha=1):
        '''
        后剪枝
        '''

        if self.type in ['ID3','C4.5']:
            c_ta = self._cal_entropy(targets) * targets.shape[0] + alpha    # 该节点缩为叶节点后的损失函数


            if tree['is_leaf']:
                return c_ta
            else:
                c_tb = 0 # 该节点缩为叶节点前的损失函数
                for value,sub_tree in tree['childs'].items():
                    indicies = datas[:,tree['best_feat']] == value
                    sub_datas = np.hstack((datas[indicies,:tree['best_feat']],datas[indicies,tree['best_feat']+1:]))
                    c_tb += self.prune(sub_tree, sub_datas, targets[indicies], alpha=alpha)


                if c_ta <= c_tb:    # 将该节点设为叶节点后损失函数变小
                    tree['is_leaf'] = True
                    tree['label'] = self._majority_vote(targets)
                    return c_ta
                else:
                    return c_tb
        else:
            pass


    def _cal_entropy(self, targets):
        '''
        计算信息熵
        '''
        _,counts =  np.unique(targets, return_counts=True)
        probs = counts / targets.shape[0]
        entropy = -(probs * np.log2(probs)).sum()
        return entropy

    def _is_all_attr_unique(self, datas):
        '''
        判断是否所有样本在所有属性上取值相同
        '''
        for i in range(datas.shape[1]):
            if len(np.unique(datas[:,i])) > 1:
                return False
        
        return True

    def _majority_vote(self, targets):
        '''
        多数投票
        '''
        if len(targets) == 0:
            return 
        uniques,counts = np.unique(targets, return_counts=True)
        return uniques[np.argmax(counts)]


if __name__ == '__main__':

    _type = 'C4.5'  # 决策树类别 ID3,C4.5,CART

    # 加载sklearn自带的手写数字识别数据集
    digits = load_digits()
    features = digits.data
    targets = digits.target
    targets = (digits.target > 4).astype(int)   # 0-4 设为标签0 5-9 设为标签1
    features = (features > 7).astype(int)   # 特征值0-7设为0，8-16设为1 
    
    np.random.seed(2020)
    # 随机打乱数据
    shuffle_indices = np.random.permutation(features.shape[0])
    features = features[shuffle_indices]
    targets = targets[shuffle_indices]

    # 划分训练、测试集
    train_count = int(len(features)*0.8)
    train_datas, train_targets = features[:train_count], targets[:train_count]
    test_datas, test_targets = features[train_count:], targets[train_count:]

    # 指定每个属性的类别，0代表离散属性，1代表连续属性
    attr_type = [0] * train_datas.shape[1]

    decision_tree = DecisionTree(_type=_type)

    decision_tree.tree = decision_tree.build_tree(train_datas, train_targets, attr_type)

    decision_tree.test(test_datas, test_targets)

    # 剪枝
    decision_tree.prune(decision_tree.tree, train_datas, train_targets, alpha=10)

    decision_tree.test(test_datas, test_targets)




    

    




    