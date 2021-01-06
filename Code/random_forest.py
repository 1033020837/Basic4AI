"""
随机森林
"""

from decision_tree import DecisionTree
import numpy as np
import threading
from sklearn.datasets import load_digits    # 手写数字类别预测，用于分类任务
from sklearn.datasets import load_boston    # 波士顿房价预测，用于回归任务


class RandomForest(object):

    def __init__(self, tree_count = 50, attr_ratio = 0.5, _type='CART', predict_type='classification', split_count=10, thread_count=5) -> None:
        '''
        tree_count 决策树数量
        attr_ratio 每一刻决策树所选属性数目占总属性数目的比例
        _type 决策树类型
        predict_type 预测类型 classification 分类 regression 回归
        split_count 对于连续属性切分的次数
        process_count 建立随机森林的进程数
        '''
        super().__init__()

        assert _type in ['ID3','C4.5','CART']
        assert predict_type in ['classification','regression']

        self.tree_count = tree_count
        self.attr_ratio = attr_ratio
        self.type = _type
        self.predict_type = predict_type
        self.split_count = split_count
        self.thread_count = min(thread_count, tree_count)

        if _type != 'CART' and predict_type == 'regression':
            raise NotImplementedError()

    def train(self, datas, targets, attr_type):
        '''
        训练函数
        多线程构建森林
        构建 tree_count 棵决策树
        '''
        self.trees = [] 
        per_count = self.tree_count // self.thread_count    # 每个进程分配的树的数目
        threads = []
        for i in range(self.thread_count):
            # 保证最后树的总数是 self.tree_count
            if i == (self.thread_count - 1):
                tree_count = self.tree_count - (self.thread_count - 1) * per_count
            else:
                tree_count = per_count
            thread = threading.Thread(target=self.train_process, args=(datas, targets, attr_type, tree_count))
            threads.append(thread)
        for thread in threads:
            thread.start()
            thread.join()

    def train_process(self, datas, targets, attr_type, tree_count):
        '''
        单线程训练函数
        构建一部分树
        tree_count 本进程要构建的树的数目
        '''
        for _ in range(tree_count):
            # 随机有放回挑选与数据集样本数目相同的样本
            sample_incides = np.random.choice(datas.shape[0],datas.shape[0])
            sample_datas, sample_targets = datas[sample_incides], targets[sample_incides]
            # 随机选取attr_ratio比例的属性
            sample_attr_incides = np.random.choice(len(attr_type),int(len(attr_type) * self.attr_ratio), replace=False)
            sample_attr_incides = sorted(sample_attr_incides)
            sample_datas = sample_datas[:,sample_attr_incides]
            sample_attr_type = [x for i,x in enumerate(attr_type) if i in sample_attr_incides]
            tree = DecisionTree(self.type, self.predict_type, self.split_count)
            tree.tree = tree.build_tree(sample_datas, sample_targets, sample_attr_type)
            self.trees.append([tree,sample_attr_incides]) # 保存树以及所选属性

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
        predict_targets = []    # 所有树的投票结果
        for tree, sample_attr_incides in self.trees:
            predict_target = tree.predict(tree.tree, data[sample_attr_incides])
            predict_targets.append(predict_target)
        
        # 投票或者取平均
        if self.predict_type == 'classification':
            uniques,counts = np.unique(predict_targets, return_counts=True)
            return uniques[np.argmax(counts)]
        else:
            return np.mean(predict_targets)

if __name__ == '__main__':

    _type = 'CART'  # 决策树类别 ID3,C4.5,CART
    predict_type = 'regression' # 分类还是回归任务  classification 分类 regression 回归

    if predict_type == 'classification':
        # 加载sklearn自带的手写数字识别数据集
        digits = load_digits()
        features = digits.data
        targets = digits.target
        targets = (digits.target > 4).astype(int)   # 0-4 设为标签0 5-9 设为标签1
        # 因为没做缺失值处理，所以对于ID3和C4.5，测试时可能存在未知路径的情况
        # 简单起见，对于ID3和C4.5，将特征值0-7设为0，8-16设为1 
        if _type != 'CART':
            features = (features > 7).astype(int)   
    else:
        boston = load_boston()
        features = boston.data
        targets = boston.target

    np.random.seed(2021)
    # 随机打乱数据
    shuffle_indices = np.random.permutation(features.shape[0])
    features = features[shuffle_indices]
    targets = targets[shuffle_indices]

    # 划分训练、测试集
    train_count = int(len(features)*0.8)
    train_datas, train_targets = features[:train_count], targets[:train_count]
    test_datas, test_targets = features[train_count:], targets[train_count:]

    # 指定每个属性的类别，0代表离散属性，1代表连续属性
    if _type != 'CART': # ID3和C4.5只实现了处理离散属性
        attr_type = [0] * train_datas.shape[1]
    else:   # CART既可以处理离散属性，也可以处理连续属性
        attr_type = [1] * train_datas.shape[1]

    tree_count = 20
    attr_ratio = 0.5
    random_forest = RandomForest(tree_count=tree_count, attr_ratio=attr_ratio, _type=_type, predict_type=predict_type)
    random_forest.train(train_datas, train_targets, attr_type)
    random_forest.test(test_datas, test_targets)
