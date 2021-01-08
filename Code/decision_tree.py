"""
决策树
ID3 C4.5 CART
"""

import numpy as np
from sklearn.datasets import load_digits    # 手写数字类别预测，用于分类任务
from sklearn.datasets import load_boston    # 波士顿房价预测，用于回归任务


class DecisionTree(object):

    def __init__(self, _type='ID3', predict_type='classification', split_count=10,\
                 max_depth=float('inf'), max_features=-1) -> None:
        '''
        _type 决策树类型
        predict_type 预测类型 classification 分类 regression 回归
        split_count 对于连续属性切分的次数
        max_depth 决策树最大深度
        max_features 每一次决策树分裂时随机所选属性数目，方便随机森林随机选择特征进行分裂
        '''
        super().__init__()

        assert _type in ['ID3','C4.5','CART']
        assert predict_type in ['classification','regression']

        self.type = _type
        self.predict_type = predict_type
        self.split_count = split_count
        self.max_depth = max_depth
        self.max_features = max_features

        if _type != 'CART' and predict_type == 'regression':
            raise NotImplementedError()

        self.tree = None
        self.predict_values = None    # 保存每个训练样本的预测值，方便GBDT快速获取每个训练样本的预测值

    def build_tree(self, datas, targets, attr_type, depth, data_ids=None):
        '''
        递归构建树
        对于 ID3 C4.5 只支持所有属性均为离散类型
        attr_type   属性类型，0代表离散，1代表连续 list  (m,)
        depth 树深度
        data_ids 每个样本的id，方便在每个叶节点将每个样本的预测值保存，以便GBDT快速获取每个训练样本的预测值
        '''

        if self.predict_values is None: # 初始化训练样本的预测值
            self.predict_values = np .zeros((datas.shape[0],))
            

        m = len(attr_type)
        n = datas.shape[0]

        tree = {'is_leaf':False}
        tree['attr_type'] = attr_type   # 保存每个属性的类别方便测试的时候使用

        # 判断是否符合终止条件
        # 样本数为0、所有样本属于同一类别、每个特征都只有一种取值、树深度超过最大深度
        if len(datas) == 0 or len(np.unique(targets)) == 1 or \
            self._is_all_attr_unique(datas) or depth > self.max_depth:
            tree['is_leaf'] = True
            tree['label'] = self._majority_vote(targets)
            if data_ids is not None:
                for data_id in data_ids:
                    self.predict_values[data_id] = tree['label']
            return tree

        # 计算原数据集的信息熵
        if self.type in ['ID3','C4.5']:
            origin_entropy = self._cal_entropy(targets)
        # 保存使用每一个属性分割后带来的信息增益
        gains = []
        # 对于C4.5还需要计算信息增益率
        gain_ratios = []
        
        # CART 对于分类问题，保存分割后的最低基尼不纯度、分割特征、特征取值
        # CART 对于回归问题，保存分割后的最低平方误差、分割特征、特征阈值
        cart_split_cache = [float('inf'),-1,-1]

        # 随机选取max_features个属性
        if self.max_features != -1 and self.max_features < len(attr_type):
            sample_attr_incides = np.random.choice(len(attr_type),self.max_features, replace=False)
        else:
            sample_attr_incides = range(m)

        for i in sample_attr_incides:
            if attr_type[i] == 0:
                uniques,counts = np.unique(datas[:,i], return_counts=True)
                if self.type in ['ID3','C4.5']:
                    # 针对每一个属性计算分割后的信息熵
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
                    for j in range(uniques.shape[0]):
                        value,count = uniques[j],counts[j] 
                        if self.predict_type == 'classification':
                            # 计算使用该取值分割后的基尼不纯度
                            metric = (count/n) * self._cal_gini(targets[datas[:,i]==value]) + (1-count/n) * self._cal_gini(targets[datas[:,i]!=value])
                            
                        else:
                            # 计算使用该取值分割后的平方误差
                            metric = self._cal_square_error(targets[datas[:,i]==value]) + self._cal_square_error(targets[datas[:,i]!=value])
                        if metric < cart_split_cache[0]:
                            cart_split_cache[0] = metric
                            cart_split_cache[1] = i
                            cart_split_cache[2] = value
            else:
                if self.type == 'CART':
                    _min,_max = np.min(datas[:,i]), np.max(datas[:,i])
                    step = (_max-_min) / self.split_count    # 步长
                    # 按步长遍历该属性
                    for j in range(1,self.split_count):
                        threshold = _min + j * step
                        count = np.sum(datas[:,i]<=threshold)   # 左子树样本个数
                        if self.predict_type == 'classification':
                            # 计算使用该取值分割后的基尼不纯度
                            metric = (count/n) * self._cal_gini(targets[datas[:,i]<=threshold]) + (1-count/n) * self._cal_gini(targets[datas[:,i]>threshold])
                        else:
                            # 计算使用该取值分割后的平方误差
                            metric = self._cal_square_error(targets[datas[:,i]<=threshold]) + self._cal_square_error(targets[datas[:,i]>threshold])
                        if metric < cart_split_cache[0]:
                            cart_split_cache[0] = metric
                            cart_split_cache[1] = i
                            cart_split_cache[2] = threshold
                else:
                    raise NotImplementedError()
        
        if self.type in ['ID3','C4.5']:
            if self.type == 'ID3':
                best_feat = sample_attr_incides[np.argmax(gains)]    # 挑选信息增益最大的特征切分
            else:
                # 先计算平均信息增益
                mean_gain = np.mean(gains)
                best_gain_ratio = -1
                best_feat = -1
                for index,(gain,gain_ratio) in enumerate(zip(gains, gain_ratios)):
                    # 在信息增益大于均值的特征中挑选信息增益率最高的特征切分
                    if gain >= mean_gain and gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_feat = sample_attr_incides[index]
            tree['best_feat'] = best_feat
            tree['childs'] = {}
            uniques = np.unique(datas[:,best_feat])
            for value in uniques:
                indicies = datas[:,best_feat] == value
                sub_datas = np.hstack((datas[indicies,:best_feat],datas[indicies,best_feat+1:]))
                sub_data_ids = None
                if data_ids is not None:
                    sub_data_ids = data_ids[indicies]
                tree['childs'][value] = self.build_tree(sub_datas,targets[indicies],attr_type[:best_feat]+attr_type[best_feat+1:],\
                                depth+1,sub_data_ids)
        else:
            tree['best_feat'] = cart_split_cache[1:]
            tree['childs'] = {}

            if attr_type[cart_split_cache[1]] == 0: # 对于离散属性，将等于选取特征最佳取值的样本分到左子树，不等于最佳取值的样本分到右子树
                indicies = datas[:,cart_split_cache[1]] == cart_split_cache[2]
                sub_datas = np.hstack((datas[indicies,:cart_split_cache[1]],datas[indicies,cart_split_cache[1]+1:]))
                sub_data_ids_left = None
                sub_data_ids_right = None
                if data_ids is not None:
                    sub_data_ids_left = data_ids[indicies]
                    sub_data_ids_right = data_ids[~indicies]
                tree['childs']['left'] = self.build_tree(sub_datas,targets[indicies],\
                    attr_type[:cart_split_cache[1]]+attr_type[cart_split_cache[1]+1:],depth+1,sub_data_ids_left)
                tree['childs']['right'] = self.build_tree(datas[~indicies],targets[~indicies],attr_type,\
                                depth+1,sub_data_ids_right)
            else:   # 对于连续属性，将小于选取特征阈值的样本分到左子树，大于选取特征阈值的样本分到右子树
                indicies = datas[:,cart_split_cache[1]] <= cart_split_cache[2]
                sub_data_ids_left = None
                sub_data_ids_right = None
                if data_ids is not None:
                    sub_data_ids_left = data_ids[indicies]
                    sub_data_ids_right = data_ids[~indicies]
                tree['childs']['left'] = self.build_tree(datas[indicies],targets[indicies],attr_type,depth+1,sub_data_ids_left)
                tree['childs']['right'] = self.build_tree(datas[~indicies],targets[~indicies],attr_type,depth+1,sub_data_ids_right)

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
        if self.predict_type == 'classification':   # 分类任务计算准确率
            accuracy = (predict_targets == targets).mean()
            print('Accuracy:%.4f'%accuracy)
        else:   # 回归任务计算均方误差
            mse = ((predict_targets - targets) ** 2).mean()
            print('MSE:%.4f'%mse)

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
        else:
            best_feat,value = tree['best_feat']
            if tree['attr_type'][best_feat] == 0: # 对于离散属性，将等于选取特征最佳取值的样本分到左子树，不等于最佳取值的样本分到右子树
                if data[best_feat] == value:
                    return self.predict(tree['childs']['left'],np.hstack((data[:best_feat],data[best_feat+1:])))
                else:
                    return self.predict(tree['childs']['right'],data)
            else:   # 对于连续属性，将小于选取特征阈值的样本分到左子树，大于选取特征阈值的样本分到右子树
                if data[best_feat] <= value:
                    return self.predict(tree['childs']['left'],data)
                else:
                    return self.predict(tree['childs']['right'],data)


    def _cal_entropy(self, targets):
        '''
        计算信息熵
        '''
        _,counts =  np.unique(targets, return_counts=True)
        probs = counts / targets.shape[0]
        entropy = -(probs * np.log2(probs)).sum()
        return entropy

    def _cal_gini(self, targets):
        '''
        计算基尼不纯度
        '''
        _,counts =  np.unique(targets, return_counts=True)
        probs = counts / targets.shape[0]
        gini = 1 - np.sum(probs ** 2)
        return gini

    def _cal_square_error(self, targets):
        '''
        计算平方误差
        '''
        if len(targets) == 0:
            return 0
        mean = np.mean(targets)
        square_error = np.sum((targets - mean) ** 2)
        return square_error

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
        对于分类采用多数投票
        对于回归采用均值
        '''
        if len(targets) == 0:
            return 
        if self.predict_type == 'classification':
            uniques,counts = np.unique(targets, return_counts=True)
            return uniques[np.argmax(counts)]
        else:
            return np.mean(targets)

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

    decision_tree = DecisionTree(_type=_type, predict_type=predict_type, max_depth=3)

    decision_tree.tree = decision_tree.build_tree(train_datas, train_targets, attr_type, 0)

    decision_tree.test(test_datas, test_targets)




    

    




    