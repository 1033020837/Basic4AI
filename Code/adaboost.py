"""
AdaBoost算法
"""

import numpy as np
from sklearn.datasets import load_digits
from tqdm import tqdm

def build_base_classifier(train_xs, train_ys, weights, attr_type, split_count=10):
    '''
    构建决策树桩作为基分类器,N 样本数目 m特征数目
    train_xs    训练样本特征 np.array (N,m)
    train_ys    训练样本标签 np.array (N,) {-1,+1}
    weights 样本权重 np.array (N,)
    attr_type   属性类型，0代表离散，1代表连续 list  (m,)
    split_count 对于连续属性切分的次数
    '''

    N,m = train_xs.shape

    assert len(weights) == N
    assert len(attr_type) == m

    min_error = float('inf')    # 最小错误率
    attr_index = -1 # 选取属性的索引
    threshold = -1  # 对于连续属性为选取属性的阈值，对于离散属性为选取属性的取值

    # 对于连续属性为选取属性哪边预测为-1，lt为左侧，gt为右侧
    # 对于离散属性为选中类别的选中取值预测为1还是-1，'eq'为选中取值预测为-1，'neq'为选中取值预测为1
    side = None   

    predict_ys = None   # 最优基分类器预测的y

    # 遍历属性
    for i in range(m):
        if attr_type[i] == 0:   # 离散属性
            uniques = np.unique(train_xs[:,i])
            for unique_value in uniques:    # 遍历每一种取值
                for ineq in ['eq','neq']:
                    _predict_ys = np.ones((N,))

                    # eq代表取该值预测为-1，neq代表不取该值预测为1
                    if ineq == 'eq':
                        _predict_ys[train_xs[:,i] == unique_value] = -1
                    else:
                        _predict_ys[train_xs[:,i] != unique_value] = -1

                    error = ((_predict_ys != train_ys) * weights).sum() # 加权错误率

                    # 如果错误率更小则选择该模型为当前最优基模型
                    if error < min_error:
                        min_error = error
                        attr_index = i
                        threshold = unique_value
                        side = ineq
                        predict_ys = _predict_ys

        else:   # 连续属性
            _min,_max = np.min(train_xs[:,i]), np.max(train_xs[:,i])
            step = (_max-_min) / split_count    # 步长

            # 按步长遍历该属性
            for j in range(split_count+1):
                _threshold = _min + j * step # 阈值
                for ineq in ['lt','gt']:   # 阈值左侧还是右侧取-1
                    _predict_ys = np.ones((N,))

                    if ineq == 'lt':
                        _predict_ys[train_xs[:,i] < _threshold] = -1
                    else:
                        _predict_ys[train_xs[:,i] >= _threshold] = -1

                    error = ((_predict_ys != train_ys) * weights).sum() # 加权错误率

                    # 如果错误率更小则选择该模型为当前最优基模型
                    if error < min_error:
                        min_error = error
                        attr_index = i
                        threshold = _threshold
                        side = ineq
                        predict_ys = _predict_ys

    return min_error, attr_index, threshold, side, predict_ys

class AdaBoost(object):
    '''
    AdaBoost类
    '''

    def __init__(self) -> None:
        super().__init__()

        self.classifiers = []   # 基分类器以及其权重
        self.attr_type = None   # 各项特征是离散还是连续

    def train(self, train_xs, train_ys, attr_type, test_xs=None, test_ys=None, \
                         base_count = 100, test_freq = 50):
        '''
        训练函数
        train_xs 训练数据特征
        train_ys 训练数据标签
        attr_type 各项特征是离散还是连续
        test_xs 测试数据特征
        test_ys 测试数据标签
        base_count 基分类器个数
        test_freq 测试频率
        '''
        self.classifiers = []
        self.attr_type = attr_type
        N,m = train_xs.shape

        weights = np.ones((N,)) / N # 初始权重为均匀分布

        for i in tqdm(range(base_count)):   # 前向训练
            error, attr_index, select_threshold, side, predict_ys = build_base_classifier(train_xs, \
                                        train_ys, weights, attr_type)

            # 根据公式更新权重
            alpha = 0.5 * np.log((1-error)/error)
            weights = weights * np.exp(-alpha*(predict_ys*train_ys))
            weights = weights / np.sum(weights)

            self.classifiers.append((attr_index, select_threshold, side, alpha))    # 添加基分类器

            # 测试
            if test_xs is not None and test_ys is not None and (i+1) % test_freq == 0:
                predict_ys = self.test(test_xs)
                accuracy = (predict_ys == test_ys).sum() / test_ys.shape[0]
                print('\nStep:%d, accuracy:%.4f'%(i+1,accuracy))

    def test(self, test_xs):
        '''
        获取测试集的预测结果
        test_xs 测试数据特征
        '''
        N = test_xs.shape[0]
        predict_ys = np.zeros((N,))

        for attr_index, threshold, side, alpha in self.classifiers:
            if self.attr_type[attr_index] == 0:
                _predict_ys = np.ones((N,))
                if side == 'eq':
                    _predict_ys[test_xs[:,attr_index] == threshold] = -1
                else:
                    _predict_ys[test_xs[:,attr_index] != threshold] = -1
                predict_ys += _predict_ys * alpha
            else:
                _predict_ys = np.ones((N,))
                if side == 'lt':
                    _predict_ys[test_xs[:,attr_index] < threshold] = -1
                else:
                    _predict_ys[test_xs[:,attr_index] >= threshold] = -1
                predict_ys += _predict_ys * alpha

        predict_ys[predict_ys>0] = 1
        predict_ys[predict_ys<0] = -1

        return predict_ys

if __name__ == '__main__':

    # 加载sklearn自带的手写数字识别数据集
    digits = load_digits()
    features = digits.data
    targets = digits.target
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

    adaboost = AdaBoost()

    attr_type = [1] * train_xs.shape[1]

    base_count = 500

    adaboost.train(train_xs, train_ys, attr_type, test_xs, test_ys, base_count=base_count)

        












