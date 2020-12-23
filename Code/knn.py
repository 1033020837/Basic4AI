"""
KNN
分别实现线性搜索以及kd树搜索
"""

import numpy as np
from sklearn.datasets import load_digits
from tqdm import tqdm


class Node(object):
    '''
    树节点
    '''
    def __init__(self, data, left, right, dim):
        self.data = data    # 节点数据
        self.left = left    # 左节点
        self.right = right  # 右节点
        self.dim = dim  # 节点对应数据维度

class KDTree(object):
    '''
    kd树
    datas 特征以及标签，最后一列是标签值，方便最后进行分类
    '''
    def __init__(self, datas):
        self.build_tree(datas,0,datas.shape[1]-1)

    def build_tree(self, datas, cur_dim, max_dim):
        if len(datas) == 0:
            return
        datas = datas[np.argsort(datas[:,cur_dim])]
        mid = datas.shape[0] // 2
        return Node(datas[mid], self.build_tree(datas[:mid], (cur_dim+1)%max_dim, max_dim),\
                                self.build_tree(datas[mid+1:], (cur_dim+1)%max_dim, max_dim), cur_dim)

    def predict(self, x):


class KNN(object):

    def __init__(self, k, train_xs, train_ys, p=2):
        '''
        k KNN的k值
        train_xs np.array 训练集样本特征
        train_ys np.array 训练集样本标签
        p Lp 距离的p值
        '''
        self.k = k

        self.train_xs = train_xs
        self.train_ys = train_ys
        self.p = p

        self.kdtree = None

    def lp_distance(self, x1, x2):
        '''
        Lp距离
        '''
        dis = np.sum(np.abs(x1 - x2)**self.p,-1)**(1/self.p)
        return dis
    
    def liner_test(self, test_xs):
        '''
        线性搜索预测
        '''
        predict_ys = []
        print('Testing.')
        for test_x in tqdm(test_xs):
            dis = self.lp_distance(self.train_xs,test_x.reshape((1,-1)))
            top_k = dis.argsort()[:self.k]
            count = {}
            max_freq, predict_y = 0,0
            for index in top_k:
                key = self.train_ys[index]
                count[key] = count.get(key,0) + 1

                if count[key] > max_freq:
                    max_freq = count[key]
                    predict_y = key
            predict_ys.append(predict_y)
        predict_ys = np.array(predict_ys)
        return predict_ys

    def kdtree_test(self, test_xs):
        if self.kdtree is None:
            self.kdtree = KDTree(np.concatenate([self.train_xs,self.train_ys.reshape((-1,1))],-1))


if __name__ == '__main__':
    digits = load_digits()
    features = digits.data
    targets = digits.target
    shuffle_indices = np.random.permutation(features.shape[0])
    features = features[shuffle_indices]
    targets = targets[shuffle_indices]

    train_count = int(len(features)*0.8)
    train_xs, train_ys = features[:train_count], targets[:train_count]
    test_xs, test_ys = features[train_count:], targets[train_count:]

    k = 5
    p = 2
    knn = KNN(k,train_xs, train_ys, p)

    knn.kdtree_test(test_xs)

    # predict_ys = knn.liner_test(test_xs)


    # accuracy = (predict_ys == test_ys).sum() / test_ys.shape[0]

    # print('Accuracy:%.4f'%accuracy)


