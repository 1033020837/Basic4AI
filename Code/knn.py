"""
KNN
分别实现线性搜索以及kd树搜索
"""

import numpy as np
from sklearn.datasets import load_digits
from tqdm import tqdm
import heapq

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
    '''

    def __init__(self, datas):
        '''
        datas 特征以及标签，最后一列是标签值，方便最后进行分类
        '''
        self.root = self.build_tree(datas,0,datas.shape[1]-1)   # 根节点

    def build_tree(self, datas, cur_dim, max_dim):
        '''
        构建kd树
        '''
        if len(datas) == 0:
            return
        datas = datas[np.argsort(datas[:,cur_dim])]
        mid = datas.shape[0] // 2
        return Node(datas[mid], self.build_tree(datas[:mid], (cur_dim+1)%max_dim, max_dim),\
                                self.build_tree(datas[mid+1:], (cur_dim+1)%max_dim, max_dim), cur_dim)

    def predict(self, x, k, lp_distance):
        '''
        使用kd树进行预测
        '''
        top_k = [(-np.inf,None)] * k

        # 递归访问节点
        def visit(node):
            if node is None:
                return
            dis_with_axis = x[node.dim] - node.data[node.dim]
            visit(node.left if dis_with_axis < 0 else node.right)

            dis_with_node = lp_distance(x.reshape((1,-1)),node.data.reshape((1,-1))[:,:-1])[0]
            heapq.heappushpop(top_k,(-dis_with_node,node.data[-1]))

            if -top_k[0][0] > abs(dis_with_axis):
                visit(node.right if dis_with_axis < 0 else node.left)

        visit(self.root)

        top_k = [int(x[1]) for x in heapq.nlargest(k, top_k)]

        return top_k

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
        线性搜索测试
        '''
        predict_ys = []
        print('Testing.')
        for test_x in tqdm(test_xs):
            dis = self.lp_distance(self.train_xs,test_x.reshape((1,-1)))
            top_k_index = dis.argsort()[:self.k]
            top_k = [self.train_ys[index] for index in top_k_index]
            predict_y = self.vote(top_k)
            predict_ys.append(predict_y)
        predict_ys = np.array(predict_ys)
        return predict_ys

    def kdtree_test(self, test_xs):
        '''
        kd树搜索测试
        '''
        if self.kdtree is None:
            self.kdtree = KDTree(np.concatenate([self.train_xs,self.train_ys.reshape((-1,1))],-1))

        predict_ys = []
        for test_x in tqdm(test_xs):
            top_k = self.kdtree.predict(test_x, self.k, self.lp_distance)
            predict_y = self.vote(top_k)
            predict_ys.append(predict_y)
        predict_ys = np.array(predict_ys)

        return predict_ys

    def vote(self, top_k):
        '''
        多数表决
        '''
        count = {}
        max_freq, predict_y = 0,0
        for key in top_k:
            
            count[key] = count.get(key,0) + 1

            if count[key] > max_freq:
                max_freq = count[key]
                predict_y = key
        return predict_y

if __name__ == '__main__':

    # 加载sklearn自带的手写数字识别数据集
    digits = load_digits()
    features = digits.data
    targets = digits.target
    shuffle_indices = np.random.permutation(features.shape[0])
    features = features[shuffle_indices]
    targets = targets[shuffle_indices]

    # 划分训练、测试集
    train_count = int(len(features)*0.8)
    train_xs, train_ys = features[:train_count], targets[:train_count]
    test_xs, test_ys = features[train_count:], targets[train_count:]

    k = 5
    p = 2
    knn = KNN(k,train_xs, train_ys, p)

    # predict_ys = knn.liner_test(test_xs)
    predict_ys = knn.kdtree_test(test_xs)        


    accuracy = (predict_ys == test_ys).sum() / test_ys.shape[0]

    print('Accuracy:%.4f'%accuracy)


