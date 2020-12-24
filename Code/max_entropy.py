'''
最大熵模型
主要参考 李航《统计学习方法》 以及 https://www.pkudodo.com/2018/12/05/1-7/
'''


import time
import numpy as np
from collections import defaultdict
from sklearn.datasets import load_digits
from tqdm import tqdm


def load_data():
    '''
    加载sklearn自带的手写数字识别数据集
    返回 输入、输出
    '''

    digits = load_digits()
    xs = digits.data.tolist()
    ys = (digits.target > 4).astype(int).tolist()
    return xs, ys


class MaxEntropy:
    '''
    最大熵类
    '''

    def __init__(self, train_xs, train_ys, test_xs, test_ys):
        '''
        各参数初始化
        '''
        self.train_xs = train_xs                        # 训练数据集
        self.train_ys = train_ys                        # 训练标签集
        self.test_xs = test_xs                        # 训练数据集
        self.test_ys = test_ys                        # 训练标签集
        self.class_count = len(set(self.test_ys))      # 标签取值数量
        self.m = len(train_xs[0])  # 原始输入特征的数量，需要跟特征函数的数量区分开
        self.N = len(train_xs)  # 训练样本数目
        self.features, self.feature_count = self.get_features()  # 所有特征   特征函数数量
        self.M = self.m                    # 假定任意样本中所有特征函数的和是固定值，简化IIS算法
        self.w = [0] * self.feature_count  # 所有特征的权重
        self.xy2id, self.id2xy = self.createSearchDict()  # 特征->id、id->特征 的对应字典
        self.Ep_xy = self.get_Ep_xy()  # 特征函数f(x, y)关于经验分布P_(x, y)的期望值

    def get_Epxy(self):
        '''
        计算特征函数f(x, y)关于模型P(Y|X)与经验分布P_(X, Y)的期望值
        即“6.2.2 最大熵模型的定义”中第二个期望（83页最上方的期望）
        :return:
        '''
        # 初始化期望存放列表，对于每一个xy对都有一个期望
        # 这里的x是单个的特征，不是一个样本的全部特征。例如x={x1，x2，x3.....，xk}，实际上是（x1，y），（x2，y），。。。
        # 但是在存放过程中需要将不同特诊的分开存放，李航的书可能是为了公式的泛化性高一点，所以没有对这部分提及
        # 具体可以看我的博客，里面有详细介绍  www.pkudodo.com
        Epxy = [0] * self.feature_count
        # 对于每一个样本进行遍历
        for i in range(self.N):

            # 初始化公式中的P(y|x)列表
            Pwxy = self.calcPwy_x(self.train_xs[i])

            for feature in range(self.m):
                for y in range(self.class_count):
                    if (self.train_xs[i][feature], y) in self.features[feature]:
                        id = self.xy2id[feature][(
                            self.train_xs[i][feature], y)]
                        Epxy[id] += (1 / self.N) * Pwxy[y]
        return Epxy

    def get_Ep_xy(self):
        '''
        计算特征函数f(x, y)关于经验分布P_(x, y)的期望值（下划线表示P上方的横线，
        同理Ep_xy中的“_”也表示p上方的横线）
        即“6.2.2 最大熵的定义”中第一个期望（82页最下方那个式子）
        :return: 计算得到的Ep_xy
        '''
        # 初始化Ep_xy列表，长度为n
        Ep_xy = [0] * self.feature_count
        # 遍历每一个特征
        for feature in range(self.m):
            # 遍历每个特征中的(x, y)对
            for (x, y) in self.features[feature]:
                # 获得其id
                id = self.xy2id[feature][(x, y)]
                # 将计算得到的Ep_xy写入对应的位置中
                # fixy中存放所有对在训练集中出现过的次数，处于训练集总长度N就是概率了
                Ep_xy[id] = self.features[feature][(x, y)] / self.N
        # 返回期望
        return Ep_xy

    def createSearchDict(self):
        '''
        创建查询字典
        xy2idDict：通过(x,y)对找到其id,所有出现过的xy对都有一个id
        id2xyDict：通过id找到对应的(x,y)对
        '''
        # 设置xy搜多id字典
        # 这里的x指的是单个的特征，而不是某个样本，因此将特征存入字典时也需要存入这是第几个特征
        # 这一信息，这是为了后续的方便，否则会乱套。
        # 比如说一个样本X = (0, 1, 1) label =(1)
        # 生成的标签对有(0, 1), (1, 1), (1, 1)，三个(x，y)对并不能判断属于哪个特征的，后续就没法往下写
        # 不可能通过(1, 1)就能找到对应的id，因为对于(1, 1),字典中有多重映射
        # 所以在生成字典的时总共生成了特征数个字典，例如在mnist中样本有784维特征，所以生成784个字典，属于
        # 不同特征的xy存入不同特征内的字典中，使其不会混淆
        xy2idDict = [{} for i in range(self.m)]
        # 初始化id到xy对的字典。因为id与(x，y)的指向是唯一的，所以可以使用一个字典
        id2xyDict = {}
        # 设置缩影，其实就是最后的id
        index = 0
        # 对特征进行遍历
        for feature in range(self.m):
            # 对出现过的每一个(x, y)对进行遍历
            # fixy：内部存放特征数目个字典，对于遍历的每一个特征，单独读取对应字典内的(x, y)对
            for (x, y) in self.features[feature]:
                # 将该(x, y)对存入字典中，要注意存入时通过[feature]指定了存入哪个特征内部的字典
                # 同时将index作为该对的id号
                xy2idDict[feature][(x, y)] = index
                # 同时在id->xy字典中写入id号，val为(x, y)对
                id2xyDict[index] = (x, y)
                # id加一
                index += 1
        # 返回创建的两个字典
        return xy2idDict, id2xyDict

    def get_features(self):
        '''
        根据训练集统计所有特征以及总的特征的数量
        :return:
        '''
        n = 0
        # 建立特征数目个字典，属于不同特征的(x, y)对存入不同的字典中，保证不被混淆
        fixyDict = [defaultdict(int) for i in range(self.m)]
        # 遍历训练集中所有样本
        for i in range(len(self.train_xs)):
            # 遍历样本中所有特征
            for j in range(self.m):
                # 将出现过的(x, y)对放入字典中并计数值加1
                fixyDict[j][(self.train_xs[i][j],
                             self.train_ys[i])] += 1
        # 对整个大字典进行计数，判断去重后还有多少(x, y)对，写入n
        for i in fixyDict:
            n += len(i)
        # 返回大字典
        return fixyDict, n

    def calcPwy_x(self, x):
        '''
        计算“6.23 最大熵模型的学习” 式6.22
        :param X: 要计算的样本X（一个包含全部特征的样本）
        :param y: 该样本的标签
        :return: 计算得到的Pw(Y|X)
        '''
        # 分子
        numerators = [0] * self.class_count

        # 对每个特征进行遍历
        for i in range(self.m):
            for j in range(self.class_count):
                if (x[i], j) in self.xy2id[i]:
                    index = self.xy2id[i][(x[i], j)]
                    numerators[j] += self.w[index]

        # 计算分子的指数
        numerators = np.exp(numerators)
        # 计算分母的z
        Z = np.sum(numerators)
        # 返回Pw(y|x)
        res = numerators / Z
        return res

    def iis_train(self, iter=200):
        # 使用iis进行训练
        for i in tqdm(range(iter)):

            # 计算“6.2.3 最大熵模型的学习”中的第二个期望（83页最上方哪个）
            Epxy = self.get_Epxy()
            # 使用的是IIS，所以设置sigma列表
            sigmaList = [0] * self.feature_count
            # 对于所有的n进行一次遍历
            
            for j in range(self.feature_count):
                # 依据“6.3.1 改进的迭代尺度法” 式6.34计算
                sigmaList[j] = (1 / self.M) * np.log(self.Ep_xy[j] / Epxy[j])
            # 按照算法6.1步骤二中的（b）更新w
            self.w = [self.w[i] + sigmaList[i] for i in range(self.feature_count)]

            if (i+1) % 5 == 0:
                accuracy = self.test()
                print('the accuracy is:%.4f' % accuracy)


    def predict(self, X):
        '''
        预测标签
        :param X:要预测的样本
        :return: 预测值
        '''
        return np.argmax(self.calcPwy_x(X))

    def test(self):
        '''
        对测试集进行测试
        :return:
        '''
        # 错误值计数
        errorCnt = 0
        # 对测试集中所有样本进行遍历
        for i in range(len(self.test_xs)):
            # 预测该样本对应的标签
            result = self.predict(self.test_xs[i])
            # 如果错误，计数值加1
            if result != self.test_ys[i]:
                errorCnt += 1
        # 返回准确率
        return 1 - errorCnt / len(self.test_xs)


if __name__ == '__main__':
    features, targets = load_data()
    

    train_count = int(len(features)*0.8)

    train_xs, train_ys = features[:train_count], targets[:train_count]
    test_xs, test_ys = features[train_count:], targets[train_count:]

    # 初始化最大熵类
    maxEnt = MaxEntropy(train_xs, train_ys, test_xs, test_ys)
    # 开始训练
    print('start to train')
    maxEnt.iis_train()
    # 开始测试
    print('start to test')
    accuracy = maxEnt.test()    # 200轮准确率为86.39%
    print('the accuracy is:%.4f'%accuracy)
