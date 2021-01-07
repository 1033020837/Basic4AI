"""
朴素贝叶斯文本分类
实现了多项式朴素贝叶斯以及伯努利朴素贝叶斯
"""

import numpy as np
from data_util import load_text_cla_corpus  # 加载文本分类数据的函数
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.model_selection import train_test_split  # 划分测试集和训练集

class NaiveBayes(object):

    def __init__(self, _type='poly', _lambda=1) -> None:
        '''
        _type ['poly','bernoulli'] 多项式朴素贝叶斯、伯努利朴素贝叶斯
        _lambda 平滑因子
        '''
        super().__init__()

        assert _type in ['poly','bernoulli']

        self.type = _type
        self._lambda = _lambda

    def train(self, train_xs, train_ys):
        '''
        训练   
        计算先验概率和似然概率
        '''
        
        # 对于伯努利朴素贝叶斯，将特征矩阵转换为0-1矩阵
        if self.type == 'bernoulli':
            train_xs = (train_xs > 0).astype(int)
        
        # 样本数、特征数
        n,m = train_xs.shape

        # 统计label数目
        unique_ys,counts = np.unique(train_ys,return_counts=True)
        self.unique_ys = unique_ys
        # 先验概率
        denominator = n + self._lambda * len(unique_ys)  # 先验概率的分母
        self.prior_probs = np.log2((counts+self._lambda)/denominator)

        # 似然概率
        self.likelihood_probs = np.zeros((m,len(unique_ys)))
        if self.type == 'bernoulli':    # 对于伯努利朴素贝叶斯，后验概率为 包含词w且属于类别c的样本/属于类别c的样本
            for i,y in enumerate(unique_ys):
                sub_xs = train_xs[train_ys==y]
                self.likelihood_probs[:,i] = (np.sum(sub_xs,axis=0) + self._lambda) \
                                            / (counts[i] + 2 * self._lambda)
            
        else:   # 对于多项式朴素贝叶斯，后验概率为 词w在属于类别c的样本中出现的次数/属于类别c的样本的总词数
            for i,y in enumerate(unique_ys):
                sub_xs = train_xs[train_ys==y]
                self.likelihood_probs[:,i] = (np.sum(sub_xs,axis=0) + self._lambda) \
                                            / (np.sum(sub_xs) + m * self._lambda)
        
        # 因为伯努利朴素贝叶斯在预测阶段还需要计算 不包含词w且属于类别c的概率，所以先进行计算并取对数存储
        if self.type == 'bernoulli':
            self.negative_likelihood_probs = np.log2(1-self.likelihood_probs)
        self.likelihood_probs = np.log2(self.likelihood_probs)

    def test(self, test_xs, test_ys):
        '''
        测试函数
        '''
        predict_ys = []
        for i in range(test_xs.shape[0]):
            predict_target = self.predict(test_xs[i])
            predict_ys.append(predict_target)
        predict_ys = np.array(predict_ys)
        accuracy = (predict_ys == test_ys).mean()
        print('Accuracy:%.4f'%accuracy)

    def predict(self, x):
        '''
        预测一个样本的类别
        '''
        x = x.reshape((1,-1))
        if self.type == 'bernoulli':
            # 伯努利朴素贝叶斯需要考虑未出现词的概率
            log_probs = np.dot(x, self.likelihood_probs) + np.dot(1-x, self.negative_likelihood_probs)
        else:
            log_probs = np.dot(x, self.likelihood_probs)
        log_probs = log_probs.reshape((-1,)) + self.prior_probs
        return self.unique_ys[np.argmax(log_probs)]

if __name__ == '__main__':
    _type = 'poly' # ['poly','bernoulli'] 多项式朴素贝叶斯、伯努利朴素贝叶斯

    texts, labels = load_text_cla_corpus('../Data/TextClassification/datasets.tsv')
    train_texts, test_texts, train_ys, test_ys = train_test_split(texts, labels, \
                    train_size=0.8, random_state=2021)

    # 将文本转换为代表出现频率的数值特征
    vectorizer = CountVectorizer(binary=True)   
    train_xs = vectorizer.fit_transform(train_texts).toarray()
    test_xs = vectorizer.transform(test_texts).toarray()

    naive_bayes = NaiveBayes(_type)

    naive_bayes.train(train_xs, train_ys)
    naive_bayes.test(test_xs, test_ys)

