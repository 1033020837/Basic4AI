"""
使用隐马尔可夫模型进行序列标注
数据集为人民日报语料集，实体为人名、地名、组织机构名
数据集使用BIO格式进行标注
"""

from data_util import load_seq_label_corpus
import numpy as np


class HMM(object):


    def __init__(self, o_count, h_count, word2id, tag2id) -> None:
        '''
        o_count 观测状态数量
        h_count 隐藏状态数量
        word2id 词->索引
        tag2id  标签->索引
        '''
        super().__init__()

        self.PI = np.zeros((h_count,))  # 初始状态概率矩阵
        self.A = np.zeros((h_count,h_count))  # 状态转移概率矩阵
        self.B = np.zeros((h_count,o_count)) # 观测概率矩阵
        self.word2id = word2id
        self.tag2id = tag2id
        self.id2tag = {v:k for k,v in self.tag2id.items()}
        self.h_count = h_count

    def train(self, train_word_lists, train_tag_lists):
        '''
        使用训练语料计算三要素
        '''
        for words,tags in zip(train_word_lists, train_tag_lists):
            assert len(words) == len(tags)
            pre_tag_id = -1
            for index,(word,tag) in enumerate(zip(words, tags)):
                word_id,tag_id = self.word2id[word], self.tag2id[tag]

                if index == 0:
                    self.PI[tag_id] += 1
                
                self.B[tag_id,word_id] += 1

                if index > 0:
                    self.A[pre_tag_id,tag_id] += 1
                
                pre_tag_id = tag_id
        
        self.A[self.A == 0.] = 1e-6
        self.B[self.B == 0.] = 1e-6
        self.PI[self.PI == 0.] = 1e-6
        self.PI = np.log(self.PI / np.sum(self.PI))
        self.A = np.log(self.A / np.sum(self.A, axis=1, keepdims=True))
        self.B = np.log(self.B / np.sum(self.B, axis=1, keepdims=True))

    def test(self, test_word_lists, test_tag_lists):
        '''
        测试
        '''
        counter = {}
        for words,tags in zip(test_word_lists, test_tag_lists):
            pred_tags = self.predict(words)

            for gold_tag,pred_tag in zip(tags, pred_tags):
                if gold_tag not in counter:
                    counter[gold_tag] = {'tp':0,'fp':0,'fn':0}
                if gold_tag == pred_tag:
                    counter[gold_tag]['tp'] += 1
                else:
                    if pred_tag not in counter:
                        counter[pred_tag] = {'tp':0,'fp':0,'fn':0}
                    counter[gold_tag]['fn'] += 1
                    counter[pred_tag]['fp'] += 1
            
        for tag,report in counter.items():
            precision = report['tp'] / (report['tp']+report['fp'])
            recall = report['tp'] / (report['tp']+report['fn'])
            f1 = 2 * precision * recall / (precision + recall)

            print('Tag: %-5s\tprecision: %.4f\trecall: %.4f\tf1: %.4f'%(tag,precision,recall,f1))
    
    def predict(self, words):
        '''
        给定观测序列预测概率最大的隐藏序列
        words 观测词序列
        '''

        sigma = np.zeros((self.h_count,))
        psis = []

        for index,word in enumerate(words):
            if word in self.word2id:
                emmision_probs = self.B[:,self.word2id[word]]
            else:
                # 不存在的词将观测概率设为均匀分布
                emmision_probs = np.ones((self.h_count,))
            if index == 0:
                sigma += self.PI + emmision_probs
            else:
                tmp = self.A + sigma.reshape((-1,1))    # h_count * h_count
                psis.append(np.argmax(tmp,axis=0))
                sigma = np.max(tmp, axis=0) + emmision_probs
        
        # 路径回溯
        res = [np.argmax(sigma)]
        for psi in psis[::-1]:
            res.append(psi[res[-1]])

        res = [self.id2tag[x] for x in res[::-1]]
        
        return res




if __name__ == '__main__':
    data_dir = '../Data/RenMinRiBao/'
    train_word_lists, train_tag_lists, word2id, tag2id = \
        load_seq_label_corpus(data_dir + 'example.train')
    # dev_word_lists, dev_tag_lists = load_seq_label_corpus(data_dir + 'example.dev', make_vocab=False)
    test_word_lists, test_tag_lists = load_seq_label_corpus(data_dir + 'example.test', make_vocab=False)

    hmm = HMM(len(word2id), len(tag2id), word2id, tag2id)
    hmm.train(train_word_lists, train_tag_lists)
    hmm.test(test_word_lists, test_tag_lists)


