"""
提供一些数据处理的功能
"""

from codecs import open
import pandas as pd
import re

def load_seq_label_corpus(path, make_vocab=True):
    '''
    读取序列标注的数据
    参考自：https://github.com/luopeixiang/named_entity_recognition/blob/master/data.py
    '''

    word_lists = []
    tag_lists = []
    with open(path, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\r\n':
                word, tag = line.split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps

def load_text_cla_corpus(path):
    '''
    读取文本分类数据集
    '''
    df = pd.read_csv(path, sep='\t')
    texts, labels = [],[] # 文本和对应类别
    for line in df.itertuples():
        text = re.sub(r"([,.!?])", r" \1 ", line.text)
        text = re.sub(" {2,}", " ", text)
        texts.append(text)
        labels.append(int(line.target))
    return texts, labels