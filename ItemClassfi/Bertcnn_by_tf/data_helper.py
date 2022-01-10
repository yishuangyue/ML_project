#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/12/24 10:59 上午 
:@File : get_embedging
:Version: v.1.0
:Description:
最终返回分词映射到id的字典、词嵌入矩阵
未知（unk）和补全（pad）字符的index分别为0和1，词向量用全0表示
"""
import codecs
import json
import random

import numpy as np
import pandas as pd
from keras_bert import  Tokenizer
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

def get_data(data_path):
    """
    :param data_path: 新闻数据集目录
    :return: 测试集和训练集以及lable
    """
    # 读取数据，划分训练集和验证集

    df=pd.read_json(data_path)["data"].values.tolist()
    df=pd.DataFrame(df)  # SPBMMC_6,MC
    target_names=list(set(df["SPBMMC_6"].values))
    df = shuffle(df)   #shuffle数据
    class_le = LabelEncoder()
    df["labels"]= class_le.fit_transform(df["SPBMMC_6"].values) #将label转换为数字
    data=df[["MC","labels"]][0:100].values
    # 按照9:1的比例划分训练集和验证集
    train_data = np.array([j for i, j in enumerate(data) if i % 10 != 0])
    valid_data = np.array([j for i, j in enumerate(data) if i % 10 == 0])

    return train_data, valid_data,target_names


def get_token_dict(dict_path):
    """
    # 将词表中的词编号转换为token字典
    :param dict_path: 词路径
    :return:  返回字典{词：token_id}
    """
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict




# 导入Bert的Tokenizer并重构它
# Tokenizer自带的_tokenize会自动去掉空格，然后有些字符会粘在一块输出，导致tokenize之后的列表不等于原来字符串的长度了
# ，这样如果做序列标注的任务会很麻烦。而为了避免这种麻烦，还是自己重写一遍好了～主要就是用[unused1]来表示空格类字符，
# 而其余的不在列表的字符用[UNK]表示，其中[unused*]这些标记是未经训练的（随即初始化），是Bert预留出来用来增量添加词汇的标记，
# 所以我们可以用它们来指代任何新字符。
class OurTokenizer(Tokenizer):

    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
        return R










