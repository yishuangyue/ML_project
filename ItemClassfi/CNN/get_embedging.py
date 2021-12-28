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
import logging as logger
from collections import defaultdict
from gensim.models.keyedvectors import load_word2vec_format
import numpy as np

def get_embedding(embed_path, token_key, embed_dim):
    ''' 读取词向量

    Args:
        embed_path    : embedding文件路径
        token_key : [dict：id] 词id嵌入字典
        freq_threshold: [int]词频最低阈值，低于此阈值的词不会进行词向量抽取
        embed_dim     : [int]词向量维度
        token_counter : 【词集合set】

    Returns:
        embed_mat: [ListOfList]嵌入矩阵
        not_cnt:没有查到的词有多少
    '''
    embed_dict = {}
    not_cnt=0

    with codecs.open(embed_path, 'r', 'utf-8') as infs:
        # 从第二行开始读
        for inf in infs.readlines()[1:]:
            inf = inf.strip()
            inf_list = inf.split(" ")
            token = ''.join(inf_list[0:-embed_dim])

            if token in token_key.keys():
                embed_dict[token] = list(map(float, inf_list[-embed_dim:]))

    print("{} / {} tokens have corresponding embedding vector".format(len(embed_dict)))

    unk = "<unk>"
    pad = "<pad>"
    # enumerate(iterable, start=0)，start代指起始idx（不影响token输出）
    # token2id = {token: idx for idx, token in enumerate(embed_dict.keys(), 2)}
    # token2id = token_key
    # token2id[unk] = 0
    # token2id[pad] = 1
    # embed_dict[unk] = [0. for _ in range(embed_dim)]
    # embed_dict[pad] = [0. for _ in range(embed_dim)]
    # 循环原始词，对每个词都上一个词向量,对没有找到embedding的用0处理
    for token in token_key.keys():
        if token not in embed_dict.keys():
            embed_dict[token]= [0. for _ in range(embed_dim)]
            not_cnt+=1




    id2embed = {idx: embed_dict[token] for token, idx in token_key.items()}

    embed_mat = [id2embed[idx] for idx in range(len(id2embed))]

    return  embed_mat,not_cnt












