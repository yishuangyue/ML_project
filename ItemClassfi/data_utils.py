#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2022/1/6 3:32 下午 
:@File : utils
:Version: v.1.0
:Description:
"""
import logging
import time,codecs
from datetime import timedelta

import torch
from torch.utils.data import Dataset
import os

def seq_padding(data, max_len):
    """
        对数据做缺失值补白，让所有输入序列长度一致
        :param X: 传入的数据
        :param max_len: 补白长度
        :param padding: 补白值
        :return: array或者是一个torch.tensor
        """
    if len(data) < max_len:
        pad_len = max_len - len(data)
        padding = [0 for _ in range(pad_len)]
        data = torch.LongTensor([data + padding])  # to ->torch
        # data=np.concatenate([data, padding])  # to->array
    else:
        data = torch.LongTensor(data[:max_len])  # to ->torch
        # data=np.array(data[:max_len]) # to->array

    return data

# 需要继承torch.utils.data.Dataset
class InputDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        """
        :param data:
        :param tokenizer:
        :param max_len:
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
         返回数据集的总大小。
         """
        return len(self.data)

    def __getitem__(self, item):
        """
        #1 从文件中读取一个数据（例如，plt.imread）。
        #2 预处理数据（例如torchvision.Transform）。
        #3 返回数据对（例如图像和标签）。
        :param item: item是索引，用来取数据
        :return:
        """
        # 添加特殊的token文本
        text = str(self.data[:,0][item])
        lable = self.data[:,1][item]
        labels = torch.LongTensor([lable])

        # 手动构建inputs
        tokens = self.tokenizer.tokenize(text)  # 获取 对句子分词,对没有的词处理成unk
        tokens=self.tokenizer.convert_tokens_to_ids(tokens) # 分词结果查到到词的tokenid
        token_ids=[101]+tokens+[102]  # [cls]+tokens+[sep]
        # 对token_ids补白得到token_embedding(tensor格式)
        token_embedding = seq_padding(token_ids, self.max_len)
        segment_ids=[0] * len(token_ids) #分句id编码
        segment_embedding = seq_padding(segment_ids, self.max_len)  # 只有一个句子就全是0

        ##attentionmask ，注意力编码()
        attention_ids = [1 for _ in range(len(token_ids))]  # 1111
        mask_embedding = seq_padding(attention_ids, self.max_len)

        # 设置为字典格式
        sample = {"text":text
            ,'token_embedding': token_embedding.flatten()
            , 'segment_embedding': segment_embedding.flatten()
            , "mask_embedding": mask_embedding.flatten()
            , "labels": labels.flatten()
                  }
        return sample

def log_creater(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_name='{}.log'.format(time.strftime("%Y-%m-%d-%H-%M"))
    final_log_file=os.path.join(output_path,log_name)
    # create a log
    log=logging.getLogger("train.log")
    log.setLevel(logging.INFO)
    #
    fine=logging.FileHandler(final_log_file)
    fine.setLevel(logging.DEBUG)

    stream=logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    formatter=logging.Formatter("%(asctime)s line:%(lineno)d ==> %(message)s")

    fine.setFormatter(formatter)
    stream.setFormatter(formatter)

    log.addHandler(fine)
    log.addHandler(stream)

    log.info("creating {}".format(final_log_file))

    return log



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
    embed_dict = {}

    with codecs.open(embed_path, 'r', 'utf-8') as infs:
        # 从第二行开始读
        for inf in infs.readlines()[1:]:
            inf = inf.strip()
            inf_list = inf.split(" ")
            token = ''.join(inf_list[0:-embed_dim])
            if token in token_key.keys() and len(inf_list[-embed_dim:]) == embed_dim:
                embed_dict[token] = list(map(float, inf_list[-embed_dim:]))
    print("{}tokens have corresponding embedding vector".format(len(embed_dict)))
    return  embed_dict

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

