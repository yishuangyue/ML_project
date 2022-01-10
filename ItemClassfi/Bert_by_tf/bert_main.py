#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/12/30 4:42 下午 
:@File : bert_main
:Version: v.1.0
:Description:
{
"attention_probs_dropout_prob": 0.1, #乘法attention时，softmax后dropout概率
"hidden_act": "gelu", #激活函数
"hidden_dropout_prob": 0.1, #隐藏层dropout概率
"hidden_size": 768, #隐藏单元数
"initializer_range": 0.02, #初始化范围
"intermediate_size": 3072, #升维维度
"max_position_embeddings": 512,#一个大于seq_length的参数，用于生成position_embedding "num_attention_heads": 12, #每个隐藏层中的attention head数
"num_hidden_layers": 12, #隐藏层数
"type_vocab_size": 2, #segment_ids类别 [0,1]
"vocab_size": 30522 #词典中词数
}
"""
import codecs
import os
os.environ['TF_KERAS'] = '1'
import pandas as pd
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Dense,Lambda
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# [101, 100, 2094, 3221, 784, 720, 3416, 2094, 4638]
# 镊子是什么样子的
def get_data(data_path):
    """
    :param data_path: 新闻数据集目录
    :return: 测试集和训练集以及lable
    """
    # 读取数据，划分训练集和验证集
    df=pd.read_json(data_path)["data"].values.tolist()
    df=pd.DataFrame(df)  # SPBMMC_6,MC
    df = shuffle(df)   #shuffle数据
    class_le = LabelEncoder()
    df["labels"]= class_le.fit_transform(df["SPBMMC_6"].values) #将label转换为数字
    df=df[["MC","labels","SPBMMC_6"]][:10]
    target_names=list(set(df["SPBMMC_6"].values))
    data=df[["MC","labels"]].values

    # 按照9:1的比例划分训练集和验证集
    train_data = np.array([j for i, j in enumerate(data) if i % 10 != 0])
    valid_data = np.array([j  for i, j in enumerate(data) if i % 10 == 0])


    return train_data, valid_data, target_names

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


def seq_padding(X, padding=0):
    # 用 0 填充序列
    # 让所有输入序列长度一致
    L = [len(x) for x in X]
    ML = max(L)   #这个值<=max_len
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator(DataGenerator):
    """
    数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_len=maxlen)#[1,3,2,5,9,12,243,0,0,0]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# class data_generator:
#     def __init__(self, data, batch_size=32):
#         self.data = data
#         self.batch_size = batch_size
#         self.steps = len(self.data) // self.batch_size
#         if len(self.data) % self.batch_size != 0:
#             self.steps += 1
#     def __len__(self):
#         return self.steps
#     def __iter__(self):
#         while True:
#             idxs = list(range(len(self.data)))
#             np.random.shuffle(idxs)
#             batch_token_ids, batch_segment_ids, batch_labels = [], [], []
#             for i in idxs:
#                 d = self.data[i]
#                 text = d[0][:maxlen]
#                 # x1 是字对应的索引
#                 # x2 是句子对应的索引
#                 token_ids, segment_ids  = tokenizer.encode(first=text)
#                 label = d[1]
#                 batch_token_ids.append(token_ids)
#                 batch_segment_ids.append(segment_ids)
#                 batch_labels.append([label])
#                 if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
#                     batch_token_ids = seq_padding(batch_token_ids)
#                     batch_segment_ids = seq_padding(batch_segment_ids)
#                     batch_labels = seq_padding(batch_labels)
#                     yield [batch_token_ids, batch_segment_ids], batch_labels
#                     [batch_token_ids, batch_segment_ids, batch_labels] = [], [], []



def build_bert_model():
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    p = Dense(11, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(1e-5),
                  metrics=['accuracy'])
    model.summary()
    return model



if __name__=="__main__":
    # 预训练模型
    user_path="/Users/liting/Documents/python/Moudle/ML_project/data/"
    # user_path="/opt/liting/ML_project/data/"
    data_path=os.path.join(user_path,'test.json')
    config_path = os.path.join(user_path,'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json')
    checkpoint_path = os.path.join(user_path,'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt')
    dict_path = os.path.join(user_path,'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt')

    maxlen = 30  # 设置序列长度为100，要保证序列长度不超过512
    batch_size = 128

    token_dict = get_token_dict(dict_path)
    train_data, valid_data,target_names = get_data(data_path)  #获取训练测试数据以及词字典

    tokenizer = OurTokenizer(token_dict) # 重写tokenizer

    # 加载预训练模型
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型
    print("模型加载完毕")
    # 同意模型训练
    for l in bert_model.layers:
        l.trainable = True
    # 搭建网络
    model=build_bert_model()
    train_D = data_generator(train_data,batch_size)
    valid_D = data_generator(valid_data,batch_size)
    print("数据生成完毕：训练集每一个epochs要执行{}步".format(train_D.__len__()))
    #
    model.fit(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=1,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D)
    )
    print("模型训练完毕")

    test_pred = []
    test_true = []
    for x,y in valid_D:
        print(x,y)
        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

    test_true = valid_data[:,1].tolist()
    print(set(test_true))
    print(set(test_pred))

    print(classification_report(test_true, test_pred))
