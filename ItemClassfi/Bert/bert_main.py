#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/12/30 4:42 下午 
:@File : bert_main
:Version: v.1.0
:Description:
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
from sklearn.model_selection import train_test_split


def get_data(data_path):
    """
    :param data_path: 新闻数据集目录
    :return: 测试集和训练集以及lable
    """
    # 读取数据，划分训练集和验证集

    # df = pd.read_csv(data_path, delimiter = "_!_", names=['labels','text'], header = None, encoding='utf-8')
    df=pd.read_json(data_path)["data"].values.tolist()
    df=pd.DataFrame(df)  # SPBMMC_6,MC
    df = shuffle(df)   #shuffle数据
    #把类别转换为数字，一共15个类别"民生故事","文化","娱乐","体育","财经","房产","汽车","教育","科技","军事","旅游","国际","证券股票","农业","电竞游戏"
    class_le = LabelEncoder()
    df["labels"]= class_le.fit_transform(df["SPBMMC_6"].values) #将label转换为数字
    x=df["MC"].values
    y=df["labels"].to_frame().values  # 加to_frame可以让[1,2,3] 变成[[1],[2],[3]]
    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.8,
                                                                      test_size=0.2)
    return train_data, test_data, train_label, test_label

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
    L = [len(x) for x in X]
    ML = max(L)   #这个值<=max_len
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data,label, batch_size=32, maxlen=50,shuffle=True):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.maxlen=maxlen
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(self.data.shape[0]))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2 = [], []
            i0=0
            for i in idxs:
                d = self.data[i]
                text = d[:self.maxlen]  # 控制最大的句子长度
                x1, x2 = tokenizer.encode(first=text) # x1为（cls,句子，SEP)在词典中的id,x2为seging几段的id:比如句子有两端[0,0,0,1,1]
                X1.append(x1)
                X2.append(x2)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y=self.label[i0:i+1]
                    i0=i+1
                    yield [X1, X2],Y  # 返回一个个batch(array)
                    [X1, X2] = [], []

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
    data_path=os.path.join(user_path,'bert/toutiao_news_dataset.txt')
    config_path = os.path.join(user_path,'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json')
    checkpoint_path = os.path.join(user_path,'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt')
    dict_path = os.path.join(user_path,'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt')

    maxlen = 50  # 设置序列长度为100，要保证序列长度不超过512
    token_dict= get_token_dict(dict_path)
    train_data, test_data, train_label, test_label = get_data(data_path)  #获取训练测试数据以及词字典
    tokenizer = OurTokenizer(token_dict) # 重写tokenizer

    train_D = data_generator(train_data,train_label)
    valid_D = data_generator(test_data,test_label)

    # 加载预训练模型
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型
    print("模型加载完毕")
    # 同意模型训练
    for l in bert_model.layers:
        l.trainable = True
    # 搭建网络
    model=build_bert_model()

    train_D = data_generator(train_data,train_label)
    valid_D = data_generator(test_data,test_label)

    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=5,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D)
    )
