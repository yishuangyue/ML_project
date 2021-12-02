#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/30 11:36 上午 
:@File : cnntrain
:Version: v.1.0
:Description:
"""
import pandas as pd
import numpy as np
import jieba
import re
# 如果多进程分词可以导入
import multiprocessing
from multiprocessing import Pool

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, multi_gpu_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, \
    Activation, Input, Lambda, Reshape, BatchNormalization
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, SeparableConvolution1D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import concatenate
# 准确率
from sklearn import metrics

from ItemClassfi.JiebaSplit.jiebasplit import jieba_split


class CNNclassifier():
    def __init__(self, clf_path, vec_path, output_path, if_load):
        '''
        创建对象时完成的初始化工作，判断分类器与vector路径是否为空，
        若为空则创建新的分类器与vector，否则直接加载已经持久化的分类器与vector。
        '''
        self.clf_path = clf_path
        self.vec_path = vec_path
        self.output_path = output_path
        self.if_load = if_load

        # if if_load == 0:
        #     self.vec = TfidfVectorizer(
        #             # analyzer='char'  # 定义特征为词(word)或n-gram字符，如果传递给它
        #             # # 的调用被用于抽取未处理输入源文件的特征序列
        #             # # ,token_pattern=r"(?u)\b\w+\b"  #它的默认值只匹配长度≥2的单词，这里改为大于1的
        #             # , max_df=0.7  # 默认1 ，过滤出现在超过max_df/低于min_df比例的句子中的词语；正整数时,则是超过max_df句句子。
        #     )
        # else:
        #     self.vec = joblib.load(self.vec_path)
    # 搭建cnn网络
    def cnn(words_num, embedding_dims, max_len, num_class):
        tensor_input = Input(shape=(max_len,), dtype='float64')
        embed = Embedding(words_num + 1, embedding_dims)(tensor_input)
        cnn1 = SeparableConvolution1D(200, 3, padding='same', strides=1, activation='relu',
                                      kernel_regularizer=regularizers.l1(0.00001))(embed)
        cnn1 = BatchNormalization()(cnn1)
        cnn1 = MaxPool1D(pool_size=100)(cnn1)
        cnn2 = SeparableConvolution1D(200, 4, padding='same', strides=1, activation='relu',
                                      kernel_regularizer=regularizers.l1(0.00001))(embed)
        cnn2 = BatchNormalization()(cnn2)
        cnn2 = MaxPool1D(pool_size=100)(cnn2)
        cnn3 = SeparableConvolution1D(200, 5, padding='same', strides=1, activation='relu',
                                      kernel_regularizer=regularizers.l1(0.00001))(embed)
        cnn3 = BatchNormalization()(cnn3)
        cnn3 = MaxPool1D(pool_size=100)(cnn3)
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        dropout = Dropout(0.5)(cnn)
        flatten = Flatten()(dropout)
        dense = Dense(128, activation='relu')(flatten)
        dense = BatchNormalization()(dense)
        dropout = Dropout(0.5)(dense)
        tensor_output = Dense(num_class, activation='softmax')(dropout)
        model = Model(inputs=tensor_input, outputs=tensor_output)
        print(model.summary())
        # model = multi_gpu_model(model, gpus=i) 如果有gpu,i为gpu数目
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # 训练数据
    def trainXGB(self, dataList, labelList):
        # 训练模型首先需要将分好词的文本进行向量化，这里使用的TFIDF得到词频权重矩阵
        train_features = self.vec.fit_transform(dataList)
        weight = train_features.toarray()
        print(" .shape: {}".format(train_features.shape))
        dtrain = xgb.DMatrix(weight, label=labelList)
        # dtest = xgb.DMatrix(test_weight)  # label可以不要，此处需要是为了测试效果
        param = {'max_depth': 6, 'eta': 0.3, 'eval_metric': 'merror', 'silent': 0, 'objective': 'multi:softmax',
                 'num_class': 10}  # 参数
        # param = {'eta': 0.3, 'max_depth': 6, 'objective': 'multi:softmax'
        #         # ,'silent': 0
        #     , 'num_class': 87, 'eval_metric': 'merror'}  # 参数

        evallist = [(dtrain, 'train')]  # 这步可以不要，用于测试效果
        num_round = 50  # 循环次数
        xgb.XGBClassifier
        model = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round, evals=evallist)
        # 保存模型(两个)
        print()
        model.save_model(self.clf_path)
        joblib.dump(self.vec, self.vec_path)
        return model

    # 预测数据
    def predictXGB(self, dataList, labelList):
        data = self.vec.transform(dataList)
        test_weight = data.toarray()
        dtest = xgb.DMatrix(test_weight)
        bst = xgb.Booster(model_file=self.clf_path)
        preds = bst.predict(dtest)
        print("测试集准确率：%s" % accuracy_score(labelList, preds))
        end_df = pd.concat([pd.DataFrame(dataList), pd.DataFrame(labelList)
                               , pd.DataFrame(preds.tolist())], axis=1, ignore_index=False)
        print(end_df.head(20))
        return preds

    def write_data(self, preds):
        with open(self.output_path, 'w') as f:
            for i, pre in enumerate(preds):
                f.write(str(i + 1))
                f.write(',')
                f.write(str(int(pre) + 1))
                f.write('\n')


if __name__ == '__main__':
    # 原始数据路径
    input_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/JiebaSplit/ods_data5.json"
    # 停用词路径
    chinsesstop_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/JiebaSplit/chinsesstop.txt"
    # 模型保存路径（一个是XGBst模型，一个是TFIDF词进行向量化模型）
    clfmodel_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/XGBOOST/clf.m"
    vecmodel_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/XGBOOST/vec.m"
    output_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/XGBOOST/out.csv"
    # 1、创建NB分类器
    CNNclassifier = CNNclassifier(vec_path=vecmodel_path, clf_path=clfmodel_path,
                                  output_path=output_path, if_load=0)

    # 2、载入训练数据与预测数据
    jiaba_split = jieba_split(input_path=input_path, chinsesstop_path=chinsesstop_path)
    df_data = jiaba_split.run_split_data()  # df["名称"，'分类'，'分词结果']
    # 3、生成训练集和测试集
    x = df_data["words"].to_list()
    y = df_data["SPMC_type"].to_list()
    print(set(y))
    print(set(df_data["SPMC"].to_list()))

    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1
                                                                      , train_size=0.8, test_size=0.2)
    print("训练集测试集生成好了")
    print(train_data[:10])
    print(train_label[:10])

    # 4、训练并预测分类正确性
    model_sepcnn = CNNclassifier.cnn(words_num=len(tokenizer.word_index), embedding_dims=300, max_len=max_len, num_class=num_classes)
    model_sepcnn.fit(train_data, train_lables, epochs=8, batch_size=512)
    print('训练完成')

    pred_ = [model_sepcnn.predict(vec.reshape(1, max_len)).argmax() for vec in test_data]
    df_test['分类结果_预测'] = [dig_lables[dig] for dig in pred_]
    metrics.accuracy_score(df_test['标签'], df_test['分类结果_预测'])


    trainXGB = XGBclassifier.trainXGB(train_data, train_label)
    print("模型训练并保存好了")
    # 5、预测数据并输出结果
    preds = XGBclassifier.predictXGB(test_data, test_label)
    # 6、预测数据保存csv文件
    # XGBclassifier.write_data(preds)

###############################################################
# 读取数据
df_train = pd.read_excel(r'C:\Users\admin\Desktop\text_cf\zzyw.xlsx', sheetname='训练集')
df_test = pd.read_excel(r'C:\Users\admin\Desktop\text_cf\zzyw.xlsx', sheetname='测试集')


# 分词的代码
def seg_sentences(sentence):
    # 去掉特殊字符
    sentence = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", sentence)
    sentence_seged = list(jieba.cut(sentence.strip()))
    return sentence_seged


df_train['文本分词'] = df_train['文本'].apply(seg_sentences)
df_test['文本分词'] = df_test['文本'].apply(seg_sentences)

# Y值的处理
lables_list = df_train['标签'].unique().tolist()  # list[去重y]
dig_lables = dict(enumerate(lables_list))  # 建立 y值索引map {"0":"a","1":"b"}
# Y值的大小即分为多少类
num_classes = len(dig_lables)
lable_dig = dict((lable, dig) for dig, lable in dig_lables.items())  # 准换y值索引map {"a":0,"b":1}
df_train['标签_数字'] = df_train['标签'].apply(lambda lable: lable_dig[lable])
train_lables = to_categorical(df_train['标签_数字'], num_classes=num_classes)  # 将标签名字转换为数字

# 对X值的处理
num_words = 10000  # 总词数
max_len = 200  # ，不足补0，多余截取掉。
tokenizer = Tokenizer(num_words=num_words)
# df_all=pd.concat([df_train['文本分词'],df_test['文本分词']])
tokenizer.fit_on_texts(df_train['文本分词'])

train_sequences = tokenizer.texts_to_sequences(df_train['文本分词'])
train_data = pad_sequences(train_sequences, maxlen=max_len, padding='post')

# 测试集处理
test_sequences = tokenizer.texts_to_sequences(df_test['投诉分词'])
test_data = pad_sequences(test_sequences, maxlen=max_len, padding='post')


