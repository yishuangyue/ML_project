#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/25 3:40 下午 
:@File : BayesClassfi
:Version: v.1.0
:Description:
使用贝叶斯分类思想对分词后的数据进行分类
"""
import os.path

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from ItemClassfi.data_helper import data_deal


class NBclassifier():
    def __init__(self, clf_path, vec_path, if_load):
        '''
        创建对象时完成的初始化工作，判断分类器与vector路径是否为空，
        若为空则创建新的分类器与vector，否则直接加载已经持久化的分类器与vector。
        '''
        self.clf_path = clf_path
        self.vec_path = vec_path
        self.if_load = if_load

        if if_load == 0:
            self.clf = MultinomialNB()
            self.vec = TfidfVectorizer(
                analyzer='char'  # 定义特征为词(word)或n-gram字符，如果传递给它
                # 的调用被用于抽取未处理输入源文件的特征序列
                # ,token_pattern=r"(?u)\b\w+\b"  #它的默认值只匹配长度≥2的单词，这里改为大于1的
                , max_df=0.7  # 默认1 ，过滤出现在超过max_df/低于min_df比例的句子中的词语；正整数时,则是超过max_df句句子。
            )
        else:
            self.clf = joblib.load(self.clf_path)
            self.vec = joblib.load(self.vec_path)

    # 保存模型
    def saveModel(self):
        joblib.dump(self.clf, self.clf_path)
        joblib.dump(self.vec, self.vec_path)

    # 训练数据
    def trainNB(self, dataList, labelList):
        # 训练模型首先需要将分好词的文本进行向量化，这里使用的TFIDF得到词频权重矩阵
        train_features = self.vec.fit_transform(dataList)
        self.clf.fit(train_features, labelList)
        self.saveModel()
        print("模型保存好了！！！")

    # 预测数据
    def predictNB(self, dataList, labelList):
        data = self.vec.transform(dataList)
        predictList = self.clf.predict(data)
        print("准确率：%s" % metrics.accuracy_score(labelList, predictList))
        end_df = pd.concat([pd.DataFrame(dataList), pd.DataFrame(labelList)
                               , pd.DataFrame(predictList.tolist())], axis=1, ignore_index=False)
        # print(end_df.head(20))
        return predictList


if __name__ == '__main__':
    # 原始数据路径
    user_path= "/Users/liting/Documents/python/Moudle/ML_project/data/"
    # 模型保存路径（一个是贝叶斯模型，一个是TFIDF词进行向量化模型）
    clfmodel_path = os.path.join(user_path,"model/Bayes/clf.m")
    vecmodel_path = os.path.join(user_path,"model/Bayes/vec.m")

    # 1、创建NB分类器
    nbclassifier = NBclassifier(vec_path=vecmodel_path, clf_path=clfmodel_path, if_load=0)

    # 2、载入训练数据与预测数据
    jiaba_split = data_deal(dataset=user_path)
    train_data, test_data,dev_data = jiaba_split.get_data(ifsplit_data=1)  # df["text分词结果"，'lable']
    if nbclassifier.if_load == 0: #是否需要训练数据或者直接加载模型
        print("开始训练")
        print(train_data.shape)
        nbclassifier.trainNB(train_data[:,0].astype("str"), train_data[:,1].astype("int"))  # 一定注意输入数据的类型
        print("模型训练并保存好了")
    print("训练集准确率：")
    nbclassifier.predictNB(train_data[:,0].astype("str"), train_data[:,1].astype("int"))
    print("测试集准确率：")
    predictList = nbclassifier.predictNB(test_data[:,0].astype("str"), test_data[:,1].astype("int"))









