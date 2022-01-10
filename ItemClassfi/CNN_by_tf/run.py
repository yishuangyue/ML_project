#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/30 11:36 上午
:@File : cnnmain
:Version: v.1.0
:Description:
"""
import codecs
import os,sys
import numpy as np
import logging 
import pandas as pd

from CNN_by_tf.cnn import cnn

sys.path.append(r"/Users/liting/Documents/python/Moudle/ML_project/")
# 如果多进程分词可以导入
from gensim.models.keyedvectors import load_word2vec_format
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models,callbacks
from ItemClassfi.data_helper import data_deal


from data.jiebasplit import jieba_split
import datetime

class CNNclassifier():
    def __init__(self, clf_path, model_path,output_path, if_load):
        '''
        创建对象时完成的初始化工作，判断分类器与vector路径是否为空，
        若为空则创建新的分类器与vector，否则直接加载已经持久化的分类器与vector。
        '''
        self.clf_path = clf_path
        self.model_path=model_path
        self.output_path = output_path
        self.if_load = if_load

    def get_vocab_token(self, dataset, tokenizer,max_len=10):
        """
        :param dataset: [分词结果，lable]
        :param tokenizer: tokenizer对象
        :return: x,y,实际词总数，类别个数
        """
        # y = to_categorical(df_data['SPMC_type'], num_classes=num_classes)  # 将标签one_hot处理,返回array([])
        # 获取词的token,对token补白
        token_ids = tokenizer.texts_to_sequences(dataset[:,0])
        token_ids = pad_sequences(token_ids, maxlen=max_len, padding='post').astype("float64")  # 将词转换为数字位置array([[2,1,3,0,0],[4,1,4,0,0]...],不足用0填充
        labels=dataset[:,1].astype("int")
        return token_ids,labels

    def get_embedding_matrix(self,embed_path,vocab_size,token_key,embedding_dims):
        wv_from_text = load_word2vec_format(embed_path, binary=False) # 直接加载预训练文件
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size+1, embedding_dims)).astype("float32")
        print("预训练 embedding shape000000:{}".format(embedding_matrix.shape))

        cnt=0
        for word, i in token_key.items():
            # embedding_vector = embed_dict.get(word)
            try:
                embedding_vector= wv_from_text.get_vector(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                    cnt+=1
            except:
                pass
            continue
        print("预训练 embedding shape:{}".format(embedding_matrix.shape))
        print("有{} 词有须训练结果".format(cnt))
        return embedding_matrix

    # #加载模型并创建模型 预测数据
    def predictCNN(self, dataList, labelList):
        json_file = open(self.model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.clf_path)
        print("Loaded model from disk")
        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.evaluate(dataList, labelList, verbose=0)
        print("测试集精确度%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        # preds = model.predict(dataList)
        # print("准确率：%s" % accuracy_score(labelList, preds))
        # end_df = pd.concat([pd.DataFrame(dataList), pd.DataFrame(labelList)
        #                        , pd.DataFrame(preds.tolist())], axis=1, ignore_index=False)
        # print(end_df.head(20))
        # return preds

    def write_data(self, preds):
        with open(self.output_path, 'w') as f:
            for i, pre in enumerate(preds):
                f.write(str(i + 1))
                f.write(',')
                f.write(str(int(pre) + 1))
                f.write('\n')


if __name__ == '__main__':
    # 0、定义初始参数
    # 原始数据路径
    user_path= "/Users/liting/Documents/python/Moudle/ML_project/data/"
    # user_path="/opt/liting/ML_project"
    # 模型保存路径
    model_path = os.path.join(user_path,"model/cnn/cnn_clf.json")
    clfmodel_path = os.path.join(user_path,"model/cnn/cnn_clf.ckpt")
    output_path = os.path.join(user_path,"model/cnn/cnn_out.csv")
    # 预训练的ebedding词
    embed_path=os.path.join(user_path,"sgns.wiki.char")
    # 模型在线logging目录
    log_dir = os.path.join(user_path,"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    num_words = 12000  # 总词数
    max_len = 13  # ，句子长度，不足取0
    embedding_dims=300  # embedding维度大小
    epochs=1     # 训练数据全部训练轮数
    batch_size=128   # 经过多少个样本更新参数


    # 1、创建NB分类器
    CNNclassifier = CNNclassifier(clf_path=clfmodel_path,model_path=model_path,
                                  output_path=output_path, if_load=0)

    # 2、载入分词好的数据
    jiaba_split = data_deal(dataset=user_path)
    train_data, test_data,dev_data,labels_name = jiaba_split.get_data(ifsplit_data=1,ifget_classname=1)  # df["text分词结果"，'lable']
    num_classes=len(labels_name)
    data_all=np.concatenate([train_data[:,0],dev_data[:,0],test_data[:,0]])
    # 3、获得词的token
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(data_all)
    vocab_size=len(tokenizer.word_index) # 词总数
    token_key =tokenizer.word_index #词id嵌入字典 eg:{'丙烯画': 1, '颜料': 2, '300ml': 3}

    # 4、获取训练测试集的token
    train_token_ids,train_labels = CNNclassifier.get_vocab_token(train_data, tokenizer,max_len=max_len)
    dev_token_ids,dev_labels = CNNclassifier.get_vocab_token(dev_data, tokenizer,max_len=max_len)
    test_token_ids,test_labels = CNNclassifier.get_vocab_token(test_data, tokenizer,max_len=max_len)
    print("训练集生成好了, max_len句子最大词数={} ,vocab_size 训练集词总数={}, num_classes类别数量={} ".format(max_len, vocab_size, num_classes))
    print("train_data.shape:{},{}".format(train_token_ids.shape , train_labels))

    # 获取词嵌入矩阵预训练结果
    embedding_matrix=CNNclassifier.get_embedding_matrix(embed_path,vocab_size,token_key,embedding_dims)
    # # 4、训练model
    # 4.1 定义模型结构
    model_sepcnn = cnn(vocab_size=vocab_size, embedding_dims=embedding_dims, max_len=max_len,
                                     num_class=num_classes
                       ,embed_matr=embedding_matrix
                       )
    # 4.2、定义模型参数
    # model_sepcnn = multi_gpu_model(model_sepcnn, gpus=i) 如果有gpu,i为gpu数目
    model_sepcnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # # 修改callback的参数列表,选择需要保存的参数，回调函数
    # 下面是通过tensorboard --logdir=callbacks生成的本地网页数据:http://localhost:6006/(通过端口6006在网页中打开)
    # python3 -m /Users/liting/Library/Python/3.6/lib/python/site-packages/tensorboard/tensorboard.main --logdir=/Users/liting/Documents/python/Moudle/ML_project/logs/fit/
    # python -m /root/anaconda3/lib/python3.6/site-packages/tensorboard/main.py  --logdir=/opt/liting/ML_project/logs/fit/
    # callbacks = [
    #     callbacks.ModelCheckpoint(filepath=clfmodel_path,  # 模型的输出目录
    #                                     save_best_only = True,  # 只保存输出最好结果的模型数据
    #                                     save_weights_only = False),  # 保存模型与参数
    #     callbacks.EarlyStopping(patience=5, min_delta=1e-3),  # 当训练效果提示缓慢时提前终止训练
    #     callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # ]

    # 4.3训练模型
    model_sepcnn.fit(train_token_ids
                     , train_labels
                     , epochs=epochs
                     , batch_size=batch_size
                     ,validation_data=(dev_token_ids, dev_labels)
                    # ,callbacks =callbacks
                     )

    print('训练完成')
    # 5、模型预估evaluate the model
    scores = model_sepcnn.evaluate(train_token_ids, train_labels, verbose=0)
    print("训练集%s: %.2f%%" % (model_sepcnn.metrics_names[1], scores[1]*100))
    scores = model_sepcnn.evaluate(test_token_ids, dev_labels, verbose=0)
    print("测试集%s: %.2f%%" % (model_sepcnn.metrics_names[1], scores[1]*100))
    model_sepcnn.save(clfmodel_path)
    print("模型保存成功")

   # 6、模型加载
   # 模型的载入(包含模型结构与权重)
    loaded_model = models.load_model(clfmodel_path)
    print("模型加载成功")
    # 使用测试集验证模型
    scores = loaded_model.evaluate(test_token_ids, dev_labels)
    print(scores)









    # # serialize model to JSON保存模型结构，加载方法二
    # model_json = model_sepcnn.to_json()
    # with open(model_path, "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5保存模型参数
    # model_sepcnn.save_weights("model.h5")
    # print("Saved model to disk")
    #
    # # later...
    # print("模型保存成功")
    # # 5、预测数据并输出结果# load json and create model
    # preds = CNNclassifier.predictCNN(test_data, test_label)
    # pred_ = [model_sepcnn.predict(vec.reshape(1, max_len)).argmax() for vec in test_data]
    # df_test['分类结果_预测'] = [dig_lables[dig] for dig in pred_]
    # metrics.accuracy_score(df_test['标签'], df_test['分类结果_预测'])
    # 6、预测数据保存csv文件
    # XGBclassifier.write_data(preds)

#
# position_encoding = np.array(
#     [[pos / np.power(10000, 2.0 * (j // 2) / 4) for j in range(4)] for pos in range(3)])
#
# position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
# position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

