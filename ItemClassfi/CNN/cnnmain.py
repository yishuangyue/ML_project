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
import pandas as pd
# 如果多进程分词可以导入

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, multi_gpu_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model,models,callbacks
from tensorflow.keras.layers import Dense, Embedding, Input, BatchNormalization
from tensorflow.keras.layers import  Flatten, Dropout, MaxPool1D, SeparableConvolution1D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import concatenate
from tensorflow import saved_model
# 准确率
from ItemClassfi.JiebaSplit.jiebasplit import jieba_split


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

        # if if_load == 0:
        #     self.vec = TfidfVectorizer(
        #             # analyzer='char'  # 定义特征为词(word)或n-gram字符，如果传递给它
        #             # # 的调用被用于抽取未处理输入源文件的特征序列
        #             # # ,token_pattern=r"(?u)\b\w+\b"  #它的默认值只匹配长度≥2的单词，这里改为大于1的
        #             # , max_df=0.7  # 默认1 ，过滤出现在超过max_df/低于min_df比例的句子中的词语；正整数时,则是超过max_df句句子。
        #     )
        # else:
        #     self.vec = joblib.load(self.vec_path)

    def create_x_y(self, df_data, num_words, max_len):
        """
        :param df_data: 句子，标签，分词结果
        :param num_words: 设置的总词数
        :param max_len: 句子最大词数
        :return: x,y,实际词总数，类别个数
        """
        num_classes = len(set(df_data["SPMC_type"]))
        y = to_categorical(df_data['SPMC_type'], num_classes=num_classes)  # 将标签one_hot处理,返回array([])
        # 对X值的处理
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(df_data["words"])
        words_num = len(tokenizer.word_index)  # 词的大小
        x_words = tokenizer.texts_to_sequences(df_data["words"])
        x = pad_sequences(x_words, maxlen=max_len, padding='post')  # 将词转换为数字位置array([[2,1,3,0,0],[4,1,4,0,0]...],不足用0填充
        return x, y, words_num, num_classes

    # 搭建cnn网络
    def cnn(self, words_num, embedding_dims, max_len, num_class):
        tensor_input = Input(shape=(max_len,), dtype='float64')
        embed = Embedding(words_num + 1, embedding_dims)(tensor_input)
        #
        cnn1 = SeparableConvolution1D(128, 3, padding='same', strides=1, activation='relu',
                                      kernel_regularizer=regularizers.l1(0.00001))(embed)  # 卷积层 深度可分的一维卷积
        cnn1 = BatchNormalization()(cnn1)  # BN标准化
        cnn1 = MaxPool1D(pool_size=4)(cnn1)  # 池化层
        cnn2 = SeparableConvolution1D(128, 4, padding='same', strides=1, activation='relu',
                                      kernel_regularizer=regularizers.l1(0.00001))(embed)
        cnn2 = BatchNormalization()(cnn2)
        cnn2 = MaxPool1D(pool_size=4)(cnn2)  # 池化大小（1*100）
        cnn3 = SeparableConvolution1D(128, 5, padding='same', strides=1, activation='relu',
                                      kernel_regularizer=regularizers.l1(0.00001))(embed)
        cnn3 = BatchNormalization()(cnn3)
        cnn3 = MaxPool1D(pool_size=4)(cnn3)
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
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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
    # 原始数据路径
    input_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/JiebaSplit/test.json"
    # 停用词路径
    chinsesstop_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/JiebaSplit/chinsesstop.txt"
    # 模型保存路径（一个是XGBst模型，一个是TFIDF词进行向量化模型）
    model_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/CNN/clf.json"
    clfmodel_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/CNN/clf.h5"
    output_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/CNN/out.csv"
    # 1、创建NB分类器
    CNNclassifier = CNNclassifier(clf_path=clfmodel_path,model_path=model_path,
                                  output_path=output_path, if_load=0)

    # 2、载入分词好的数据
    jiaba_split = jieba_split(input_path=input_path, chinsesstop_path=chinsesstop_path)
    df_data = jiaba_split.run_split_data()  # df["名称"，'分类'，'分词结果']

    # 3、生成训练集和测试集
    num_words = 12000  # 总词数
    max_len = 8  # ，句子长度，不足取0
    x, y, words_num, num_classes = CNNclassifier.create_x_y(df_data, num_words, max_len)
    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.8,
                                                                      test_size=0.2)
    print("训练集测试集生成好了, max_len句子最大词数={},words_num词总数={}, num_classes类别数量={} ".format(max_len, words_num, num_classes))
    print("train_data.shape:{}".format(train_data.shape))
    print("train_label.shape:{}".format(train_label.shape))
    print(num_classes, words_num, max_len)
    # # 4、训练model
    # 定义模型结构
    model_sepcnn = CNNclassifier.cnn(words_num=words_num, embedding_dims=200, max_len=max_len,
                                     num_class=num_classes)
    # # 修改callback的参数列表,选择需要保存的参数，回调函数
    callbacks = [
        callbacks.ModelCheckpoint(filepath=clfmodel_path,  # 模型的输出目录
                                        save_best_only = True,  # 只保存输出最好结果的模型数据
                                        save_weights_only = False),  # 保存模型与参数
        callbacks.EarlyStopping(patience=5, min_delta=1e-3),  # 当训练效果提示缓慢时提前终止训练
    ]

    model_sepcnn.fit(train_data, train_label, epochs=50, batch_size=512
                     # ,validation_data=(x_valid_scaled, y_valid)
                    ,callbacks =callbacks)
    print('训练完成')
    # 5、模型预估evaluate the model
    scores = model_sepcnn.evaluate(train_data, train_label, verbose=0)
    print("训练集%s: %.2f%%" % (model_sepcnn.metrics_names[1], scores[1]*100))
    scores = model_sepcnn.evaluate(test_data, test_label, verbose=0)
    print("测试集%s: %.2f%%" % (model_sepcnn.metrics_names[1], scores[1]*100))
    model_sepcnn.save(clfmodel_path)
    print("模型保存成功")

   # 6、模型加载
   # 模型的载入(包含模型结构与权重)
    loaded_model = models.load_model(clfmodel_path)
    print("模型加载成功")
    # 使用测试集验证模型
    scores = loaded_model.evaluate(test_data, test_label)
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