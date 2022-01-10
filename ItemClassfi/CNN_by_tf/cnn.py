#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2022/1/10 12:46 下午 
:@File : cnn
:Version: v.1.0
:Description:
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Input, BatchNormalization
from tensorflow.keras.layers import  Flatten, Dropout, MaxPool1D, SeparableConvolution1D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Model
import numpy as np

# 搭建cnn网络
def cnn( vocab_size, embedding_dims, max_len, num_class,embed_matr=None):
    """
    :param words_num:  词的个数
    :param embedding_dims: embedding大小
    :param max_len: 最大词个数
    :param num_class:  类别大小
    :param embed_matr:  查找到的ebeeding矩阵
    :return:
    """
    tensor_input = Input(shape=(max_len,), dtype='float64')
    embed = Embedding(input_dim = vocab_size+1   #还有padding占据了0这个索引
                      , output_dim = embedding_dims
                      , embeddings_initializer= tf.keras.initializers.Constant(embed_matr)
                      # , embeddings_initializer= tf.constant_initializer(embed_matr)
                     # , trainable=False
                      )(tensor_input)
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
    return model
