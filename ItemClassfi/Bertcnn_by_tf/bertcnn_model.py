# !/usr/bin/env python
# -*- coding:utf8 -*-
from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from tensorflow.keras.layers import SeparableConvolution1D,BatchNormalization,MaxPool1D,concatenate,Dropout,Flatten,GlobalMaxPooling1D
from bert4keras.optimizers import Adam
from tensorflow.keras import regularizers


def textcnn(inputs, kernel_initializer):
    # 3,4,5 inputs:shape=[batch_size,maxlen-2(28),768]
    cnn1 = SeparableConvolution1D(256, 3, padding='same', strides=1, activation='relu',
                                    kernel_initializer=kernel_initializer,
                                    kernel_regularizer=regularizers.l1(0.00001))(inputs)  # 卷积层 深度可分的一维卷积  shape=[batch_size,maxlen-2(28),128]
    cnn1 = BatchNormalization()(cnn1)  # BN标准化
    cnn1 = GlobalMaxPooling1D()(cnn1)  # #shape=[batch_size,128]

    cnn2 = SeparableConvolution1D(256, 4, padding='same', strides=1, activation='relu',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=regularizers.l1(0.00001))(inputs)  # shape=[batch_size,maxlen-2(28),128]
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = GlobalMaxPooling1D()(cnn2)  # #shape=[batch_size,128]
    cnn3 = SeparableConvolution1D(256, 5, padding='same', strides=1, activation='relu',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=regularizers.l1(0.00001))(inputs) # shape=[batch_size,maxlen-2(28),128]
    cnn3 = BatchNormalization()(cnn3)
    cnn3 = GlobalMaxPooling1D()(cnn3)  #shape=[batch_size,128]
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)  #shape=[batch_size,256*3]
    dropout = Dropout(0.5)(cnn)  # shape=[batch_size,256*3]

    return dropout


def build_bert_model(config_path, checkpoint_path, class_nums):
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='bert',
        return_keras_model=False)

    # 获取cls特征 字-》bert>768维，从这个输出中抽取cls,在第一个位置，如果不用cnn,后面接一个全连接层就可以分类了
    cls_features = keras.layers.Lambda(
        lambda x: x[:, 0],  #
        name='cls-token'
    )(bert.model.output)  # shape=[batch_size,768]
    # 去除第一个cls和最后一个sep后的embedding，把这个传给textcnn
    all_token_embedding = keras.layers.Lambda(
        lambda x: x[:, 1:-1],  #
        name='all-token'
    )(bert.model.output)  # shape=[batch_size,maxlen-2,768]

    cnn_features = textcnn(
        all_token_embedding, bert.initializer)  # shape=[batch_size,cnn_output_dim]
    # 拼接cls_embedding和cnn特征，输入给全连接层
    concat_features = keras.layers.concatenate(
        [cls_features, cnn_features],
        axis=-1)
    ## 全连接层
    dense = keras.layers.Dense(
        units=512,
        activation='relu',
        kernel_initializer=bert.initializer
    )(concat_features)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    output = keras.layers.Dense(
        units=class_nums,
        activation='softmax',
        kernel_initializer=bert.initializer
    )(dense)

    model = keras.models.Model(bert.model.input, output)

    return model
