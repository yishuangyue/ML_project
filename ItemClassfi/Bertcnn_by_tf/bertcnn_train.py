#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2022/1/4 2:10 下午 
:@File : bertcnn_train
:Version: v.1.0
:Description:
"""
import datetime
import json, os
import pandas as pd
import numpy as np

os.environ['TF_KERAS'] = '1'
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam
import tensorflow as tf
from bertcnn_model import build_bert_model
import data_helper


class data_generator(DataGenerator):
    """
    数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)  # [1,3,2,5,9,12,243,0,0,0]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


if __name__ == '__main__':
    # 定义路径以及参数
    user_path = "/data/"
    # user_path="/opt/liting/ML_project/data/"
    data_path = os.path.join(user_path, 'test.json')
    config_path = os.path.join(user_path, 'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json')
    checkpoint_path = os.path.join(user_path, 'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt')
    dict_path = os.path.join(user_path, 'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt')
    # 定义超参数和配置文件
    class_nums = 11
    maxlen = 15
    batch_size = 32

    token_dict = data_helper.get_token_dict(dict_path)
    tokenizer = Tokenizer(token_dict)
    # 加载数据集
    train_data, test_data, target_names = data_helper.get_data(data_path)

    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    test_generator = data_generator(test_data, batch_size)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = build_bert_model(config_path, checkpoint_path, class_nums)
    print(model.summary())
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(5e-6),
        metrics=['accuracy'],
    )

    earlystop = keras.callbacks.EarlyStopping( # 当被监控的数量停止提高时，就停止训练
        monitor='val_loss',  # 需要监控的数量。
        patience=3,  # 产生被监视对象的epoch的数量
        verbose=2, # 冗长模式。
        mode='min'  # auto, min, max}之一。在“min”模式下,当培训数量停止时，训练数据已停止下降;“max”模式下，训练数据停止时，数量监测已停止增加;“auto”模式下，自动推断出来
    )
    # 保存模型参数
    # bast_model_filepath = os.path.join(user_path, 'checkpoint/model_{epoch:02d}-{val_accuracy:.2f}.h5')
    bast_model_filepath = os.path.join(user_path, 'checkpoint/best_model.weights')
    checkpoint = keras.callbacks.ModelCheckpoint(
        bast_model_filepath,
        monitor='val_loss',  # 需要监视的值，val_accuracy、val_loss或者accuracy
        verbose=1,  # 信息展示模式
        save_best_only=True,  # 当设置为True时，表示当模型这次epoch的训练评判结果（monitor的监测值）比上一次保存训练时的结果有提升时才进行保存
        mode='min',
        # ‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
        save_weights_only=True
        # 若设置为True，占用内存小（只保存模型权重），但下次想调用的话，需要搭建和训练时一样的网络。若设置为False，占用内存大（包括了模型结构和配置信息），下次调用可以直接载入，不需要再次搭建神经网络结构。
    )
    log_dir = os.path.join(user_path, "logs/fit/" )
                           #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 模型可视化
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if os.path.exists(bast_model_filepath):
         model.load_weights(bast_model_filepath)
         # 若成功加载前面保存的参数，输出下列信息
         print("checkpoint_loaded")
    model.fit(
        train_generator.forfit(),  # BERT的输入是token_ids和segment_ids
        steps_per_epoch=len(train_generator),
        epochs=1,
        validation_data=test_generator.forfit(),
        validation_steps=len(test_generator),
        shuffle=True,
        callbacks=[earlystop, checkpoint,tensorboard_callback ]
    )
    # model.load_weights(bast_model_filepath)
    test_pred = []
    test_true = []
    for x, y in test_generator:
        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

    test_true = test_data[:, 1].tolist()
    print(set(test_true))
    print(set(test_pred))
    # target_names = [line.strip() for line in open('../../data/bert/label', 'r', encoding='utf8')]
    # print(classification_report(test_true, test_pred,target_names=target_names))
    print(classification_report(test_true, test_pred))
