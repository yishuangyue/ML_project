#! -*- coding: utf-8 -*-
import json,os
import pandas as pd 
import numpy as np 
os.environ['TF_KERAS'] = '1'
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam
import tensorflow as tf
from bert_model import build_bert_model
import data_helper


class data_generator(DataGenerator):
    """
    数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)#[1,3,2,5,9,12,243,0,0,0]
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
    user_path="/Users/liting/Documents/python/Moudle/ML_project/data/"
    # user_path="/opt/liting/ML_project/data/"
    # data_path=os.path.join(user_path,'bert/toutiao_news_dataset.txt')
    data_path=os.path.join(user_path,'test.json')
    config_path = os.path.join(user_path,'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json')
    checkpoint_path = os.path.join(user_path,'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt')
    dict_path = os.path.join(user_path,'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt')
    #定义超参数和配置文件
    class_nums = 11
    maxlen = 15
    batch_size = 32

    token_dict=data_helper.get_token_dict(dict_path)
    tokenizer = Tokenizer(token_dict)
    # 加载数据集
    train_data,test_data ,target_names = data_helper.get_data(data_path)

    # 转换数据集
    train_generator = data_generator(train_data, batch_size)
    test_generator = data_generator(test_data, batch_size)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = build_bert_model(config_path,checkpoint_path,class_nums)
    print(model.summary())
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(5e-6), 
        metrics=['accuracy'],
    )

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        verbose=2, 
        mode='min'
        )
    bast_model_filepath = os.path.join(user_path,'checkpoint/best_model.weights')
    checkpoint = keras.callbacks.ModelCheckpoint(
        bast_model_filepath, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True,
        mode='min'
        )

    model.fit(
        train_generator.forfit(),   # BERT的输入是token_ids和segment_ids
        steps_per_epoch=len(train_generator),
        epochs=1,
        validation_data=test_generator.forfit(), 
        validation_steps=len(test_generator),
        shuffle=True,
        # callbacks=[checkpoint]
        callbacks=[earlystop,checkpoint]
    )

    model.load_weights(bast_model_filepath)
    test_pred = []
    test_true = []
    for x,y in test_generator:
        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

    test_true = test_data[:,1].tolist()
    print(set(test_true))
    print(set(test_pred))
    # target_names = [line.strip() for line in open('../../data/bert/label', 'r', encoding='utf8')]
    print(classification_report(test_true, test_pred,target_names=target_names))
