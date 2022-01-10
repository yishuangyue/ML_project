#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2022/1/7 9:54 上午 
:@File : bert
:Version: v.1.0
:Description:
"""
import os,sys

sys.path.append("/opt/liting/ML_project/ItemClassfi/")
# sys.path.append("/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/")
from data_utils import InputDataset, log_creater
from data_helper import data_deal
from pytorch_pretrained_bert import BertModel, BertTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np
from train_eval import train
# from transformers import get_linear_schedule_with_warmup

class Model(torch.nn.Module):
    def __init__(self, bert_path, hidden_size, num_classes):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # softmax的输入 在解码中使用softmax
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # loss ,预测值, =None, attention_mask=None
        _, pooled = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


if __name__ == '__main__':
    ## ____参数配置
    user_path="/opt/liting/"
    # user_path = "/Users/liting/Documents/python/Moudle/"
    data_path = os.path.join(user_path, "ML_project/data")
    # bert_path = os.path.join(user_path, "study/Bert_by_tf-Chinese-Text-Classification-Pytorch/bert_pretrain")
    bert_path = os.path.join(user_path, "student/Bert_by_tf-Chinese-Text-Classification-Pytorch/bert_pretrain")
    log_path = os.path.join(user_path, "ML_project/data/logs/")
    save_path = os.path.join(user_path, "ML_project/data/model/bert/bert_bytorch.ckpt")
    batch_size = 32
    epochs = 4
    max_len = 32
    hidden_size = 768
    num_classes = 25
    learning_rate = 2e-5
    require_improvement = 1000
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)  # 在所有gpu上的随机种子
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  ## #指定模型设备 进阶-》分布式

    # _______加载json数据预处理并划分训练、测试、验证集
    df_data = data_deal(data_path)
    train_data, test_data, dev_data, class_list = df_data.get_data(ifsplit_data=0, ifget_classname=1)
    print(dev_data,class_list)
    # _______数据进行处理成输入tensor,返回字典
    tokenizer = BertTokenizer.from_pretrained(bert_path)  # 词处理
    train_dataset = InputDataset(train_data, tokenizer, max_len=max_len)
    test_dataset = InputDataset(test_data, tokenizer, max_len=max_len)
    dev_dataset = InputDataset(dev_data, tokenizer, max_len=max_len)

    # ________调用dataloader，生成iterable对象
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    print(next(iter(train_dataloader))) # 打印看看，后面调数据格式

    # _________建立模型结构，加载预处理模型
    model = Model(bert_path, hidden_size, num_classes)

    # # 优化器，这部分写入了train.py
    # optimizer = optimization.BertAdam(model.named_parameters(),
    #                                   lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=0,  # 刚开始训练不会快，后面到达一个峰值
    #                                             num_training_steps=total_steps)  # 调度器，控制训练步数

    # __创建日志
    log = log_creater(log_path)
    log.info(" train batch_size = {}".format(batch_size))
    log.info("total steps = {}".format(total_steps))
    log.info("***** 开始训练 ****")

    # ____模型训练,并输出损失与准确度
    config = {"learning_rate": learning_rate
        , "epochs": epochs
        , "total_steps": total_steps
        , "device": device
        , "save_path": save_path
        , "class_list": class_list
        , "require_improvement": require_improvement}
    train(config, model, train_dataloader, test_dataloader, dev_dataloader, log)


    # 训练部分
    # for epoch in range(epochs):
    #     total_train_loss=0
    #     t0 =time.time()
    #     model.to(device)  # 将模型传到到机器上
    #     model.train()   #训练
    #     for step,batch in enumerate(train_dataloader):
    #         token_embedding=batch["token_embedding"].to(device) # 将数据发送到机器上
    #         segment_embedding=batch["segment_embedding"].to(device)
    #         mask_embedding=batch["mask_embedding"].to(device)
    #         labels=batch["labels"].to(device)
    #         model.zero_grad()  # 每一个batch开始都会梯度清零
    #         outputs =model(input_ids=token_embedding,token_type_ids=segment_embedding
    #                        ,attention_mask=mask_embedding)
    #         loss = F.cross_entropy(outputs, labels)
    #         total_train_loss +=loss
    #         loss.backward()
    #         # torch.nn.utils.clip_grad_norm_(model.parameters()) # 梯度剪裁，避免梯度爆炸(大于1的)
    #         optimizer.step() # 优化器调度器更新
    #         scheduler.step()
    #         #现在得到所有batch总的损失
    #     # 求平均损失
    #     avg_train_loss=total_train_loss / len(train_dataloader)
    #     train_time=format_time(time.time()-t0)
    #
    #     log.info("==Epoch:[{}/{}] avg_train_loss={:.5f} ===".format(epoch+1,epochs,avg_train_loss))
    #     log.info("===Training epoc tookL {:}====".format(train_time))
    #     log.info("开始跑验证集")
    #
    #     model.eval()
    #     avg_val_loss,avg_val_acc=evaluate(model,dev_dataloader)
    #
    # if epoch = epoch-1:
    #     torch.save(model,model_path)
