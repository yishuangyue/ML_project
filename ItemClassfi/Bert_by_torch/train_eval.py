#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2022/1/7 1:57 下午 
:@File : train
:Version: v.1.0
:Description:
"""

import time,sys
sys.path.append("/opt/liting/ML_project/ItemClassfi/")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from data_utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained.optimization import BertAdam


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_dataloader, test_dataloader, dev_dataloader,log):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config["learning_rate"],
                         warmup=0.05,
                         t_total=config["total_steps"])

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')  # 这个初始值很大
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config["epochs"]):
        print('Epoch [{}/{}]'.format(epoch + 1, config["epochs"]))
        for step, batch in enumerate(train_dataloader):
            token_embedding = batch["token_embedding"].to(config["device"])  # 将数据发送到机器上
            segment_embedding = batch["segment_embedding"].to(config["device"])
            mask_embedding = batch["mask_embedding"].to(config["device"])
            labels = batch["labels"].flatten().to(config["device"]) # 前面是tensor[[1],[2],[3]]->tensor[1,2,3]
            outputs = model(input_ids=token_embedding, token_type_ids=segment_embedding
                            , attention_mask=mask_embedding)
            model.zero_grad()  # 梯度清零
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step() # 优化器调度器更新
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_dataloader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config["save_path"])
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2}, ' \
                      ' Val Acc: {4:>6.2%},  Time: {5} {6}'
                log.info(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config["require_improvement"]:
                # 验证集loss超过1000batch没下降，结束训练
                log.info("长时间没有提升, 自动停止，再改改模型吧！！！...")
                flag = True
                break
        if flag:
            break

    test(config, model, test_dataloader,log)
    log.info("**** 终于训练完成了！！！！！")


def test(config, model, test_dataloader,log):
    # test
    model.load_state_dict(torch.load(config["save_path"]))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_dataloader, test=True)
    msg = '测试集 Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    log.info(msg.format(test_loss, test_acc))
    log.info("Precision, Recall and F1-Score...")
    log.info(test_report)
    log.info("Confusion Matrix...")
    log.info(test_confusion)
    time_dif = get_time_dif(start_time)
    log.info("Time usage:", time_dif)


def evaluate(config, model, dev_dataloader, test = False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in dev_dataloader:
            token_embedding = batch["token_embedding"].to(config["device"])  # 将数据发送到机器上
            segment_embedding = batch["segment_embedding"].to(config["device"])
            mask_embedding = batch["mask_embedding"].to(config["device"])
            labels = batch["labels"].flatten().to(config["device"])
            outputs = model(input_ids=token_embedding, token_type_ids=segment_embedding
                            , attention_mask=mask_embedding)
            loss = F.cross_entropy(outputs, labels)
            # 因为调用F.cross_entropy函数时会通过log_softmax和nll_loss来计算损失，也就是说使用F.cross_entropy函数时
            # ，程序会自动先对out进行softmax，再log，
            # 最后再计算nll_loss：NLLLoss的结果就是把输出与Label对应的那个值拿出来，再去掉负号，再求均值。
            # input = [-0.1187, 0.2110, 0.7463]，target = [1]，那么 loss = -0.2110
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            # input是softmax函数输出的一个tensor，转换为0，1,
            # 还有个argmax是只返回位置索引 torch.max(a,dim)[1]=torch.argmax(a,dim)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    avg_loss = loss_total / len(dev_dataloader)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config["class_list"], digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, avg_loss, report, confusion
    return acc, avg_loss
