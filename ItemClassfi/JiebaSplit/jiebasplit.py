#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2021/11/24 11:13 上午 
:@File : jiebasplit
:Version: v.1.0
:Description:
用于对发票描述信息分词
"""
import re

import jieba
import pandas as pd


def read_data(path):
    """
    :param path: json路径
    :return: 返回dataframe
    """
    ods_data = pd.read_json(path, encoding="utf8")
    ods_data = ods_data[["MC", "SPMC"]]
    return ods_data


def stopwordslist():
    """
    # 创建停用词列表
    :return: 停用词list
    """
    stopwords = [line.strip() for line in open(chinsesstop_path, encoding='UTF-8').readlines()]
    return stopwords


def split_func(x):
    """
    :param x: str
    :return: 分词后，并去掉停用词以及空格，标调符号
    """
    # 去掉非中文|非字母|非下划线
    regexp = re.compile(r'[^\u4e00-\u9fa5|a-zA-Z0-9|_]')
    # 判断正则是不是在这个字符串中：regexp.search(key)
    sentence_depart = jieba.cut(x, cut_all=False)
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词,并且每个词踢掉其他非中文|非字母|非下划线
    for word in sentence_depart:
        if word not in stopwords:
            # 去掉非中文|非字母|非下划线
            # 并且不是全是数字
            word = regexp.sub('', word)
            if word != '' and not word.isdigit():
                outstr += ","
                outstr += word
    return outstr[1:]


def split_name(data):
    print("开始分词")
    MC_data = data["MC"].map(lambda x: split_func(x)).to_frame()
    MC_data.columns = ["words"]
    # 查看分词前后的差别
    end_df = pd.concat([data, MC_data[["words"]]], axis=1, ignore_index=False)
    print(end_df.head(10))
    print("分词完成")
    return MC_data


if __name__ == "__main__":
    path = "/ItemClassfi/JiebaSplit/test.json"
    chinsesstop_path = "/ItemClassfi/JiebaSplit/chinsesstop.txt"
    data_name = read_data(path)
    split_name(data_name)
