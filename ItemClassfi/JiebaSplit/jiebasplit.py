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
import json
import re
import jieba
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class jieba_split:
    def __init__(self, input_path=None, chinsesstop_path=None):
        if input_path is None and chinsesstop_path is None:
            self.input_path = "/Users/liting/Documents/python/Moudle/ML_project/" \
                              "ItemClassfi/JiebaSplit/test.json"
            self.chinsesstop_path = "/Users/liting/Documents/python/Moudle/ML_project/ItemClassfi/JiebaSplit/chinsesstop.txt"
        else:
            self.input_path = input_path
            self.chinsesstop_path = chinsesstop_path

    def run_split_data(self):
        """
        :param data: 开始分词
        :return:
        """
        print("开始分词")
        data = self.read_data()
        # print(data)
        MC_data = data["MC"].map(lambda x: self.split_func(x)).to_frame()
        MC_data.columns = ["words"]
        # 查看分词前后的差别
        end_df = pd.concat([data, MC_data[["words"]]], axis=1, ignore_index=False)
        print(end_df.head(10))
        print("分词完成")
        return end_df

    def read_data(self):
        """
        :param path: json路径
        :return: 返回dataframe
        """
        # data=[]
        # with open(self.input_path,'r',encoding='utf-8') as f:
        #     for line in f:
        #         data.append((json.loads(line)))
        # ods_data=pd.DataFrame(data)
        ods_data = pd.read_json(self.input_path)
        ods_data = ods_data["data"].values.tolist()
        ods_data=pd.DataFrame(ods_data)
        ods_data.columns = [ "SPMC","MC"]
        # 将分类类别转换成s数值
        class_le = LabelEncoder()
        ods_data["SPMC_type"] = class_le.fit_transform(ods_data["SPMC"].values)

        return ods_data

    def stopwords_list(self):
        """
        # 创建停用词列表
        :return: 停用词list
        """
        stopwords = [line.strip() for line in open(self.chinsesstop_path, encoding='UTF-8').readlines()]
        return stopwords

    def split_func(self, x):
        """
        分词函数
        :param x: str
        :return: 分词后，并去掉停用词以及空格，标调符号
        """
        # 去掉非中文|非字母|非下划线
        regexp = re.compile(r'[^\u4e00-\u9fa5|a-zA-Z0-9|_]')
        # 判断正则是不是在这个字符串中：regexp.search(key)
        sentence_depart = jieba.cut(x, cut_all=False)
        # 创建一个停用词列表
        stopwords = self.stopwords_list()
        # 输出结果为outstr
        outstr = ''
        # 去停用词,并且每个词踢掉其他非中文|非字母|非下划线
        for word in sentence_depart:
            if word not in stopwords:
                # 去掉非中文|非字母|非下划线
                # 并且不是全是数字
                word = regexp.sub('', word)
                if word != '' and not word.isdigit():
                    outstr += " "
                    outstr += word
        return outstr[1:]

if __name__ == "__main__":
    jieba_split = jieba_split()
    aa=jieba_split.run_split_data()
    print(len(set(aa["SPMC_type"].values)))
