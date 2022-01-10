#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
:Copyright: Tech. Co.,Ltd.
:Author: liting
:Date: 2022/1/6 2:32 下午 
:@File : data_helper
:Version: v.1.0
:Description:
对数据做预处理，用于分类所有模型数据输入
"""
import codecs, os, re, jieba
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

class data_deal:
    def __init__(self, dataset):
        self.data_path = os.path.join(dataset, "test.json")
        self.class_path = os.path.join(dataset, "class.txt")
        self.chinsesstop_path = os.path.join(dataset, "chinsesstop.txt")

    def split_func(self, x):
        """
            分词函数
            :param x: str
            :return: 分词后，并去掉停用词以及空格，标调符号
            """
        # 去掉非中文|非字母|非下划线
        regexp = re.compile(r'[^\u4e00-\u9fa5|a-zA-Z0-9|_]')
        # 判断正则是不是在这个字符串中：regexp.search(key)
        sentence_depart = jieba.cut(x, cut_all=True)
        # 创建一个停用词列表
        stopwords = [line.strip() for line in open(self.chinsesstop_path, encoding='UTF-8').readlines()]
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

    def get_data(self, ifsplit_data=0, ifget_classname=0):
        """
        :param ifsplit_data: 是否分词
        :param ifout_classname: 是否输出lebel的名称到txt中
        :return: 测试集和训练集
        """
        # 读取数据，划分训练集和验证集
        df = pd.read_json(self.data_path)
        df = pd.DataFrame(df["data"].values.tolist())  # SPBMMC_6,MC
        # 将分类类别转换成数值
        class_le = LabelEncoder()
        df["labels"] = class_le.fit_transform(df["SPBMMC_6"].values)
        print("原始数据labels描述：{}".format(df["labels"].describe()))
        df = shuffle(df)  # shuffle数据
        if ifsplit_data == 1:
            print("开始分词")
            df["MC"] = df["MC"].map(lambda x: self.split_func(x))
        df["text_len"]=df["MC"].map(len)
        print("原始数据文本长度信息描述：{}".format(df["text_len"].describe()))
        data = df[["MC", "labels"]].values
        # 按照8:1:1的比例划分训练集和验证集
        train_data = np.array([j for i, j in enumerate(data) if i % 10 != 0 and i % 10 != 1])
        test_data = np.array([j for i, j in enumerate(data) if i % 10 == 0])
        dev_data = np.array([j for i, j in enumerate(data) if i % 10 == 1])

        if ifget_classname == 1:
            # 将标签按顺序写入标签txt文件
            df = df.astype({"labels": str})  # 修改lable类型为str
            df["new"] = df["SPBMMC_6"] + "|" + df["labels"]
            lables_name = set(df["new"])
            lables_name = [i.split("|") for i in lables_name]
            lables_name.sort()
            lables_name=[i for i,j in lables_name]
            print("数据处理完成！！！！")
            return train_data, test_data,dev_data,lables_name
            # 将class 数据写入
            # with open(self.class_path, 'w', encoding='utf-8') as f:
            #     for i in lables_name:
            #         f.write(i[1])
            #         f.write('\n')
            # f.close()
        print("数据处理完成！！！！")
        return train_data, test_data,dev_data



    def get_token_dict(self,dict_path):
        """
        # 将词表中的词编号转换为token字典
        :param dict_path: 词路径
        :return:  返回字典{词：token_id}
        """
        token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return token_dict



    def wirte_txt(self):
        """
        将数据输出为其他格式并写入txt
        :return:
        """
        class_path = "/THUCNews/datanew/"
        df = pd.read_json(self.input_path)
        df = pd.DataFrame(df["data"].values.tolist())  # SPBMMC_6,MC
        # 将分类类别转换成s数值
        class_le = LabelEncoder()
        df["labels"] = class_le.fit_transform(df["SPBMMC_6"].values)
        df = df.astype({"labels": str})  # 修改lable类型为str
        df["new"] = df["SPBMMC_6"] + "|" + df["labels"]
        lables_name = set(df["new"])
        lables_name = [i.split("|") for i in lables_name]
        lables_name.sort()
        # 将class 数据写入
        with open(os.path.join(class_path, "classnew.txt"), 'w', encoding='utf-8') as f:
            for i in lables_name:
                f.write(i[1])
                f.write('\n')
        f.close()

        df = shuffle(df)
        df["MC"] = df["MC"].map(lambda x: x.replace("\t", "").strip())  # 将mc这种tab，以及前后空格去掉
        df[["MC", "labels"]].iloc[:2000].to_csv(os.path.join(class_path, 'test.txt'), sep='\t', index=False,
                                                header=False)
        df[["MC", "labels"]].iloc[2000:4000].to_csv(os.path.join(class_path, 'dev.txt'), sep='\t', index=False,
                                                    header=False)
        df[["MC", "labels"]].iloc[4000:11000].to_csv(os.path.join(class_path, 'train.txt'), sep='\t', index=False,
                                                     header=False)



if __name__=="__main__":
    a=data_deal("/Users/liting/Documents/python/Moudle/ML_project/data/")
    train_data, valid_data= a.get_data(0,0)




