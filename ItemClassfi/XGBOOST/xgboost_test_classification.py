# -*- coding: utf-8 -*-
import os,joblib
import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from ItemClassfi.data_helper import data_deal


class XGBclassifier():
    def __init__(self, clf_path, vec_path, output_path, if_load):
        '''
        创建对象时完成的初始化工作，判断分类器与vector路径是否为空，
        若为空则创建新的分类器与vector，否则直接加载已经持久化的分类器与vector。
        '''
        self.clf_path = clf_path
        self.vec_path = vec_path
        self.output_path = output_path
        self.if_load = if_load

        if if_load == 0:
            self.vec = TfidfVectorizer(
                    # analyzer='char'  # 定义特征为词(word)或n-gram字符，如果传递给它
                    # # 的调用被用于抽取未处理输入源文件的特征序列
                    # # ,token_pattern=r"(?u)\b\w+\b"  #它的默认值只匹配长度≥2的单词，这里改为大于1的
                    # , max_df=0.7  # 默认1 ，过滤出现在超过max_df/低于min_df比例的句子中的词语；正整数时,则是超过max_df句句子。
            )
        else:
            self.vec = joblib.load(self.vec_path)

    # 训练数据
    def trainXGB(self, dataList, labelList):
        # 训练模型首先需要将分好词的文本进行向量化，这里使用的TFIDF得到词频权重矩阵
        train_features = self.vec.fit_transform(dataList)
        weight = train_features.toarray()
        print("train_features.shape: {}".format(train_features.shape))
        dtrain = xgb.DMatrix(weight, label=labelList)
        # dtest = xgb.DMatrix(test_weight)  # label可以不要，此处需要是为了测试效果
        param = {'max_depth': 5, 'eta': 0.3, 'eval_metric': 'merror', 'silent': 0, 'objective': 'multi:softmax',
                 'num_class': 11}  # 参数

        evallist = [(dtrain, 'train')]  # 这步可以不要，用于测试效果
        num_round = 60  # 循环次数
        model = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round, evals=evallist)
        # 保存模型(两个)
        print()
        model.save_model(self.clf_path)
        joblib.dump(self.vec, self.vec_path)
        return model

    # 预测数据
    def predictXGB(self, dataList, labelList):
        data = self.vec.transform(dataList)
        test_weight = data.toarray()
        dtest = xgb.DMatrix(test_weight)
        bst = xgb.Booster(model_file=self.clf_path)
        preds = bst.predict(dtest)
        print("准确率：%s" % accuracy_score(labelList, preds))
        end_df = pd.concat([pd.DataFrame(dataList), pd.DataFrame(labelList)
                               , pd.DataFrame(preds.tolist())], axis=1, ignore_index=False)
        # print(end_df.head(20))
        return preds

    def write_data(self, preds):
        with open(self.output_path, 'w') as f:
            for i, pre in enumerate(preds):
                f.write(str(i + 1))
                f.write(',')
                f.write(str(int(pre) + 1))
                f.write('\n')


if __name__ == '__main__':
    # 原始数据路径
    user_path= "/Users/liting/Documents/python/Moudle/ML_project/data/"
    # 模型保存路径（一个是XGBst模型，一个是TFIDF词进行向量化模型）
    clfmodel_path = os.path.join(user_path,"model/xgboost/clf.m")
    vecmodel_path = os.path.join(user_path,"model/xgboost/vec.m")
    output_path = os.path.join(user_path,"model/xgboost/out.csv")
    # 1、创建NB分类器
    XGBclassifier = XGBclassifier(vec_path=vecmodel_path, clf_path=clfmodel_path,
                                  output_path=output_path, if_load=0)

    # 2、载入训练数据与预测数据
    jiaba_split = data_deal(dataset=user_path)
    train_data, test_data,dev_data = jiaba_split.get_data(ifsplit_data=1)  # df["text分词结果"，'lable']
    print("训练集测试集生成好了")
    # 3、训练并预测分类正确性
    trainXGB = XGBclassifier.trainXGB(train_data[:,0].astype("str"), train_data[:,1].astype("int"))
    print("模型训练并保存好了")
    # 4、预测数据并输出结果
    print("训练集准确率：")
    XGBclassifier.predictXGB(train_data[:,0].astype("str"), train_data[:,1].astype("int"))
    print("测试集准确率：")
    preds = XGBclassifier.predictXGB(test_data[:,0].astype("str"), test_data[:,1].astype("int"))
    # 5、预测数据保存csv文件
    # XGBclassifier.write_data(preds)
