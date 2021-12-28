## xgboost一般选用train来训练，方便调参这些
### 参数解释
### XGBoost模型主要参数
XGBoost所有的参数分成了三类：通用参数：宏观函数控制；Booster参数：控制每一步的booster；目标参数：控制训练目标的表现。 
####（1）通用参数
* booster[默认gbtree]：gbtree：基于树的模型、gbliner：线性模型
* silent[默认0]：值为1时，静默模式开启，不会输出任何信息
* nthread[默认值为最大可能的线程数]：这个参数用来进行多线程控制，应当输入系统的核数。 如果你希望使用CPU全部的核，那就不要输入这个参数，算法会自动检测它
####（2）Booster参数
这里只介绍tree booster，因为它的表现远远胜过linear booster，所以linear booster很少用到
* eta[默认0.3]：和GBM中的 learning rate 参数类似。 通过减少每一步的权重，可以提高模型的鲁棒性。常用的值为0.2, 0.3
* max_depth[默认6]：这个值为树的最大深度。max_depth越大，模型会学到更具体更局部的样本。常用的值为6
* gamma[默认0]：Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关。
* subsample[默认1]：这个参数控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。 常用的值：0.7-1
* colsample_bytree[默认1]：用来控制每棵随机采样的列数的占比(每一列是一个特征)。 常用的值：0.7-1
####（3）学习目标参数
* objective[默认reg:linear]：这个参数定义需要被最小化的损失函数。
  binary:logistic二分类的逻辑回归，返回预测的概率。
  multi:softmax 使用softmax的多分类器，返回预测的类别。这种情况下，还需要多设一个参数：
  num_class：类别数目
  multi：softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。
* eval_metric[默认值取决于objective 参数的取值]：对于有效数据的度量方法。 对于回归问题，默认值是rmse，对于分类问题，默认值是error。其他的值：rmse 均方根误差； mae 平均绝对误差；logloss 负对数似然函数值；error 二分类错误率(阈值为0.5)； merror 多分类错误率；mlogloss 多分类logloss损失函数；auc 曲线下面积。
* seed[默认0]：随机数的种子 设置它可以复现随机数据的结果。
