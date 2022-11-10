import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.svm import SVR, SVC

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 分离数据集
# 交叉验证用在数据不是很充足的时候。比如在平时的项目中，对于普通适中问题，如果数据样本量小于一万条。
# 则采用交叉验证来训练优化选择模型。如果样本大于一万条，一般随机把样本数据分成三份，一份为训练集,一份为验证集，最后一份为测试集。
# 用训练集来训练模型，用验证集来评估模型预测的好坏和选择模型及其对应的参数。把最终得到的模型再用于测试集，最终决定使用哪个模型以及对应参数。

X_train_c = train.drop(['ID', 'CLASS'], axis=1).values  # 删除ID和CLASS列的值 列的表示axis=1 默认为0表示行
y_train_c = train['CLASS'].values   # y为class列对应的值
X_test_c = test.drop(['ID'], axis=1).values  # 删除ID列的值  注意：test中没有class的值
kf = KFold(n_splits=5, shuffle=True, random_state=2022)  # 交叉验证 n_splits 表示要分割多少个子集 shuffle：是否打乱顺序 random_state:随机状态
prediction1 = np.zeros((len(X_test_c),))  # a行b列

i = 0

for train_index, valid_index in kf.split(X_train_c, y_train_c):  # train_index + valid_index =209
    X_train, label_train = X_train_c[train_index], y_train_c[train_index]
    X_valid, label_valid = X_train_c[valid_index], y_train_c[valid_index]
    clf = SVC(C=0.9, kernel='poly', degree=2, gamma='scale', coef0=0.0, shrinking=True, probability=True,
              tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
              break_ties=False, random_state=None)
    # kernel='rbf' 即选择多项式核函数 创建一个svr模型
    # C是惩罚系数,就是说你对误差的宽容度,这个值越高，说明你越不能容忍出现误差
    # gamma是你选择径向基函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布
    clf.fit(X_train, label_train)  # 用分类器去拟合分类器模型
    x1 = clf.predict(X_valid)  # 用训练好的数据去预测x
    y1 = clf.predict(X_test_c)
    prediction1 += y1 / 5   # n_fold = 5
    i += 1
result1 = np.round(prediction1)  # 四舍五入取整  更偏向于1还是0
id_ = range(210, 314)  # 要求提交的数据210-314的分类
df = pd.DataFrame({'ID': id_, 'CLASS': result1})
df.to_csv("baseline.csv", index=False)



