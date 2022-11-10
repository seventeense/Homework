import matplotlib.pyplot as plt
import pandas
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    dataframe = pandas.read_csv("train.csv")
    dataframe_test = pandas.read_csv("test.csv")

    X = dataframe.iloc[:, 1:241].astype(float)
    Y = dataframe.iloc[:, 241]
    X_test = dataframe_test.iloc[:, 1:241].astype(float)

    #2.对训练数据进行重构
    data_new = list()
    for i in range(len(X)):    # len(X) = 210
        data_new.append(X.loc[i])    # 获取一行T0-T239的值
    data_new = np.array(data_new).reshape(-1, 1)   # 转换成1列
    time_id = np.tile(np.array([i for i in range(0, 240)]), len(X)).reshape(-1, 1)  # 即表头0-239
    id_index = np.array([i for i in range(0, 210)]).repeat(240).reshape(-1, 1)    # 即0-209行的表头
    data_format = pandas.DataFrame(np.concatenate([id_index, time_id, data_new], axis=1))   # np.concatenate()是用来对数列或矩阵进行合并的
    data_format.columns = ['id', 'time', 'time_series']

    # 3.从训练数据中提取特征
    from tsfresh import extract_features
    extracted_features = extract_features(data_format, column_id="id", column_sort="time")

    # 特征筛选
    # 过滤特征之前要先去除非数(NaN) 利用impute函数
    # 数据清洗(Tidy Data)，是对数据进行重新审查和校验的过程，目的在于删除重复信息、纠正存在的错误，并提供数据一致性。
    from tsfresh import select_features
    from tsfresh.utilities.dataframe_functions import impute
    impute(extracted_features)

    # features_filtered是提取出来的67个有用的特征 实现了数据降维
    features_filtered = select_features(extracted_features, Y, fdr_level=0.6)  # 用select_features函数过滤特征
    print(features_filtered)
    # 该方法从这个过滤后的特征矩阵的列名构造 kind_to_fc_parameters 字典，以确保只提取相关的特征。
    from tsfresh import feature_extraction
    kind_to_fc_parameters = feature_extraction.settings.from_columns(features_filtered)

    # 4.对测试数据进行重构并且提取相同的特征  方法同上
    data_new = list()
    for i in range(len(X_test)):
        data_new.append(X_test.loc[i])
    data_new = np.array(data_new).reshape(-1, 1)
    time_id = np.tile(np.array([i for i in range(0, 240)]), len(X_test)).reshape(-1, 1)
    id_index = np.array([i for i in range(0, 104)]).repeat(240).reshape(-1, 1)
    data_format_test = pandas.DataFrame(np.concatenate([id_index, time_id, data_new], axis=1))
    data_format_test.columns = ['id', 'time', 'time_series']

    features_filtered_test = extract_features(data_format_test, column_id="id", column_sort="time",
                                              kind_to_fc_parameters=kind_to_fc_parameters)

    features_filtered_test = features_filtered_test[features_filtered.columns]
    print(features_filtered_test)

    # 可以看出有些特征的名称是不符合lgb处理时的特征命名规范的，因此我们需要在后面训练lgb之前修改特征名称
    new_col = ['fea%s' % i for i in range(142)]
    features_filtered_test.columns = new_col
    features_filtered.columns = new_col

    # 下面根据上节提取的特征，使用lgb模型进行预测
    # 定义10折交叉验证和lgb参数等
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    # 交叉验证 n_splits 表示要分割多少个子集 shuffle：是否打乱顺序 random_state:随机状态
    num_folds = 10
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2020)
    test_result = np.zeros(len(features_filtered_test))  # a行b列 存放预测的结果
    auc_score = 0

    params = {'num_leaves': 9,  # 一棵树中最大的叶子数 ，默认31
              'objective': 'regression',  # 用LGB回归模型
              'max_depth': int(4),   # 树的最大深度 当模型过拟合时，需要首先考虑
              'learning_rate': 0.08,  # 学习率
              'boosting': 'gbdt',  # 要用的算法：gbdt：梯度提升决策树
              'feature_fraction': 0.8,  # 在每次迭代中随机选择80％的参数来建树
              'bagging_freq': int(2),  # bagging的次数。0表示禁用bagging，非零值表示执行k次bagging
              'bagging_fraction': 1,  # 不进行重采样的情况下随机选择部分数据
              'bagging_seed': 8,  # 一个整数，表示bagging的随机数种子，默认为 3
              'lambda_l1':  0.01,  # 指定正则化0-1
              'lambda_l2': 0.01,
              'metric': 'auc',  # 评价函数选择 ROC（Receiver Operating Characteristic）曲线是以假正率（FPR）和真正率（TPR）为轴的曲线，ROC曲线下面的面积我们叫做AUC，
              "random_state": 2020,  # 随机数种子，可以防止每次运行的结果不一致
              "verbose": -1,
              }

    # 模型的训练和预测
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(features_filtered, Y)):
        print("Fold: ", fold_ + 1)
        X_train, y_train = features_filtered.iloc[trn_idx], Y.iloc[trn_idx]
        X_valid, y_valid = features_filtered.iloc[val_idx], Y.iloc[val_idx]
        # 为lightgbm准备Dataset格式数据
        trn_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_valid, y_valid, reference=trn_data)

        # 模型训练
        clf = lgb.train(params,  # 超参数设置
                        trn_data,  # 训练数据
                        10000,  # 循环的轮数
                        valid_sets=val_data,  # 验证数据 训练期间要评估的数据列表。
                        verbose_eval=50,  # verbose_eval可以是bool类型，也可以是整数类型。如果设置为整数，则每间隔verbose_eval次迭代就输出一次信息。
                        early_stopping_rounds=50)  # 如果50轮后没有提升，就停止循环
        y_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
        auc = roc_auc_score(y_valid, y_pred)
        print(auc)  # 一种模型评估指标
        auc_score += auc
        preds = clf.predict(features_filtered_test, num_iteration=clf.best_iteration)  # 指定选用最好的迭代次数。此处才是对测试集进行测试！
        test_result += preds

    auc_score = auc_score / folds.n_splits
    print("AUC score: ", auc_score)
    test_result = test_result / folds.n_splits
    Y_test = np.round(test_result)

    id_ = range(210, 314)
    df = pd.DataFrame({'ID': id_, 'CLASS': Y_test})
    df.to_csv("data.csv", index=False)

