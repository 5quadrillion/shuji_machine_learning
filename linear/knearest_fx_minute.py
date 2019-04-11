# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
import sys
import warnings
import argparse
import datetime
import time
import util
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')  # 実行上問題ない注意は非表示にする

if __name__ == '__main__':
    args = util.get_args()
    pre_minute = args.minute

    LEARN_MINUTE_AGO = 30  # 何分前までのデータを学習に使用するのかを設定
    TECH_NUM = 1 + 4 + 4 + 4  # 終値1本、MVave4本、itimoku4本、ボリンジャー4本
    mvave_list = [5, 21, 34, 144]

    # pandasのDataFrameのままでは扱いにくい+実行速度が遅いため、numpyに変換
    print("データセット作成")
    table = np.array(pd.read_csv(args.input))
    table = util.add_technical_values(table, mvave_list)

    # 説明変数、非説明変数を作成
    print("説明変数、非説明変数を作成")
    X = util.generate_explanatory_variables(table, LEARN_MINUTE_AGO, TECH_NUM)
    Y = util.generate_dependent_variables(table, X, pre_minute)

    # 正規化
    print("正規化開始")
    X, Y = util.normalize(X, Y, LEARN_MINUTE_AGO)

    # XとYを学習データとテストデータ(2017年～)に分ける
    print("学習データとテストデータの分離開始")
    m_day = 60 * 24
    train_day = 100
    train_minute = m_day * train_day

    total_reward = 0
    total_judge = 0
    total_correct = 0

    X_train = X[mvave_list[-1]: mvave_list[-1] + train_minute, :]
    Y_train = Y[mvave_list[-1]: mvave_list[-1] + train_minute]

    print("モデル作成開始")
    model = KNeighborsRegressor(n_neighbors=10)
    model.fit(X_train, Y_train)

    offset = mvave_list[-1] + train_minute

    print("予測開始")
    while offset + 1 + m_day < len(X) - pre_minute:
        X_test = X[offset + 1: offset + 1 + m_day, :]
        Y_test = Y[offset + 1: offset + 1 + m_day]

        Y_pred = model.predict(X_test)  # 予測する

        result = pd.DataFrame(Y_pred)  # 予測
        result.columns = ['Y_pred']
        result['Y_test'] = Y_test

        judge, correct, reward = util.get_result(Y_test=Y_test, Y_pred=Y_pred, output_path=args.output, result=result)
        total_reward = total_reward + reward
        total_judge = total_judge + judge
        total_correct = total_correct + correct
        offset = offset + m_day

    print("予測総数：{0}\t正解数：{1}\t正解率：{2:.3f}\t利益合計：{3:.3f}".format(
        total_judge, total_correct, total_correct / total_judge * 100, total_reward))
