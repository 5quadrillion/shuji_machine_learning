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
import pickle
import os

warnings.filterwarnings('ignore')  # 実行上問題ない注意は非表示にする

if __name__ == '__main__':
    args = util.get_args()
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    pre_minute = args.minute

    LEARN_MINUTE_AGO = 120  # 何分前までのデータを学習に使用するのかを設定
    TECH_NUM = 1 + 4 + 4 + 4  # 終値1本、MVave4本、itimoku4本、ボリンジャー4本
    mvave_list = [5, 21, 34, 144]

    # pandasのDataFrameのままでは扱いにくい+実行速度が遅いため、numpyに変換
    print("データセット作成")
    raw_table = np.array(pd.read_csv(args.input))
    table = np.zeros((len(raw_table), 5))
    table[0:len(table), 0:5] = raw_table[0:len(raw_table), 1:6]  # 日付の列を削除
    # table = util.add_technical_values(table, mvave_list)

    # 説明変数、非説明変数を作成
    print("説明変数、被説明変数を作成")
    X = util.generate_explanatory_variables(table, LEARN_MINUTE_AGO)
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

    # モデル作成
    if os.path.exists(args.model):
        print("既存モデル使用")
        with open(args.model, mode='rb') as fp:
            model = pickle.load(fp)
    else:
        print("モデル作成開始")
        X_train = X[mvave_list[-1]: mvave_list[-1] + train_minute, :]
        Y_train = Y[mvave_list[-1]: mvave_list[-1] + train_minute]

        model = KNeighborsRegressor(n_neighbors=20)
        model.fit(X_train, Y_train)
        with open(args.model, mode='wb') as fp:
            pickle.dump(model, fp)

    offset = mvave_list[-1] + train_minute
    print("予測開始")
    total_min = 0
    total_correct = 0
    total_entry = 0
    total_entry_correct = 0
    total_reward = 0

    counter = 0
    while offset + 1 + m_day < len(X) - pre_minute:
        X_test = X[offset + 1: offset + 1 + m_day, :]
        Y_test = Y[offset + 1: offset + 1 + m_day]

        Y_pred = model.predict(X_test)  # 予測する

        result = pd.DataFrame(Y_pred)  # 予測
        result.columns = ['Y_pred']
        result['Y_test'] = Y_test

        sum_min, correct_num, entry_num, entry_correct_num, reward = util.get_result(
            Y_test=Y_test, Y_pred=Y_pred, out_tsv_path="{0}/{1}.tsv".format(args.outpath, counter))
        total_min += sum_min
        total_correct += correct_num
        total_entry += entry_num
        total_entry_correct += entry_correct_num
        total_reward += reward

        offset = offset + m_day
        counter += 1

    print("予測数: {0}\t正解率: {1:.3f}\tエントリー数: {2}\tエントリー正解率: {3:.3f}\t利益合計：{4:.3f}".format(
        total_min, total_correct / total_min * 100, total_entry, total_entry_correct / total_entry * 100, total_reward))
