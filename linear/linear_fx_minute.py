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

warnings.filterwarnings('ignore')  # 実行上問題ない注意は非表示にする

if __name__ == '__main__':
    args = util.get_args()
    pre_minute = args.minute

    DAY_AGO = 25  # 何日前までのデータを使用するのかを設定
    TECH_NUM = 1 + 4 + 4 + 4  # 終値1本、MVave4本、itimoku4本、ボリンジャー4本
    mvave_list = [5, 21, 34, 144]

    # pandasのDataFrameのままでは扱いにくい+実行速度が遅いため、numpyに変換
    table = np.array(pd.read_csv(args.input))
    table = util.add_technical_values(table, mvave_list)

    # 説明変数、非説明変数を作成
    X = util.generate_explanatory_variables(table, DAY_AGO, TECH_NUM)
    Y = util.generate_dependent_variables(table, X, pre_minute)

    # 正規化
    X, Y = util.normalize(X, Y, DAY_AGO)

    # XとYを学習データとテストデータ(2017年～)に分ける
    m_day = 60 * 24
    train_day = 40
    offset = mvave_list[-1]  # 最初、移動平均分は過去のデータがないと移動平均を取れない

    total_reward = 0
    total_judge = 0
    total_correct = 0
    buy_sell_list = []
    while offset + m_day * (train_day + 1) < len(X) - pre_minute:
        X_train = X[offset: offset + m_day * train_day, :]  # 200日平均を使うので、それ以降を学習データに使用
        Y_train = Y[offset: offset + m_day * train_day]
        X_test = X[offset + m_day * train_day: offset + m_day * (train_day + 1), :]
        Y_test = Y[offset + m_day * train_day: offset + m_day * (train_day + 1)]

        # 学習データを使用して、線形回帰モデルを作成します
        from sklearn import linear_model  # scikit-learnライブラリの関数を使用

        linear_reg_model = linear_model.LinearRegression()

        linear_reg_model.fit(X_train, Y_train)  # モデルに対して、学習データをフィットさせ係数を学習させる

        # 2017年のデータで予想し、グラフで予測具合を見る
        Y_pred = linear_reg_model.predict(X_test)  # 予測する

        result = pd.DataFrame(Y_pred)  # 予測
        result.columns = ['Y_pred']
        result['Y_test'] = Y_test

        # sns.set_style('darkgrid')
        # sns.regplot(x='Y_pred', y='Y_test', data=result)  # plotする

        for y in Y_pred:
            if y > 0:
                buy_sell_list.append(True)
            else:
                buy_sell_list.append(False)

        judge, correct, reward = util.get_result(Y_test=Y_test, Y_pred=Y_pred, output_path=args.output, result=result)
        total_reward = total_reward + reward
        total_judge = total_judge + judge
        total_correct = total_correct + correct
        offset = offset + m_day

    print("予測総数：{0}\t正解数：{1}\t正解率：{2:.3f}\t利益合計：{3:.3f}".format(
        total_judge, total_correct, total_correct / total_judge * 100, total_reward))

    pos = util.Position()
    max_pos = 0
    min_pos = 0
    for i, f in enumerate(buy_sell_list):
        pos.clear()
        if i >= pre_minute:
            for l in buy_sell_list[i - pre_minute:i]:
                if l:
                    pos.buy()
                else:
                    pos.sell()
        if max_pos < pos.get_max_pos():
            max_pos = pos.get_max_pos()
        if min_pos > pos.get_min_pos():
            min_pos = pos.get_min_pos()

    print("最大ポジション: 買い...{0}\t売り...{1}".format(max_pos, min_pos))
