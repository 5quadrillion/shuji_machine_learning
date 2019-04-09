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
    mv_width = mvave_list[-1]
    m_day = 60 * 24
    X_train = X[mv_width:mv_width + m_day * 40, :]  # 200日平均を使うので、それ以降を学習データに使用
    Y_train = Y[mv_width:mv_width + m_day * 40]

    X_test = X[mv_width + m_day * 40:len(X) - pre_minute, :]
    Y_test = Y[mv_width + m_day * 40:len(Y) - pre_minute]

    # 学習データを使用して、線形回帰モデルを作成します
    from sklearn import linear_model  # scikit-learnライブラリの関数を使用

    linear_reg_model = linear_model.LinearRegression()

    linear_reg_model.fit(X_train, Y_train)  # モデルに対して、学習データをフィットさせ係数を学習させる

    # 2017年のデータで予想し、グラフで予測具合を見る
    Y_pred = linear_reg_model.predict(X_test)  # 予測する

    result = pd.DataFrame(Y_pred)  # 予測
    result.columns = ['Y_pred']
    result['Y_test'] = Y_test

    sns.set_style('darkgrid')
    sns.regplot(x='Y_pred', y='Y_test', data=result)  # plotする

    util.show_result(Y_test=Y_test, Y_pred=Y_pred, output_path=args.output, result=result)
