# coding=utf-8
import numpy as np
import argparse
import datetime
import time
from joblib import Parallel, delayed
import tensorflow as tf
import random


num_of_input_nodes = 1
num_of_hidden_nodes = 80
num_of_output_nodes = 1
length_of_sequences = 3
num_of_training_epochs = 5000
size_of_mini_batch = 100
num_of_prediction_epochs = 100
learning_rate = 0.5
forget_bias = 0.8


def get_args():
    # 準備
    parser = argparse.ArgumentParser(description='線形回帰で未来チャートの予想をします')

    parser.add_argument("--input", "-i", help="入力ファイル csv", default="../data/USDJPY_minute_20190104.csv")
    parser.add_argument("--outpath", "-o", help="出力ファイル",
                        default="output/result{}.txt".format(int(time.mktime(datetime.datetime.now().timetuple()))))
    parser.add_argument("--learn_minute_ago", "-l", help="何分前までの値を使って学習するか", type=int, default=120)
    parser.add_argument("--predict_minute_later", "-p", help="何分後の値を予想するか", type=int, default=30)
    parser.add_argument("--nearest_neighbor", "-n", help="何要素近傍まで結果に寄与させるか", type=int, default=20)
    parser.add_argument("--model", "-m", help="モデルのdumpデータのpath", default="")
    return parser.parse_args()


def add_technical_values(_table, _moving_average_list=[5, 21, 34, 144]):
    # 5分移動平均線を追加
    _table = np.c_[_table, np.zeros((len(_table), 1))]  # 列の追加
    ave = _moving_average_list[0]
    for i in range(ave, len(_table)):
        tmp = _table[i - ave + 1:i + 1, 4].astype(np.float)  # pythonは0番目からindexが始まります
        _table[i, 5] = np.mean(tmp)

    # 21分移動平均線を追加
    _table = np.c_[_table, np.zeros((len(_table), 1))]
    ave = _moving_average_list[1]
    for i in range(ave, len(_table)):
        tmp = _table[i - ave + 1:i + 1, 4].astype(np.float)
        _table[i, 6] = np.mean(tmp)

    # 34分移動平均線を追加
    _table = np.c_[_table, np.zeros((len(_table), 1))]  # 列の追加
    ave = _moving_average_list[2]
    for i in range(ave, len(_table)):
        tmp = _table[i - ave + 1:i + 1, 4].astype(np.float)
        _table[i, 7] = np.mean(tmp)

    # 144分移動平均線を追加
    _table = np.c_[_table, np.zeros((len(_table), 1))]  # 列の追加
    ave = _moving_average_list[3]
    for i in range(ave, len(_table)):
        tmp = _table[i - ave + 1:i + 1, 4].astype(np.float)
        _table[i, 8] = np.mean(tmp)

    # 一目均衡表を追加 (9,26,52)
    para1 = 9
    para2 = 26
    para3 = 52

    # 転換線 = （過去(para1)日間の高値 + 安値） ÷ 2
    _table = np.c_[_table, np.zeros((len(_table), 1))]  # 列の追加
    for i in range(para1, len(_table)):
        tmp_high = _table[i - para1 + 1:i + 1, 2].astype(np.float)
        tmp_low = _table[i - para1 + 1:i + 1, 3].astype(np.float)
        _table[i, 9] = (np.max(tmp_high) + np.min(tmp_low)) / 2

    # 基準線 = （過去(para2)日間の高値 + 安値） ÷ 2
    _table = np.c_[_table, np.zeros((len(_table), 1))]
    for i in range(para2, len(_table)):
        tmp_high = _table[i - para2 + 1:i + 1, 2].astype(np.float)
        tmp_low = _table[i - para2 + 1:i + 1, 3].astype(np.float)
        _table[i, 10] = (np.max(tmp_high) + np.min(tmp_low)) / 2

    # 先行スパン1 = ｛ （転換値+基準値） ÷ 2 ｝を(para2)日先にずらしたもの
    _table = np.c_[_table, np.zeros((len(_table), 1))]
    for i in range(0, len(_table) - para2):
        tmp = (_table[i, 9] + _table[i, 10]) / 2
        _table[i + para2, 11] = tmp

    # 先行スパン2 = ｛ （過去(para3)日間の高値+安値） ÷ 2 ｝を(para2)日先にずらしたもの
    _table = np.c_[_table, np.zeros((len(_table), 1))]
    for i in range(para3, len(_table) - para2):
        tmp_high = _table[i - para3 + 1:i + 1, 2].astype(np.float)
        tmp_low = _table[i - para3 + 1:i + 1, 3].astype(np.float)
        _table[i + para2, 12] = (np.max(tmp_high) + np.min(tmp_low)) / 2

    # 25日ボリンジャーバンド（±1, 2シグマ）を追加
    parab = 25
    _table = np.c_[_table, np.zeros((len(_table), 4))]  # 列の追加
    for i in range(parab, len(_table)):
        tmp = _table[i - parab + 1:i + 1, 4].astype(np.float)
        _table[i, 13] = np.mean(tmp) + 1.0 * np.std(tmp)
        _table[i, 14] = np.mean(tmp) - 1.0 * np.std(tmp)
        _table[i, 15] = np.mean(tmp) + 2.0 * np.std(tmp)
        _table[i, 16] = np.mean(tmp) - 2.0 * np.std(tmp)
    return _table


def generate_explanatory_variables_with_tech(_table, _learn_minute_ago, _technical_num):
    table_items = 5 # start/high/low/end/vol
    ret_table = np.zeros((len(_table), _learn_minute_ago * _technical_num))
    for s in range(0, _technical_num):  # 日にちごとに横向きに並べる
        for i in range(0, _learn_minute_ago):
            ret_table[i:len(_table), _learn_minute_ago * s + i] = _table[0:len(_table) - i, s + table_items]
    return ret_table


def generate_explanatory_variables(_table, _learn_minute_ago, n):
    # 並列化のための関数
    def __fix_table_for_exp(in_table, learn_minute_ago, table_items):
        ret_table = np.zeros((len(in_table), learn_minute_ago * table_items))
        for s in range(0, table_items):  # 日にちごとに横向きに並べる
            for i in range(0, learn_minute_ago):
                ret_table[i:len(in_table), learn_minute_ago * s + i] = in_table[0:len(in_table) - i, s]
        return ret_table

    table_items = 5 # start/high/low/end/vol
    result_tables = Parallel(n_jobs=n)([delayed(__fix_table_for_exp)(in_table, _learn_minute_ago, table_items) for in_table in np.array_split(_table, n)])
    return np.vstack(result_tables)


def generate_dependent_variables(_table, explanatory_table, _predict_minute_later):
    ret_table = np.zeros(len(_table))
    result_column = 0  # start を結果として使用
    ret_table[0:len(ret_table) - _predict_minute_later] = \
        explanatory_table[_predict_minute_later:len(explanatory_table), result_column] \
        - explanatory_table[0:len(explanatory_table) - _predict_minute_later, result_column]
    return ret_table


def normalize(_x, _minute_ago):
    ret = np.copy(_x)
    for i in range(_minute_ago, len(_x)):
        tmp_mean = np.mean(_x[i - _minute_ago + 1:i + 1, 0])  # 平均値
        for j in range(0, _x.shape[1]):
            ret[i, j] = (_x[i, j] - tmp_mean)  # Xを正規化
    return ret


def normalize2(_x, _minute_ago):
    ret = np.zeros((len(_x), 3))
    # "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"
    for i in range(0, len(_x)):
        ret[i, 0] = _x[i, 0] - _x[i, 3]  # open/closeの差
        if ret[i, 0] >= 0:
            ret[i, 1] = _x[i, 1] - _x[i, 0] # 上髭の長さ
            ret[i, 2] = _x[i, 2] - _x[i, 3] # 下髭の長さ（マイナス）
    return ret
