# coding=utf-8
import numpy as np
import argparse
import datetime
import time


def get_args():
    # 準備
    parser = argparse.ArgumentParser(description='線形回帰で未来チャートの予想をします')

    parser.add_argument("--input", "-i", help="入力ファイル csv", default="../data/USDJPY_minute_20190104.csv")
    parser.add_argument("--output", "-o", help="出力ファイル",
                        default="output/result{}.txt".format(int(time.mktime(datetime.datetime.now().timetuple()))))
    parser.add_argument("--minute", "-m", help="何分後の値を予想するか", type=int, default=120)
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


def generate_explanatory_variables(_table, _day_ago, _technical_num):
    ret_table = np.zeros((len(_table), _day_ago * _technical_num))
    for s in range(0, _technical_num):  # 日にちごとに横向きに並べる
        for i in range(0, _day_ago):
            ret_table[i:len(_table), _day_ago * s + i] = _table[0:len(_table) - i, s + 4]
    return ret_table


def generate_dependent_variables(_table, explanatory_table, _pre_minute):
    ret_table = np.zeros(len(_table))
    ret_table[0:len(ret_table) - _pre_minute] = \
        explanatory_table[_pre_minute:len(explanatory_table), 0] \
        - explanatory_table[0:len(explanatory_table) - _pre_minute, 0]
    return ret_table


def normalize(_x, _y, _day_ago):
    original_X = np.copy(_x)  # コピーするときは、そのままイコールではダメ
    tmp_mean = np.zeros(len(_x))

    for i in range(_day_ago, len(_x)):
        tmp_mean[i] = np.mean(original_X[i - _day_ago + 1:i + 1, 0])  # DAY_AGO 日分の平均値
        for j in range(0, _x.shape[1]):
            _x[i, j] = (_x[i, j] - tmp_mean[i])  # Xを正規化
        _y[i] = _y[i]  # X同士の引き算しているので、Yはそのまま
    return _x, _y


def get_result(Y_test, Y_pred, output_path, result):
    # 正答率を計算
    correct_num = 0
    for i in range(len(Y_pred)):
        if Y_pred[i] * Y_test[i] >= 0:
            correct_num += 1

    # 予測結果の合計を計算ーーーーーーーーー
    # 前々日終値に比べて前日終値が高い場合は、買いとする
    reward = 0

    for i in range(0, len(Y_test)):
        if Y_pred[i] >= 0:
            reward += Y_test[i]
        else:
            reward -= Y_test[i]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("予測：")
        for pred in Y_pred:
            f.write("\t" + str(round(pred, 4)))
        f.write("\n実際：")
        for test in Y_test:
            f.write("\t" + str(round(test, 4)))
            # f.write("実際\t{0}".format(Y_test))

    print("予測数：{0}\t正解数：{1}\t正解率：{2:.3f}\t利益合計：{3:.3f}".format(
        len(Y_pred), correct_num, correct_num / len(Y_pred) * 100, reward))

    return len(Y_pred), correct_num, reward

    # # 予測結果の総和グラフを描くーーーーーーーーー
    # total_return = np.zeros(len(Y_test))
    #
    # if Y_pred[0] >= 0:  # 2017年の初日を格納
    #     total_return[0] = Y_test[0]
    # else:
    #     total_return[0] = -Y_test[0]
    #
    # for i in range(1, len(result)):  # 2017年の2日以降を格納
    #     if Y_pred[i] >= 0:
    #         total_return[i] = total_return[i - 1] + Y_test[i]
    #     else:
    #         total_return[i] = total_return[i - 1] - Y_test[i]

    # plt.cla()
    # plt.plot(total_return)
    # plt.show()


class Position:
    __pos = 0
    __max_pos = 0
    __min_pos = 0

    def __init__(self):
        pass

    def buy(self):
        self.__pos += 1
        if self.__max_pos < self.__pos:
            self.__max_pos = self.__pos

    def sell(self):
        self.__pos -= 1
        if self.__min_pos > self.__pos:
            self.__min_pos = self.__pos

    def get_max_pos(self):
        return self.__max_pos

    def get_min_pos(self):
        return self.__min_pos

    def clear(self):
        self.__pos = 0
        self.__min_pos = 0
        self.__max_pos = 0
