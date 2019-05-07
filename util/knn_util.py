# coding=utf-8
import numpy as np
import argparse
import datetime
import time


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


def generate_explanatory_variables(_table, _learn_minute_ago, _technical_num):
    ret_table = np.zeros((len(_table), _learn_minute_ago * _technical_num))
    for s in range(0, _technical_num):  # 日にちごとに横向きに並べる
        for i in range(0, _learn_minute_ago):
            ret_table[i:len(_table), _learn_minute_ago * s + i] = _table[0:len(_table) - i, s + 4]
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


def get_result(Y_test, Y_pred, out_tsv_path):
    # 正答率を計算
    correct_num = 0
    for i in range(len(Y_pred)):
        if Y_pred[i] * Y_test[i] >= 0:
            correct_num += 1

    # 予測結果の合計を計算ーーーーーーーーー
    # 前々日終値に比べて前日終値が高い場合は、買いとする
    reward = 0

    entry_num = 0
    entry_correct_num = 0
    for i in range(0, len(Y_test)):
        if Y_pred[i] >= 0.05:
            reward += Y_test[i]
            entry_num += 1
            if Y_test[i] >= 0:
                entry_correct_num += 1
        if Y_pred[i] <= -0.05:
            reward -= Y_test[i]
            entry_num += 1
            if Y_test[i] <= 0:
                entry_correct_num += 1

    with open(out_tsv_path, "w", encoding="utf-8") as f:
        f.write("予測")
        for pred in Y_pred:
            f.write("\t" + str(round(pred, 4)))
        f.write("\n実際")
        for test in Y_test:
            f.write("\t" + str(round(test, 4)))
            # f.write("実際\t{0}".format(Y_test))

    print("予測数: {0}\t正解率: {1:.3f}\tエントリー数: {2}\tエントリー正解率: {3:.3f}\t利益合計：{4:.3f}".format(
        len(Y_pred), correct_num / len(Y_pred) * 100, entry_num, entry_correct_num / entry_num * 100, reward))

    return len(Y_pred), correct_num, entry_num, entry_correct_num, reward

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
