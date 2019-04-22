# coding=utf-8
import numpy as np
import argparse
import datetime
import time
import tensorflow as tf
import random


num_of_input_nodes = 1
num_of_hidden_nodes = 80
num_of_output_nodes = 1
length_of_sequences = 10
num_of_training_epochs = 5000
size_of_mini_batch = 100
num_of_prediction_epochs = 100
learning_rate = 0.01
forget_bias = 0.8
num_of_sample = 1000


def get_args():
    # 準備
    parser = argparse.ArgumentParser(description='線形回帰で未来チャートの予想をします')

    parser.add_argument("--input", "-i", help="入力ファイル csv", default="../data/USDJPY_minute_20190104.csv")
    parser.add_argument("--outpath", "-o", help="出力ファイル",
                        default="output".format(int(time.mktime(datetime.datetime.now().timetuple()))))
    parser.add_argument("--minute", "-M", help="何分後の値を予想するか", type=int, default=30)
    parser.add_argument("--model", "-m", help="モデルのdumpデータのpath", default="./model.pickle")
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


def inference(input_ph, istate_ph):
    with tf.name_scope("inference") as scope:
        weight1_var = tf.Variable(tf.truncated_normal(
            [num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
        weight2_var = tf.Variable(tf.truncated_normal(
            [num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
        bias1_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=0.1), name="bias1")
        bias2_var = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias2")

        in1 = tf.transpose(input_ph, [1, 0, 2])
        in2 = tf.reshape(in1, [-1, num_of_input_nodes])
        in3 = tf.matmul(in2, weight1_var) + bias1_var
        in4 = tf.split(in3, length_of_sequences, 0)

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=False)
        rnn_output, states_op = tf.contrib.rnn.static_rnn(cell, in4, initial_state=istate_ph)
        output_op = tf.matmul(rnn_output[-1], weight2_var) + bias2_var

        # Add summary ops to collect data
        w1_hist = tf.summary.histogram("weights1", weight1_var)
        w2_hist = tf.summary.histogram("weights2", weight2_var)
        b1_hist = tf.summary.histogram("biases1", bias1_var)
        b2_hist = tf.summary.histogram("biases2", bias2_var)
        output_hist = tf.summary.histogram("output",  output_op)
        results = [weight1_var, weight2_var, bias1_var,  bias2_var]
        return output_op, states_op, results


def loss(output_op, supervisor_ph):
    with tf.name_scope("loss") as scope:
        square_error = tf.reduce_mean(tf.square(output_op - supervisor_ph))
        loss_op = square_error
        tf.summary.scalar("loss", loss_op)
        return loss_op


def training(optimizer, loss_op):
    with tf.name_scope("training") as scope:
        training_op = optimizer.minimize(loss_op)
        return training_op


def get_batch(batch_size, X, t):
    rnum = [random.randint(0, len(X) - 1) for x in range(batch_size)]
    xs = np.array([[[y] for y in list(X[r])] for r in rnum])
    ts = np.array([[t[r]] for r in rnum])
    return xs, ts
