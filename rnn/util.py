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
learning_rate = 0.01
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


# 並列化のための関数
def __fix_table_for_exp(in_table, learn_minute_ago, table_items):
    ret_table = np.zeros((len(in_table), learn_minute_ago * table_items))
    for s in range(0, table_items):  # 日にちごとに横向きに並べる
        for i in range(0, learn_minute_ago):
            ret_table[i:len(in_table), learn_minute_ago * s + i] = in_table[0:len(in_table) - i, s]
    return ret_table


def generate_explanatory_variables(_table, _learn_minute_ago, n):
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


def inference(input_ph, istate_ph):
    with tf.compat.v1.name_scope("inference") as scope:
        weight1_var = tf.Variable(tf.random.truncated_normal(
            [num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
        weight2_var = tf.Variable(tf.random.truncated_normal(
            [num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
        bias1_var = tf.Variable(tf.random.truncated_normal([num_of_hidden_nodes], stddev=0.1), name="bias1")
        bias2_var = tf.Variable(tf.random.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias2")

        in1 = tf.transpose(a=input_ph, perm=[1, 0, 2])
        in2 = tf.reshape(in1, [-1, num_of_input_nodes])
        in3 = tf.matmul(in2, weight1_var) + bias1_var
        in4 = tf.split(in3, length_of_sequences, 0)

        cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_of_hidden_nodes, forget_bias=forget_bias, state_is_tuple=False)
        rnn_output, states_op = tf.compat.v1.nn.static_rnn(cell, in4, initial_state=istate_ph)
        output_op = tf.matmul(rnn_output[-1], weight2_var) + bias2_var

        # Add summary ops to collect data
        w1_hist = tf.compat.v1.summary.histogram("weights1", weight1_var)
        w2_hist = tf.compat.v1.summary.histogram("weights2", weight2_var)
        b1_hist = tf.compat.v1.summary.histogram("biases1", bias1_var)
        b2_hist = tf.compat.v1.summary.histogram("biases2", bias2_var)
        output_hist = tf.compat.v1.summary.histogram("output",  output_op)
        results = [weight1_var, weight2_var, bias1_var,  bias2_var]
        return output_op, states_op, results


def loss(output_op, supervisor_ph):
    with tf.compat.v1.name_scope("loss") as scope:
        square_error = tf.reduce_mean(input_tensor=tf.square(output_op - supervisor_ph))
        loss_op = square_error
        tf.compat.v1.summary.scalar("loss", loss_op)
        return loss_op


def training(optimizer, loss_op):
    with tf.compat.v1.name_scope("training") as scope:
        training_op = optimizer.minimize(loss_op)
        return training_op


def get_batch(batch_size, X, t):
    rnum = [random.randint(0, len(X) - 1) for x in range(batch_size)]
    xs = np.array([[[y] for y in list(X[r])] for r in rnum])
    ts = np.array([[t[r]] for r in rnum])
    return xs, ts


def create_data(nb_of_samples, sequence_len):
    X = np.zeros((nb_of_samples, sequence_len))
    for row_idx in range(nb_of_samples):
        X[row_idx, :] = np.around(np.random.rand(sequence_len)).astype(int)
    # Create the targets for each sequence
    t = np.sum(X, axis=1)
    return X, t


def make_prediction(nb_of_samples):
    sequence_len = 10
    xs, ts = create_data(nb_of_samples, sequence_len)
    return np.array([[[y] for y in x] for x in xs]), np.array([[x] for x in ts])


def calc_accuracy(output_op, input_ph, supervisor_ph, istate_ph, sess, prints=False):
    inputs, ts = make_prediction(num_of_prediction_epochs)
    pred_dict = {
        input_ph:  inputs,
        supervisor_ph: ts,
        istate_ph:    np.zeros((num_of_prediction_epochs, num_of_hidden_nodes * 2)),
    }
    output = sess.run([output_op], feed_dict=pred_dict)

    def print_result(i, p, q):
        [print(list(x)[0]) for x in i]
        print("output: %f, correct: %d" % (p, q))
    if prints:
        [print_result(i, p, q) for i, p, q in zip(inputs, output[0], ts)]

    opt = abs(output - ts)[0]
    total = sum([1 if x[0] < 0.05 else 0 for x in opt])
    print("accuracy %f" % (total / float(len(ts))))
    return output