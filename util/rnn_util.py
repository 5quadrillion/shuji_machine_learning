# coding=utf-8
import numpy as np
import tensorflow as tf
import random
import argparse
import datetime
import time
from joblib import Parallel, delayed


USE_COLS = ["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"]
num_of_input_nodes = 1
num_of_hidden_nodes = 80
num_of_output_nodes = 1
num_of_training_epochs = 500000
size_of_mini_batch = 100
num_of_prediction_epochs = 100
forget_bias = 0.8
sequesnce_num = 17


def get_args():
    # 準備
    parser = argparse.ArgumentParser(description='線形回帰で未来チャートの予想をします')

    parser.add_argument("--input", "-i", help="入力ファイル csv", default="../data/USDJPY_minute_20190104.csv")
    parser.add_argument("--outpath", "-o", help="出力ファイル",
                        default="output/result{}.txt".format(int(time.mktime(datetime.datetime.now().timetuple()))))
    parser.add_argument("--learn_minute_ago", "-l", help="何分前までの値を使って学習するか", type=int, default=100)
    parser.add_argument("--predict_minute_later", "-p", help="何分後の値を予想するか", type=int, default=15)
    parser.add_argument("--model", "-m", help="モデルのdumpデータのpath", default="")
    # parser.add_argument("--rate", "-r", help="learning_rate", default=0.01)
    return parser.parse_args()


def generate_explanatory_variables_with_tech(_table, _learn_minute_ago, _technical_num):
    ret_table = np.zeros((len(_table), _learn_minute_ago * _technical_num))
    for s in range(0, _technical_num):  # 日にちごとに横向きに並べる
        for i in range(0, _learn_minute_ago):
            ret_table[i:len(_table), _learn_minute_ago * s + i] = _table[0:len(_table) - i, s + sequesnce_num]
    return ret_table


def generate_explanatory_variables(_table, _learn_minute_ago, n):
    # 並列化のための関数
    def __fix_table_for_exp(in_table, learn_minute_ago, table_items):
        ret_table = np.zeros((len(in_table), learn_minute_ago * table_items))
        for s in range(0, table_items):  # 日にちごとに横向きに並べる
            for i in range(0, learn_minute_ago):
                ret_table[i:len(in_table), learn_minute_ago * s + i] = in_table[0:len(in_table) - i, s]
        return ret_table

    result_tables = Parallel(n_jobs=n)([delayed(__fix_table_for_exp)(in_table, _learn_minute_ago, sequesnce_num) for in_table in np.array_split(_table, n)])
    return np.vstack(result_tables)


def generate_dependent_variables(_table, _predict_minute_later):
    ret_table = np.zeros(len(_table))
    result_column = 0  # start を結果として使用
    ret_table[0:len(ret_table) - _predict_minute_later] = \
        _table[_predict_minute_later:len(_table), result_column] \
        - _table[0:len(_table) - _predict_minute_later, result_column]
    return ret_table


def normalize(_x):
    ret = np.copy(_x)
    for j in range(0, _x.shape[1]):
        tmp_mean = np.mean(_x[0:len(_x), j])  # 平均値
        for i in range(0, len(_x)):
            ret[i, j] = (_x[i, j] - tmp_mean)  # Xを正規化
    return ret


def normalize_with_candlestick(_x, _minute_ago):
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
        in4 = tf.split(in3, sequesnce_num, 0)

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


def get_batch(batch_size, X, y):
    rnum = [random.randint(0, len(X) - 1) for x in range(batch_size)]
    xs = np.array([[[x] for x in list(X[r])] for r in rnum])
    ys = np.array([[y[r]] for r in rnum])
    return xs, ys


def make_prediction(X_test, Y_test, offset):
    return np.array([[[y] for y in x] for x in X_test[offset:offset+size_of_mini_batch]]), np.array([[x] for x in Y_test[offset:offset+size_of_mini_batch]])


def calc_accuracy(X_test, Y_test, output_op, input_ph, supervisor_ph, istate_ph, sess, rate, prints=False):
    correct = 0
    total = 0
    file = open('output.txt', 'a')
    for offset in range(0, len(X_test) - size_of_mini_batch):
        if offset % 100 == 0:
            inputs, ys = make_prediction(X_test, Y_test, offset)
            pred_dict = {
                input_ph:  inputs,
                supervisor_ph: ys,
                istate_ph:    np.zeros((num_of_prediction_epochs, num_of_hidden_nodes * 2)),
            }
            output = sess.run([output_op], feed_dict=pred_dict)

            def print_result(i, p, q):
                # [print(list(x)[0]) for x in i]
                print("offset: {}, output: {}, correct: {}".format(offset, p, q))
            if prints:
                # [print_result(i, p, q) for i, p, q in zip(inputs, output[0], ys)]
                file.write("offset: {}, output: {}, correct: {}".format(offset, output[0][-1], ys[-1]))
                print("offset: {}, output: {}, correct: {}".format(offset, output[0][-1], ys[-1]))

            if output[0][-1]*ys[-1] > 0:
                correct += 1
            total += 1
    print("l_rate: {}, accuracy: {}".format(rate, correct / float(total)))
    file.write("l_rate: {}, accuracy: {}".format(rate, correct / float(total)))
    file.close()
    # return output
