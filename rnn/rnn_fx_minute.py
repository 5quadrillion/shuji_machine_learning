# coding=utf-8
import os
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed

from ..util import common, rnn_util

warnings.filterwarnings('ignore')  # 実行上問題ない注意は非表示にする

if __name__ == '__main__':
    args = rnn_util.get_args()
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    PARALLEL_NUM = 7

    # pandasのDataFrameのままでは扱いにくい+実行速度が遅いため、numpyに変換
    print("データセット作成")
    table = np.array(pd.read_csv(args.input, usecols=rnn_util.USE_COLS, sep=",", skipfooter=1, engine='python'), dtype=np.float)
    result_tables = Parallel(n_jobs=PARALLEL_NUM)([delayed(common.add_technical_values)(x) for x in np.array_split(table, PARALLEL_NUM)])
    table = np.vstack(result_tables)

    # 説明変数、非説明変数を作成
    print("説明変数、被説明変数を作成")
    # X = rnn_util.generate_explanatory_variables(_table=table, _learn_minute_ago=args.learn_minute_ago, n=PARALLEL_NUM)
    X = table
    Y = rnn_util.generate_dependent_variables(table, args.predict_minute_later)

    # メモリクリア
    table = None

    # 正規化
    print("正規化開始")
    result_tables = Parallel(n_jobs=PARALLEL_NUM)([delayed(rnn_util.normalize)(x) for x in np.array_split(X, PARALLEL_NUM)])
    X = np.vstack(result_tables)

    # 学習用とテストように分ける
    learning_len = int(len(X)*0.9)
    X_train = X[0: learning_len, :]
    Y_train = Y[0: learning_len]

    X_test = X[learning_len: -args.predict_minute_later, :]
    Y_test = Y[learning_len: -args.predict_minute_later]

    learning_rate_list = [0.001, 0.003, 0.07, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7]

    for rate in learning_rate_list:
        # モデル作成
        # print("モデル作成開始")
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=float(rate))
        with tf.Graph().as_default():
            input_ph = tf.compat.v1.placeholder(tf.float32, [None, X_train.shape[1], rnn_util.num_of_input_nodes], name="input")
            supervisor_ph = tf.compat.v1.placeholder(tf.float32, [None, rnn_util.num_of_output_nodes], name="supervisor")
            istate_ph = tf.compat.v1.placeholder(tf.float32, [None, rnn_util.num_of_hidden_nodes * 2], name="istate")

            output_op, states_op, datas_op = rnn_util.inference(input_ph, istate_ph)
            loss_op = rnn_util.loss(output_op, supervisor_ph)
            training_op = rnn_util.training(optimizer, loss_op)

            summary_op = tf.compat.v1.summary.merge_all()
            init = tf.compat.v1.initialize_all_variables()

            with tf.compat.v1.Session() as sess:
                saver = tf.compat.v1.train.Saver()
                summary_writer = tf.compat.v1.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)
                sess.run(init)

                for epoch in range(rnn_util.num_of_training_epochs):
                    inputs, supervisors = rnn_util.get_batch(rnn_util.size_of_mini_batch, X_train, Y_train)
                    train_dict = {
                        input_ph: inputs,
                        supervisor_ph: supervisors,
                        istate_ph: np.zeros((rnn_util.size_of_mini_batch, rnn_util.num_of_hidden_nodes * 2)),
                    }
                    sess.run(training_op, feed_dict=train_dict)

                    if (epoch) % 100 == 0:
                        summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                        # print("train#%d, train loss: %e" % (epoch, train_loss))
                        summary_writer.add_summary(summary_str, epoch)
                        # if (epoch) % 500 == 0:
                        # rnn_util.calc_accuracy(X_test, Y_test, output_op, input_ph, supervisor_ph, istate_ph, sess)

                rnn_util.calc_accuracy(X_test, Y_test, output_op, input_ph, supervisor_ph, istate_ph, sess, rate, prints=True)
                datas = sess.run(datas_op)
                filename = "RNNmodel/{}_M{}_L{}.ckpt".format(args.model, args.predict_minute_later,
                                                                   args.learn_minute_ago)
                # saver.save(sess, filename)
