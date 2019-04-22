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
from joblib import Parallel, delayed
import tensorflow as tf

warnings.filterwarnings('ignore')  # 実行上問題ない注意は非表示にする

if __name__ == '__main__':
    args = util.get_args()
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    PARALLEL_NUM = 7
    # TECH_NUM = 1 + 4 + 4 + 4  # 終値1本、MVave4本、itimoku4本、ボリンジャー4本
    # mvave_list = [5, 21, 34, 144]

    # pandasのDataFrameのままでは扱いにくい+実行速度が遅いため、numpyに変換
    print("データセット作成")
    # pandas_table = pd.read_csv(args.input, usecols=["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"], sep=",", skipfooter=5615872, engine='python')
    table = np.array(pd.read_csv(args.input, usecols=["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"], sep=",", skipfooter=4500000, engine='python'), dtype=np.float)
    # table = util.add_technical_values(table, mvave_list)

    # 説明変数、非説明変数を作成
    print("説明変数、被説明変数を作成")
    X = util.generate_explanatory_variables(_table=table, _learn_minute_ago=args.learn_minute_ago, n=PARALLEL_NUM)
    Y = util.generate_dependent_variables(table, X, args.predict_minute_later)

    # メモリクリア
    table = None

    # 正規化
    print("正規化開始")
    result_tables = Parallel(n_jobs=PARALLEL_NUM)([delayed(util.normalize2)(x, args.learn_minute_ago) for x in np.array_split(X, PARALLEL_NUM)])
    X = np.vstack(result_tables)

    # XとYを学習データとテストデータ(2017年～)に分ける
    print("学習データとテストデータの分離開始")
    m_day = 60 * 24
    train_day = 500
    train_minute = m_day * train_day

    total_reward = 0
    total_judge = 0
    total_correct = 0

    filename = "KNNmodel/{}_M{}_L{}_N{}.pickle".format(args.model, args.predict_minute_later, args.learn_minute_ago, args.nearest_neighbor)
    # モデル作成
    if os.path.exists(filename):
        print("既存モデル使用")
        with open(filename, mode='rb') as fp:
            model = pickle.load(fp)
    else:
        print("モデル作成開始")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=util.learning_rate)
        with tf.Graph().as_default():
            input_ph = tf.placeholder(tf.float32, [None, util.length_of_sequences, util.num_of_input_nodes],
                                      name="input")
            supervisor_ph = tf.placeholder(tf.float32, [None, util.num_of_output_nodes], name="supervisor")
            istate_ph = tf.placeholder(tf.float32, [None, util.num_of_hidden_nodes * 2], name="istate")

            output_op, states_op, datas_op = util.inference(input_ph, istate_ph)
            loss_op = util.loss(output_op, supervisor_ph)
            training_op = util.training(optimizer, loss_op)

            summary_op = tf.summary.merge_all()
            init = tf.initialize_all_variables()

            with tf.Session() as sess:
                saver = tf.train.Saver()
                summary_writer = tf.summary.FileWriter("/tmp/tensorflow_log", graph=sess.graph)
                sess.run(init)

                for epoch in range(util.num_of_training_epochs):
                    inputs, supervisors = util.get_batch(util.size_of_mini_batch, X, Y)
                    train_dict = {
                        input_ph: inputs,
                        supervisor_ph: supervisors,
                        istate_ph: np.zeros((util.size_of_mini_batch, util.num_of_hidden_nodes * 2)),
                    }
                    sess.run(training_op, feed_dict=train_dict)

                    if (epoch) % 100 == 0:
                        summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                        print("train#%d, train loss: %e" % (epoch, train_loss))
                        summary_writer.add_summary(summary_str, epoch)
                        if (epoch) % 500 == 0:
                            util.calc_accuracy(output_op)

                util.calc_accuracy(output_op, prints=True)
                datas = sess.run(datas_op)
                saver.save(sess, filename)
