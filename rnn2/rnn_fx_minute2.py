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

    # pandasのDataFrameのままでは扱いにくい+実行速度が遅いため、numpyに変換
    print("データセット作成")
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

    sess = tf.compat.v1.InteractiveSession()
    # 再現性の確保のために乱数シードを固定
    tf.compat.v1.set_random_seed(12345)


    filename = "KNNmodel/{}_M{}_L{}_N{}.pickle".format(args.model, args.predict_minute_later, args.learn_minute_ago, args.nearest_neighbor)
    # モデル作成
    print("モデル作成開始")
    X_train = X[0: train_minute, :]
    Y_train = Y[0: train_minute]

    # パラメーター
    # 学習時間長
    SERIES_LENGTH = 72
    # 特徴量数
    FEATURE_COUNT = 3

    # 入力（placeholderメソッドの引数は、データ型、テンソルのサイズ）
    # 訓練データ
    x = tf.compat.v1.placeholder(tf.float32, [None, len(X_train), 3])
    # 教師データ
    y = tf.compat.v1.placeholder(tf.float32, [None, 1])

    # RNNセルの作成
    cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(20)
    initial_state = cell.zero_state(tf.shape(input=x)[0], dtype=tf.float32)
    outputs, last_state = tf.compat.v1.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)

    # 全結合
    # 重み
    w = tf.Variable(tf.zeros([20, 3]))
    # バイアス
    b = tf.Variable([0.1] * 3)
    # 最終出力（予測）
    prediction = tf.matmul(last_state, w) + b

    # 損失関数（平均絶対誤差：MAE）と最適化（Adam）
    loss = tf.reduce_mean(input_tensor=tf.map_fn(tf.abs, y - prediction))
    optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

    # バッチサイズ
    BATCH_SIZE = 16
    # 学習回数
    NUM_TRAIN = 10_000
    # 学習中の出力頻度
    OUTPUT_BY = 500

    # 標準化
    # 学習の実行
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(NUM_TRAIN):
        mae, _ = sess.run([loss, optimizer], feed_dict={x: X_train[i], y: Y_train[i]})
        if i % OUTPUT_BY == 0:
            print('step {:d}, error {:.2f}'.format(i, mae))

    offset = train_minute
    print("予測開始")
    total_min = 0
    total_correct = 0
    total_entry = 0
    total_entry_correct = 0
    total_reward = 0

    counter = 0
    while offset + 1 + m_day < len(X) - args.predict_minute_later:
        X_test = X[offset + 1: offset + 1 + m_day, :]
        Y_test = Y[offset + 1: offset + 1 + m_day]

        Y_pred = prediction.eval({x: X_test[offset + 1 + m_day]})
        print("pred: {0}, correct: {1}".format(Y_pred, Y_test[offset + 1 + m_day]))
