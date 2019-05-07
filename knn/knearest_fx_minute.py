# coding=utf-8
import numpy as np
import pandas as pd
import warnings
from sklearn.neighbors import KNeighborsRegressor
import pickle
import os
from joblib import Parallel, delayed
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import knn_util

warnings.filterwarnings('ignore')  # 実行上問題ない注意は非表示にする

if __name__ == '__main__':
    args = knn_util.get_args()
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    PARALLEL_NUM = 7
    # TECH_NUM = 1 + 4 + 4 + 4  # 終値1本、MVave4本、itimoku4本、ボリンジャー4本
    # mvave_list = [5, 21, 34, 144]

    # pandasのDataFrameのままでは扱いにくい+実行速度が遅いため、numpyに変換
    print("データセット作成")
    # pandas_table = pd.read_csv(args.input, usecols=["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"], sep=",", skipfooter=5615872, engine='python')
    table = np.array(pd.read_csv(args.input, usecols=["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"], sep=",", skipfooter=4500000, engine='python'), dtype=np.float)
    # table = common.add_technical_values(table, mvave_list)

    # 説明変数、非説明変数を作成
    print("説明変数、被説明変数を作成")
    X = knn_util.generate_explanatory_variables(_table=table, _learn_minute_ago=args.learn_minute_ago, n=PARALLEL_NUM)
    Y = knn_util.generate_dependent_variables(table, X, args.predict_minute_later)

    # メモリクリア
    table = None

    # 正規化
    print("正規化開始")
    result_tables = Parallel(n_jobs=PARALLEL_NUM)([delayed(knn_util.normalize2)(x, args.learn_minute_ago) for x in np.array_split(X, PARALLEL_NUM)])
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
        X_train = X[0: train_minute, :]
        Y_train = Y[0: train_minute]

        model = KNeighborsRegressor(n_neighbors=args.nearest_neighbor)
        model.fit(X_train, Y_train)
        with open(filename, mode='wb') as fp:
            pickle.dump(model, fp)

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

        result_tables = Parallel(n_jobs=PARALLEL_NUM)(
            [delayed(model.predict)(x) for x in np.array_split(X_test, PARALLEL_NUM)])
        Y_pred = np.hstack(result_tables)
        # Y_pred = model.predict(X_test)

        # result = pd.DataFrame(Y_pred)  # 予測
        # result.columns = ['Y_pred']
        # result['Y_test'] = Y_test

        sum_min, correct_num, entry_num, entry_correct_num, reward = knn_util.get_result(
            Y_test=Y_test, Y_pred=Y_pred, out_tsv_path="{0}/{1}.tsv".format(args.outpath, counter))
        total_min += sum_min
        total_correct += correct_num
        total_entry += entry_num
        total_entry_correct += entry_correct_num
        total_reward += reward

        offset = offset + m_day
        counter += 1

    print("予測数: {0}\t正解率: {1:.3f}\tエントリー数: {2}\tエントリー正解率: {3:.3f}\t利益合計：{4:.3f}".format(
        total_min, total_correct / total_min * 100, total_entry, total_entry_correct / total_entry * 100, total_reward))
