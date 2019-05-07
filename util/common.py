# coding=utf-8
import numpy as np


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
