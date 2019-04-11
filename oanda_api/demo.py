# API接続設定のファイルを読み込む
import configparser
import numpy as np
import pandas as pd
import oandapy
import datetime
from datetime import datetime, timedelta
import pytz

# https://github.com/oanda/oandapy

# 文字列 -> datetime
def iso_to_jp(iso):
    date = None
    try:
        date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%fZ')
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%f%z')
            date = date.astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return date

# datetime -> 表示用文字列
def date_to_str(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')

# 設定
config = configparser.ConfigParser()
config.read('./api_key.txt')  # パスの指定が必要です
account_id = config['oanda']['account_id']
api_key = config['oanda']['api_key']

# APIへ接続
oanda = oandapy.API(environment="practice",
                    access_token=api_key)

# # ドル円の現在のレートを取得
# res = oanda.get_prices(instruments="USD_JPY")
#
# # 中身を確認
# print(res)
#
res_hist_1m = oanda.get_history(instrument="USD_JPY", granularity="M1", count="60")

# データフレームへ変換
res_hist_1m = pd.DataFrame(res_hist_1m['candles'])

# 日付をISOから変換
res_hist_1m['time'] = res_hist_1m['time'].apply(lambda x: iso_to_jp(x))
res_hist_1m['time'] = res_hist_1m['time'].apply(lambda x: date_to_str(x))

# 最初の5行を確認してみよう
print(res_hist_1m.head())

data2 = np.array(res_hist_1m)

# print(data2)
