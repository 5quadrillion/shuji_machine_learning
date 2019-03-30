# import関連
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
import sys

import warnings
warnings.filterwarnings('ignore') # 実行上問題ない注意は非表示にする

# dataフォルダの場所を各自指定してください
data_dir = "../data/"
data = pd.read_csv(data_dir + "USDJPY_minute_20190104.csv") # FXデータの読み込み（データは同じリポジトリのdataフォルダに入っています）
data.head() # データの概要を見てみます

# pandasのDataFrameのままでは、扱いにくい+実行速度が遅いので、numpyに変換して処理します
data2 = np.array(data)

# 5日移動平均線を追加します
data2 = np.c_[data2, np.zeros((len(data2),1))] # 列の追加
ave_day = 5
for i in range(ave_day, len(data2)):
    tmp =data2[i-ave_day+1:i+1,4].astype(np.float) # pythonは0番目からindexが始まります
    data2[i,5] = np.mean(tmp)
# 25日移動平均線を追加します
data2 = np.c_[data2, np.zeros((len(data2), 1))]
ave_day = 25
for i in range(ave_day, len(data2)):
    tmp = data2[i - ave_day + 1:i + 1, 4].astype(np.float)
    data2[i, 6] = np.mean(tmp)

# 75日移動平均線を追加します
data2 = np.c_[data2, np.zeros((len(data2), 1))]  # 列の追加
ave_day = 75
for i in range(ave_day, len(data2)):
    tmp = data2[i - ave_day + 1:i + 1, 4].astype(np.float)
    data2[i, 7] = np.mean(tmp)

# 200日移動平均線を追加します
data2 = np.c_[data2, np.zeros((len(data2), 1))]  # 列の追加
ave_day = 200
for i in range(ave_day, len(data2)):
    tmp = data2[i - ave_day + 1:i + 1, 4].astype(np.float)
    data2[i, 8] = np.mean(tmp)

# 一目均衡表を追加します (9,26,52)
para1 = 9
para2 = 26
para3 = 52

# 転換線 = （過去(para1)日間の高値 + 安値） ÷ 2
data2 = np.c_[data2, np.zeros((len(data2), 1))]  # 列の追加
for i in range(para1, len(data2)):
    tmp_high = data2[i - para1 + 1:i + 1, 2].astype(np.float)
    tmp_low = data2[i - para1 + 1:i + 1, 3].astype(np.float)
    data2[i, 9] = (np.max(tmp_high) + np.min(tmp_low)) / 2

# 基準線 = （過去(para2)日間の高値 + 安値） ÷ 2
data2 = np.c_[data2, np.zeros((len(data2), 1))]
for i in range(para2, len(data2)):
    tmp_high = data2[i - para2 + 1:i + 1, 2].astype(np.float)
    tmp_low = data2[i - para2 + 1:i + 1, 3].astype(np.float)
    data2[i, 10] = (np.max(tmp_high) + np.min(tmp_low)) / 2

# 先行スパン1 = ｛ （転換値+基準値） ÷ 2 ｝を(para2)日先にずらしたもの
data2 = np.c_[data2, np.zeros((len(data2), 1))]
for i in range(0, len(data2) - para2):
    tmp = (data2[i, 9] + data2[i, 10]) / 2
    data2[i + para2, 11] = tmp

# 先行スパン2 = ｛ （過去(para3)日間の高値+安値） ÷ 2 ｝を(para2)日先にずらしたもの
data2 = np.c_[data2, np.zeros((len(data2), 1))]
for i in range(para3, len(data2) - para2):
    tmp_high = data2[i - para3 + 1:i + 1, 2].astype(np.float)
    tmp_low = data2[i - para3 + 1:i + 1, 3].astype(np.float)
    data2[i + para2, 12] = (np.max(tmp_high) + np.min(tmp_low)) / 2

# 25日ボリンジャーバンド（±1, 2シグマ）を追加します
parab = 25
data2 = np.c_[data2, np.zeros((len(data2),4))] # 列の追加
for i in range(parab, len(data2)):
    tmp = data2[i-parab+1:i+1,4].astype(np.float)
    data2[i,13] = np.mean(tmp) + 1.0* np.std(tmp)
    data2[i,14] = np.mean(tmp) - 1.0* np.std(tmp)
    data2[i,15] = np.mean(tmp) + 2.0* np.std(tmp)
    data2[i,16] = np.mean(tmp) - 2.0* np.std(tmp)

# # データの内容を見ます
# data_show=pd.DataFrame(data2)
# print(data_show)

# 説明変数となる行列Xを作成します
day_ago = 25 # 何日前までのデータを使用するのかを設定
num_sihyou = 1 + 4 + 4 +4 # 終値1本、MVave4本、itimoku4本、ボリンジャー4本

X = np.zeros((len(data2), day_ago*num_sihyou))

for s in range(0, num_sihyou): # 日にちごとに横向きに並べる
    for i in range(0, day_ago):
        X[i:len(data2),day_ago*s+i] = data2[0:len(data2)-i,s+4]

# 被説明変数となる Y = pre_minute後の終値-当日終値 を作成します
Y = np.zeros(len(data2))

# 何分後を値段の差を予測するのか決めます
pre_minute = int(sys.argv[1])
print("pre_minute: {}".format(pre_minute))
Y[0:len(Y)-pre_minute] = X[pre_minute:len(X),0] - X[0:len(X)-pre_minute,0]

# 【重要】X, Yを正規化します
original_X = np.copy(X) # コピーするときは、そのままイコールではダメ
tmp_mean = np.zeros(len(X))

for i in range(day_ago,len(X)):
    tmp_mean[i] = np.mean(original_X[i-day_ago+1:i+1,0]) # 25日分の平均値
    for j in range(0, X.shape[1]):
        X[i,j] = (X[i,j] - tmp_mean[i]) # Xを正規化
    Y[i] =  Y[i] # X同士の引き算しているので、Yはそのまま

# XとYを学習データとテストデータ(2017年～)に分ける
X_train = X[200:54206,:] # 200日平均を使うので、それ以降を学習データに使用します
Y_train = Y[200:54206]

X_test = X[54206:len(X)-pre_minute,:]
Y_test = Y[54206:len(Y)-pre_minute]

# 学習データを使用して、線形回帰モデルを作成します
from sklearn import linear_model # scikit-learnライブラリの関数を使用します
linear_reg_model = linear_model.LinearRegression()

linear_reg_model.fit(X_train, Y_train) # モデルに対して、学習データをフィットさせ係数を学習させます

# print("回帰式モデルの係数")
# print(linear_reg_model.intercept_)
# print(linear_reg_model.coef_)

# 2017年のデータで予想し、グラフで予測具合を見る

Y_pred = linear_reg_model.predict(X_test) # 予測する

result = pd.DataFrame(Y_pred) # 予測
result.columns = ['Y_pred']
result['Y_test'] = Y_test

sns.set_style('darkgrid')
sns.regplot(x='Y_pred', y='Y_test', data=result) #plotする


# 正答率を計算
success_num = 0
for i in range(len(Y_pred)):
    if Y_pred[i] * Y_test[i] >=0:
        success_num+=1

# 2017年の予測結果の合計を計算ーーーーーーーーー
# 前々日終値に比べて前日終値が高い場合は、買いとする
sum_2017 = 0

for i in range(0,len(Y_test)): # len()で要素数を取得しています
    if Y_pred[i] >= 0:
        sum_2017 += Y_test[i]
    else:
        sum_2017 -= Y_test[i]

print(Y_test)
print(Y_pred)
print("予測数："+ str(len(Y_pred))+"\t正解数："+str(success_num)+"\t正解率："+str(success_num/len(Y_pred)*100)+"\t利益合計：%1.3lf" %sum_2017)


# 予測結果の総和グラフを描くーーーーーーーーー
total_return = np.zeros(len(Y_test))

if Y_pred[0] >=0: # 2017年の初日を格納
    total_return[0] = Y_test[0]
else:
    total_return[0] = -Y_test[0]

for i in range(1, len(result)): # 2017年の2日以降を格納
    if Y_pred[i] >=0:
        total_return[i] = total_return[i-1] + Y_test[i]
    else:
        total_return[i] = total_return[i-1] - Y_test[i]

# plt.cla()
# plt.plot(total_return)
# plt.show()


