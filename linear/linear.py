# import関連
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore') # 実行上問題ない注意は非表示にする

# 練習問題: 問題を作成
from numpy.random import * #乱数のライブラリをimport
X_train = rand(100,2) # 0〜1の乱数で 100行2列の行列を生成
X_test = rand(100,2) # 0〜1の乱数で 100行2列の行列を生成

# 係数を設定
w1 = 1.0
w2 = 2.0
b = 3.0

# モデルからの誤差となるノイズを作成
noise_train = 0.1*randn(100)
noise_test = 0.1*randn(100)

Y_train = w1*X_train[:,0] + w2*X_train[:,1]  + b + noise_train
Y_test = w1*X_test[:,0] + w2*X_test[:,1] + b + noise_test

# 練習問題: 問題を線形回帰モデルで解く
from sklearn import linear_model # scikit-learnライブラリの関数を使用します

linear_reg_model = linear_model.LinearRegression() # モデルの定義

linear_reg_model.fit(X_train, Y_train) # モデルに対して、学習データをフィットさせ係数を学習させます

print("回帰式モデルの係数")
print(linear_reg_model.coef_)
print(linear_reg_model.intercept_)

# グラフで予測具合を見る
Y_pred = linear_reg_model.predict(X_test) # テストデータから予測してみる

result = pd.DataFrame(Y_pred) # 予測
result.columns = ['Y_pred']
result['Y_test'] = Y_test
sns.set_style('darkgrid')
sns.regplot(x='Y_pred', y='Y_test', data=result)