# import関連
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore') # 実行上問題ない注意は非表示にします

# %matplotlib inline

# dataフォルダの場所を各自指定してください
data_dir = "../data/"

# FXデータの読み込み
data = pd.read_csv(data_dir + "USDJPY_day_1997_2017.csv")

# Close-Openをデータに追加します
data['Change'] = data.Close - data.Open
data.head() # データの概要を見てみます

# 2016年のデータを取り出します
data16 = data.iloc[4935:5193,:] # pythonは0番目からindexが始まります

# 2017年のデータを取り出します
data17 = data.iloc[5193:,:]
data17.head()

# 2016年の合計を計算する
# 前々日終値に比べて前日終値が高い場合は、買い、低い場合は売りで入ります
sum_2016 = 0
for i in range(2,len(data16)): # len()で要素数を取得しています
    if data16.iloc[i-2,4] <= data16.iloc[i-1,4]:
        sum_2016 += data16.iloc[i,5]
    else:
        sum_2016 -= data16.iloc[i,5]

print("2016年の利益合計：%1.3lf" %sum_2016) # 2016年の利益合計

# 2017年の合計を計算する
# 前々日終値に比べて前日終値が高い場合は、買い、低い場合は売りで入ります
sum_2017 = 0
for i in range(2,len(data17)): # len()で要素数を取得しています
    if data17.iloc[i-2,4] <= data17.iloc[i-1,4]:
        sum_2017 += data17.iloc[i,5]
    else:
        sum_2017 -= data17.iloc[i,5]

print("2017年の利益合計：%1.3lf" %sum_2017) # 2017年の利益合計

# 2016年のデータをプロットしてみます
plt.style.use('seaborn-darkgrid')
plt.plot(data16['Close'])
plt.ylim([95,125])
plt.show()

# 2017年からのデータをプロットしてみます
plt.plot(data17['Close'])
plt.ylim([95,125])
plt.show()
