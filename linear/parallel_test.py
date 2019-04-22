# coding=utf-8
import numpy as np

a = np.arange(16).reshape(4, 4)

a_split = np.split(a, 2)

print(type(a_split))
# <class 'list'>

print(len(a_split))
# 2

print(a_split[0])
# [[0 1 2 3]
#  [4 5 6 7]]