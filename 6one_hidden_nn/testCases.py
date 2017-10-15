#-*- coding: UTF-8 -*-
"""
# WANGZHE12
"""
import numpy as np

def load_planar_dataset():
    np.random.seed(1)
    m = 400  # 样本数量
    N = int(m / 2)  # 每个类别的样本量
    D = 2  # 维度数
    X = np.zeros((m, D))  # 初始化X
    Y = np.zeros((m, 1), dtype='uint8')  # 初始化Y
    a = 4  # 花儿的最大长度

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

