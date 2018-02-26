#-*- coding: UTF-8 -*-
"""
# WANGZHE12
"""

import numpy as np
import matplotlib.pyplot as plt


from init_utils import load_dataset, plot_decision_boundary, predict_dec


# % matplotlib
# inline
plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()

