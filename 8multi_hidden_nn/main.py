#-*- coding: UTF-8 -*-
"""
# WANGZHE12
"""

import scipy
from scipy import ndimage
from dnn_app_utils_v2 import *
from dnn_utils_v2 import load_dataset
from multi_hidden_function import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 设置随机数的seed，保证每次获取的随机数固定# 设置随机数的seed，保证每次获取的随机数固定
np.random.seed(1)

# 加载数据集
train_set_x_orig, train_y, test_set_x_orig, test_y, classes = load_dataset()
num_px = train_set_x_orig.shape[1]
# 对将每幅图像转为一个矢量
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 归一化
train_x = train_set_x_flatten/255.
test_x = test_set_x_flatten/255.

# 两层神经网络测试
# n_x = 12288     # num_px * num_px * 3
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)
# parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
# predictions_train = predict(train_x, train_y, parameters)
# predictions_test = predict(test_x, test_y, parameters)

# 多层神经网络测试
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
# 观察一下
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)

# 结果分析
print_mislabeled_images(classes, test_x, test_y, pred_test)

# 用自己的图片试试吧！
my_image = "my_image.jpg"  # change this to the name of your image file
my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))

my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((num_px * num_px * 3, 1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
    int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
