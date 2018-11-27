from __future__ import print_function
import keras
import h5py
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
from keras.layers import Activation
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from theano.printing import Print
from keras.models import model_from_json

model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

re_img_rows, re_img_cols = 28, 28  # 大小不变，数据量减小
(x_train_orig, y_train_orig), (x_test_orig, y_test) = mnist.load_data()
# ------------------------------------------------------------------------------
# 模拟9 欺骗DNN

print(x_test_orig.shape)
# sim_value = np.loadtxt('cnn_loc_test.txt')
#sim_value = np.loadtxt('cnn_loc_7114.txt')
sim_value = np.loadtxt('dnn_loc_5003.txt')
# print('array', array)
print(sim_value.shape)
re_sim_value = sim_value.reshape((1, re_img_rows, re_img_cols, 1))
# re_sim_value = sim_value.reshape((1, 784)) / 255
# list_of_lists = []
# print(list_of_lists)
a = model.predict(re_sim_value)
print('预测值的概率:', a)
max_index = np.unravel_index(a.argmax(), a.shape)
print('最可能的值：', max_index[1])
plt.imshow(sim_value)
plt.title('simulation cnn 9')
pyplot.show()
# ------------------------------------------------------------------------------
