'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
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


batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28
# re_img_rows, re_img_cols = 14, 14  # set下采样
# re_img_rows, re_img_cols = 28, 14  # set抛弃一半
re_img_rows, re_img_cols = 28, 28  # 大小不变，数据量减小
# ------------------------------------------------------------------------------------
# 数据预处理
# the data, split between train and test sets
(x_train_orig, y_train_orig), (x_test_orig, y_test) = mnist.load_data()
# rex_train = np.zeros((60000, 14, 14)) # set下采样
# rex_test = np.zeros((10000, 14, 14))  # set下采样
# rex_train = np.zeros((60000, 28, 14))  # set抛弃一半
# rex_test = np.zeros((10000, 28, 14))  # set抛弃一半
# rex_train = np.zeros((6000, 28, 28))  # set数据量1/10
rex_test = x_test_orig  # 原始test数据
rex_train = x_train_orig  # 原式train数据
y_train = y_train_orig  # 原始y_train数据
# y_train = np.zeros(6000)
# x_test = rex_test
# x_train = rex_train
x_test = x_test_orig  # 测试集使用原始数据
print('x_test_shape:', x_test_orig.shape)
print('y_test:', y_test)
print('rex_train', rex_train.shape)
# ------------------------------------------------------------------------------------
# 原式训练数据裁剪，裁剪为1/10大小
'''for i in range(int(len(x_train_orig)/10)):
    x_train[i] = x_train_orig[i]
# 改变y_train裁剪。
for i in range(len(y_train)):
    y_train[i] = y_train_orig[i]'''


# ------------------------------------------------------------------------------------
# 直接抛弃降采样
'''for i in range(len(x_train_orig)):
    a = x_train_orig[i]
    for j in range(len(a)):
        for k in range(int(len(a) / 2)):
            rex_train[i][j][k] = a[j][k]
    # print(x_train[i:, 0, 0])
print('rex_train_simple', rex_train.shape)

for i in range(len(x_test_orig)):
    a = x_test_orig[i]
    for j in range(len(a)):
        for k in range(int(len(a) / 2)):
            # print(i, j, k)
            rex_test[i][j][k] = a[j][k]
    # print(x_train[i:, 0, 0])
print('rex_test_simple', rex_test.shape)'''

# ------------------------------------------------------------------------------------
# 输出样例测试
print('rex_test_simple_view', rex_test[0])
print('rex_train_example_view', rex_train[0])
# ------------------------------------------------------------------------------------
# 进行降采样(隔行删除)
'''for i in range(len(x_train_orig)):
    a = x_train_orig[i]
    for j in range(len(a)):
        if j % 2 == 1:
            continue
        for k in range(len(a)):
            if k % 2 == 1:
                continue
            rex_train[i][int(j/2)][int(k/2)] = a[j][k]
    # print(x_train[i:, 0, 0])
print('rex_train_simple', rex_train.shape)
# 对测试数据降采样
for i in range(len(x_test_orig)):
    a = x_test_orig[i]
    for j in range(len(a)):
        if j % 2 == 1:
            continue
        for k in range(len(a)):
            if k % 2 == 1:
                continue
            rex_test[i][int(j/2)][int(k/2)] = a[j][k]
    # print(x_train[i:, 0, 0])
print('rex_test_simple', rex_test.shape)'''


# ---------------------------------------------------------------------------------------
# 设置通道在后，适应tensorflow
if K.image_data_format() == 'channels_first':
    '''x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)'''
else:
    print('channels_last')
    '''x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)'''
    x_train = rex_train.reshape(rex_train.shape[0], re_img_rows, re_img_cols, 1)
    x_test = rex_test.reshape(x_test.shape[0], re_img_rows, re_img_cols, 1)
    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (re_img_rows, re_img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# -----------------
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# -------------------------------------------------------------------------------------
# 定义卷积层
conv_1 = Conv2D(32, kernel_size=(3, 3),
                 input_shape=input_shape,
                name='Conv_1')
conv_2 = Conv2D(64, (3, 3), activation='relu', name='Conv_2')

model = Sequential()
model.add(conv_1)
model.add(Activation('relu', name='Activation_1'))
model.add(conv_2)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
outputs = conv_1.output
print(outputs)
# ------------------------------------------------------------------------------------------
# 编译操作
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# -------------------------------------------------------------------------------------------
# 执行卷积操作
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))
# --------------------------------------------------------------------------------------------
# 保存model模型
# 保存神经网络的结构与训练好的参数
'''json_string = model.to_json()  # 等价于 json_string = model.get_config()
open('my_model_architecture.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5')'''

# --------------------------------------------------------------------------------------------
# 对中间层输出（第一个卷积层）
'''dense1_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('Conv_1').output)
dense1_output = dense1_layer_model.predict(x_train)  # 以这个model的预测值作为输出
print(dense1_output.shape)  # (60000, 26, 26, 32)
for i in range(10):
    a = dense1_output[0, :, :, i]
    plt.imshow(a)
    plt.title('conv layer')
    pyplot.show()
# ----------------
# 中间结果输出（激励层）
dense1_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('Activation_1').output)
dense1_output = dense1_layer_model.predict(x_train)  # 以这个model的预测值作为输出
print(dense1_output.shape)  # (60000, 26, 26, 32)
for i in range(10):
    a = dense1_output[0, :, :, i]
    plt.imshow(a)
    plt.title('activation layer')
    pyplot.show()'''
# --------------------------------------------------------------------------------------------------------
# 计算结果
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# --------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 预测最可能是8的测试图像。
a = model.predict(x_test)
print(a.shape)
np.savetxt('predict_value', a)
temp = -100
loc = 0
print(a)
for i in range(len(a)):
    if a[i][8] > temp:
        temp = a[i][8]
        loc = i
print('最可能是8的图:\n', x_test_orig[loc])
print('可能性:', temp)
print('位置', loc)
plt.imshow(x_test_orig[loc])
plt.title('most likely number 8')
pyplot.show()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# 模拟9 欺骗DNN
# sim_value = np.zeros((1, 784))
print(x_test_orig.shape)
sim_value = np.loadtxt('cnn_loc_7114.txt')
# sim_value = np.loadtxt('dnn_loc_5003.txt')  # 使用dnn 的测试样例
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
