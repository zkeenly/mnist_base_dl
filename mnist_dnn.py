import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling1D, Dropout, MaxPooling2D
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg # mpimg 用于读取图片
from keras.optimizers import RMSprop  # 优化方法,加速神经网络训练

(x_train_orig, y_train_orig), (x_test_orig, y_test) = mnist.load_data()  # 获取训练/测试数据集
# print(x_test_orig[0])  # debug
re_img_rows, re_img_cols = 28, 28  # 大小不变
# input_dim = 392  # 使用抛弃一半
# input_dim = 196  # 使用隔行删除
input_dim = 784
input_shape = (re_img_rows, re_img_cols, 1)  # channel = 1
# -----------------------------------------------------------------------------------------
# 下采样
# X_train = np.zeros((60000, 14, 14))  # set下采样
# X_test = np.zeros((10000, 14, 14))  # set下采样
# X_train = np.zeros((60000, 28, 14))  # set抛弃一半
# X_test = np.zeros((10000, 28, 14))  # set抛弃一半
# X_train = np.zeros((6000, 28, 28))  # set减少训练数据集
# y_train = np.zeros(6000)
X_train = x_train_orig   # 保持训练数据集不变
y_train = y_train_orig  # 保持训练数据集不变
X_test = x_test_orig  # 测试数据集不变
# ------------------------------------------------------------------------------------
# 原式训练数据裁剪，裁剪为1/10大小
'''for i in range(int(len(x_train_orig)/10)):
    X_train[i] = x_train_orig[i]
# 改变y_train裁剪。
for i in range(len(y_train)):
    y_train[i] = y_train_orig[i]'''

# -----------------------------------------------------------------------------------------
# 直接抛弃降采样
'''for i in range(len(x_train_orig)):
    a = x_train_orig[i]
    for j in range(len(a)):
        for k in range(int(len(a) / 2)):
            X_train[i][j][k] = a[j][k]
    # print(x_train[i:, 0, 0])
print('rex_train_simple', X_train.shape)

for i in range(len(x_test_orig)):
    a = x_test_orig[i]
    for j in range(len(a)):
        for k in range(int(len(a) / 2)):
            # print(i, j, k)
            X_test[i][j][k] = a[j][k]
    # print(x_train[i:, 0, 0])
print('rex_test_simple', X_test.shape)'''

# -----------------------------------------------------------------------------------------
# 进行降采样(隔行删除)
'''for i in range(len(x_train_orig)):
    a = x_train_orig[i]
    for j in range(len(a)):
        if j % 2 == 1:
            continue
        for k in range(len(a)):
            if k % 2 == 1:
                continue
            X_train[i][int(j/2)][int(k/2)] = a[j][k]
    # print(x_train[i:, 0, 0])
print('rex_train_simple', X_train.shape)
# 对测试数据降采样
for i in range(len(x_test_orig)):
    a = x_test_orig[i]
    for j in range(len(a)):
        if j % 2 == 1:
            continue
        for k in range(len(a)):
            if k % 2 == 1:
                continue
            X_test[i][int(j/2)][int(k/2)] = a[j][k]
    # print(x_train[i:, 0, 0])
print('rex_test_simple', X_test.shape)'''

# ------------------------------------------------------------------------------------
# 数据预处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255  # 将数据标准化为0-1
X_test = X_test.reshape(X_test.shape[0], -1) / 255
y_train = np_utils.to_categorical(y_train, 10)  # 将数字转换为为二进制位形式，例如1 ->[0. 1. 0. 0. ... 0.]
y_test = np_utils.to_categorical(y_test, 10)
# -----------------------------------------------------------------------------------------
# 创建全连接层参数
model = Sequential([
    # MaxPooling2D(pool_size=(2, 2), input_shape=input_shape),
    # Dense(32, input_dim=(input_dim / 4), name='Dense_1'),
    Dense(32, input_dim=input_dim, name='Dense_1'),  # 全连接层。32个神经元，输入维度为784（28*28）
    Dropout(0.40),
    Activation('relu', name='Activation_1'),  # 激活层
    # MaxPooling1D(pool_size=2, strides=None, padding='valid'),
    # Dropout(0.25),
    Dense(10),  # 全连接层，10个神经元
    Activation('softmax')  # 激活层
])

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
print('training...')

model.fit(X_train, y_train, nb_epoch=1, batch_size=32)



print('testing...')

loss, accuracy = model.evaluate(X_test, y_test)  # 测试数据集
print('test loss:', loss)
print('test accuracy: ', accuracy)


# ------------------------------------------------------------------------------
# 预测最可能是8的测试图像。
print('X_test shape', X_test.shape)
a = model.predict(X_test)
print('a shape', a.shape)
np.savetxt('predict_value', a)
temp = -100
loc = 0
print(a)
for i in range(len(a)):
    if a[i][8] > temp:
        temp = a[i][8]
        loc = i
print('最可能是8的图:\n', x_test_orig[loc])
# print(X_test[loc])
# np.savetxt('value.txt', x_test_orig[loc])
print('可能性:', temp)
print('位置', loc)
plt.imshow(x_test_orig[loc])
plt.title('most likely number 8')
pyplot.show()
# ------------------------------------------------------------------------------
# 模拟9 欺骗DNN
# sim_value = np.zeros((1, 784))
print(x_test_orig.shape)
sim_value = np.loadtxt('dnn_loc_5003.txt')
# print('array', array)
print(sim_value.shape)
re_sim_value = sim_value.reshape((1, 784)) / 255
# list_of_lists = []
# print(list_of_lists)
a = model.predict(re_sim_value)
print('预测值的概率', a)
max_index = np.unravel_index(a.argmax(), a.shape)
print('最可能的值：', max_index[1])
plt.imshow(sim_value)
plt.title('simulation 9')
pyplot.show()
# ------------------------------------------------------------------------------
