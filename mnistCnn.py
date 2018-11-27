# -*- coding: utf-8 -*-
"""
Created on 2017/2/22

Using mnist data and the accuracy was about 0.85

@author: Mr.Fang
"""
from __future__ import absolute_import
from __future__ import print_function
import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
#数据预处理
#读取文件夹mnist下的42000张图片（灰度图）
def load_data():
    data = np.empty((42000,1,28,28),dtype="float32")
    label = np.empty((42000,),dtype="uint8")

    imgs = os.listdir("D:/ProjectPython/mnistCnn/mnist")
    num = len(imgs)
    for i in range(num):
        img = Image.open("D:/ProjectPython/mnistCnn/mnist/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0])
    return data,label
def load_testData():
    testImg = Image.open('testNum0.jpg')
    imgArr = np.asarray(testImg,dtype="float32")
    testData = np.empty((1,1,28,28),dtype="float32")
    testData[0,:,:,:] = imgArr
    return testData
if __name__ == '__main__':

    testData = load_testData()
            
    #保存模型数据的文件
    filename = 'mnistCnnTrain.h5'
    #加载数据
    data, label = load_data()
    print(data.shape[0], ' samples')
    
    #label为0~9共10个类别，转化格式
    label = np_utils.to_categorical(label, 10)
    
    ###############
    #开始建立CNN模型
    ###############
    
    #生成一个model
    model = Sequential()
    
    #第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
    #激活函数用tanh
    #你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))
    model.add(Conv2D(8, (3, 3), padding='valid', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    
    #第二个卷积层，8个卷积核，每个卷积核大小3*3。4表示输入的特征图个数，等于上一层的卷积核个数
    model.add(Conv2D(8, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #第三个卷积层，16个卷积核，每个卷积核大小3*3
    model.add(Conv2D(16, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #全连接层，先将前一层输出的二维特征图flatten为一维的。
    #Dense就是隐藏层。16就是上一层输出的特征图个数。4是根据每个卷积层计算出来的：(28-5+1)得到24,(24-3+1)/2得到11，(11-3+1)/2得到4
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    
    #Softmax分类，输出是10类别
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    #############
    #开始训练模型
    ##############
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',class_mode="categorical",metrics=["accuracy"])
    
    #validation_split=0.2，将20%的数据作为验证集。
    
    f = open(filename)
    if f == None:
        print('Training................')
        model.fit(data, label, batch_size=100,epoch=1,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.2)
        model.save(filename)
    else:
        print('Loading Data............')
        # model.load_weights(filename)
    #数据测试
    print(model.predict(testData))