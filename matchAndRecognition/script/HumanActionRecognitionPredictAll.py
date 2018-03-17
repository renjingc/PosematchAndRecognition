# coding=utf-8
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers,regularizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import EigenvalueRegularizer
from numpy.random import permutation
from keras.optimizers import SGD
import pandas as pd
import datetime
import glob
import cv2
import math
import pickle
from collections import OrderedDict
from keras import backend as K

LabelName=['双手平举', '弯腰', '行走', '半蹲', '单手挥手', '侧身舒展', '叉腰', '趴地', '打电话', '两人交流']
#输入最终的权重文件
# Enter here the path for storage of the whole model weights (VGG16+top classifier model):
whole_model_weights_path = 'E:\\work\\finish26\\predict\\predict\\whole_model.h5'
#测试文件夹
# Enter here the name of the folder where is the test images (the data evalueted in the private leaderboard):
test_data_dir = 'E:\\work\\finish26\\predict\\predict\\data'
#测试文件路径
test_images_path = 'E:\\work\\finish26\\predict\\predict\\data'

# Enter here the features of the data set:
#图像大小
img_width, img_height = 224, 224
#测试样本数量
nb_test_samples = 1002
#颜色类型
color_type_global = 3

# You can set larger values here, according with the memory of your GPU:
#一次训练样本数量
batch_size = 32


#构建vgg16网络
# build the VGG16 network:
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#构建自己的分类器，卷积模型
# building a classifier model on top of the convolutional model:
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(64, activation='relu', W_regularizer=EigenvalueRegularizer(10)))
top_model.add(Dense(10, activation='softmax', W_regularizer=EigenvalueRegularizer(10)))

#加入模型
# add the model on top of the convolutional base
model.add(top_model)

model.load_weights(whole_model_weights_path)
print('Model loaded.')

#测试的数据
test_datagen = ImageDataGenerator()
  
#开始测试
print "testing"
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

#预测测试样本
aux = model.predict_generator(test_generator, nb_test_samples)
#每个测试样本的预测值
predictions = np.zeros((nb_test_samples, 10))

#重新排列预测
# Rearranging the predictions:

'''
ord = [5, 0, 6, 2, 7, 9, 1, 4, 8, 3]
for n in range(10):
    i = ord[n]
    print(i)
    print(aux[:, i])
    predictions[:, n] = aux[:, i]
'''

for n in range(10):
    #print(i)
    print(aux[:, n])
    predictions[:, n] = aux[:, n]

#改进多类对数损失
# Trick to improve the multi-class logarithmic loss (the evaluation metric of state-farm-distracted-driver-detection from Keras):

predictions = 0.985 * predictions + 0.015

#读取图片
def get_im(path, img_width, img_height, color_type=1):
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_height, img_width))
    return resized

#读取测试图片
def load_test(img_width, img_height, color_type=1):
    print('Read test images')	 
    pathDir =  os.listdir(test_images_path)
    X_test = []
    X_test_id = []
    pathDir.sort()
 
    for allDir in pathDir:
	path = os.path.join(test_images_path, allDir, '*.bmp')
	files = glob.glob(path)
	files.sort()
	total = 0
	thr = math.floor(len(files)/10)
	for fl in files:
		#文件名
		flbase = os.path.basename(fl)
		#读取图片
		img = get_im(fl, img_width, img_height, color_type)
		#加入test
		X_test.append(img)
		#id为文件名
		X_test_id.append(flbase)
		#总数加+1
		total += 1
		if total % thr == 0:
			print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id

def judgeLabel(predictions):
    labelList=[]

    for i in range(len(predictions)):
        max=0
        label=0
        for j in range(10):
            if predictions[i,j]>max:
                max=predictions[i,j]
                label=j
        labelList.append(label)
    return labelList

#保存预测数据
def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    labelList=judgeLabel(predictions)
    print labelList
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    result1.loc[:, 'label'] = np.array(labelList)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'out_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


X_test, test_id = load_test(img_width, img_height, color_type_global)
create_submission(predictions, test_id)
print "finish"