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

#输入训练模型权重文件，即原始的vgg16权重文件
# Enter here the path to the model weights files:
weights_path = 'E:\\work\\finish26\\predict\\predict\\vgg16_weights.h5'
# 输入顶层模型权重文件，修改的权重文件
# Enter here the path to the top-model weights files:
top_model_weights_path = 'E:\\work\\finish26\\predict\\predict\\fc_model.h5'
#输入最终的权重文件
# Enter here the path for storage of the whole model weights (VGG16+top classifier model):
whole_model_weights_path = 'E:\\work\\finish26\\predict\\predict\\whole_model.h5'
#输入训练文件夹
# Enter here the name of the folder that contains the folders c0, c1,..., c9, with the training images belonging to classes 0 to 9:
train_data_dir = 'E:\\work\\finish26\\predict\\predict\\train'
#测试文件夹
# Enter here the name of the folder where is the test images (the data evalueted in the private leaderboard):
test_data_dir = 'E:\\work\\finish26\\predict\\predict\\test'
#测试文件路径
test_images_path = 'E:\\work\\finish26\\predict\\predict\\test'

# Enter here the features of the data set:
#图像大小
img_width, img_height = 224, 224
#训练样本数量
nb_train_samples = 902
#测试样本数量
nb_test_samples = 100
#颜色类型
color_type_global = 3

# You can set larger values here, according with the memory of your GPU:
#一次训练样本数量
batch_size = 32

# Enter here the number of training epochs (with 80 epochs the model was positioned among
# the 29% best competitors in the private leaderboard of state-farm-distracted-driver-detection)
# According to our results, this model can achieve a better performance if trained along a larger 
# number of epochs, due to the agressive regularization with Eigenvalue Decay that was adopted.
#训练迭代次数
nb_epoch = 80

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

# loading the weights of the pre-trained VGG16:
#载入原始的vgg16权重文件
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

#构建自己的分类器，卷积模型
# building a classifier model on top of the convolutional model:
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(64, activation='relu', W_regularizer=EigenvalueRegularizer(10)))
top_model.add(Dense(10, activation='softmax', W_regularizer=EigenvalueRegularizer(10)))
top_model.load_weights(top_model_weights_path)

#加入模型
# add the model on top of the convolutional base
model.add(top_model)

#设置前面的15层不训练，权重不改变
# setting the first 15 layers to non-trainable (the original weights will not be updated)    
for layer in model.layers[:15]:
    layer.trainable = False

#使用SGD优化方式
# Compiling the model with a SGD/momentum optimizer:
model.compile(loss = "categorical_crossentropy",
              optimizer=optimizers.SGD(lr=1e-6, momentum=0.9),
              metrics=['mean_squared_logarithmic_error', 'accuracy'])

# Data augmentation:
#随机变换堆数据进行提升，这样我们的模型将看不到任何两张完全相同的图片
#有利于我们抑制过拟合，使得模型的泛化能力更好
'''
rotation_range是一个0~180的度数，用来指定随机选择图片的角度。
width_shift和height_shift用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例。
rescale值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
shear_range是用来进行剪切变换的程度，参考剪切变换
zoom_range用来进行随机的放大
horizontal_flip随机的堆图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
fill_mode用来指定当需要进行像素填充，如旋转，水平和竖直位移时，如何填充新出现的像素
'''
#训练的数据
train_datagen = ImageDataGenerator(shear_range=0.3, zoom_range=0.3, rotation_range=0.3)
#测试的数据
test_datagen = ImageDataGenerator()

#开始训练
print "trainnin"
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')
  
#开始测试
print "testing"
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

#训练数据排序
class_dictionary = train_generator.class_indices
sorted_class_dictionary = OrderedDict(sorted(class_dictionary.items()))
sorted_class_dictionary = sorted_class_dictionary.values()
print(sorted_class_dictionary)

#fit_generator：用于从Python生成器中训练网络
# Fine-tuning the model:
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=train_generator,
        nb_val_samples=nb_train_samples)

#保存总的样本权重        
model.save_weights(whole_model_weights_path)
print "save weights"

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

X_test, test_id = load_test(img_width, img_height, color_type_global)

#保存预测数据
def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

create_submission(predictions, test_id)
print "finish"
